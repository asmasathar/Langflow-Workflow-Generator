from __future__ import annotations

import json
import os
import re
import string
import random
from typing import Any, Dict, List, Optional, Set

from knowledge_fetcher import KnowledgeFetcher
from shared_memory import SharedMemory
from component_resolver import ComponentResolver
from langflow_guidelines import (
    LANGFLOW_GENERAL_RULES,
    LANGFLOW_EDGE_PATTERNS,
    LANGFLOW_COMPONENT_FIELD_GUIDE,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rand(n: int = 5) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _extract_template_vars(template: str) -> List[str]:
    return re.findall(r"\{([^}]+)\}", template or "")


def _is_placeholder_type(t: str) -> bool:
    return _norm(t) in {
        "", "typeid", "workflowtype", "workflowtypeid",
        "componenttype", "componentid", "exactworkflowtype",
    }


def _no_questions() -> bool:
    return os.getenv("WF_NO_QUESTIONS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _extract_json_block(text: str) -> str:
    """Extract the last valid JSON object or array from LLM output."""
    text = (text or "").strip()
    text = re.sub(r"```(?:json)?|```", "", text, flags=re.IGNORECASE).strip()

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    candidates: List[str] = []
    stack: List[str] = []
    start_idx: Optional[int] = None
    in_str = escape = False

    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "[{":
            if not stack:
                start_idx = i
            stack.append(ch)
        elif ch in "]}" and stack:
            opener = stack[-1]
            if (opener == "[" and ch == "]") or (opener == "{" and ch == "}"):
                stack.pop()
                if not stack and start_idx is not None:
                    candidates.append(text[start_idx: i + 1])
                    start_idx = None

    for cand in reversed(candidates):
        try:
            return json.dumps(json.loads(cand))
        except Exception:
            continue
    return text




# ── PlanningAgent ──────────────────────────────────────────────────────────────

class PlanningAgent:
    """
    Conversational agent that:
      - Asks user what to build
      - Asks targeted clarifying questions
      - Selects the right Langflow components
      - Asks user to fill required fields per component
      - Enforces prompt template variables for downstream connectivity
    """

    def __init__(
        self,
        mem: SharedMemory,
        fetcher: Optional[KnowledgeFetcher] = None,
        rag: Optional[Any] = None,
    ):
        self.mem = mem
        self.fetcher = fetcher
        self.rag = rag
        self._rag_snippets: str = ""
        # Store fetcher/rag refs in mem so other agents can use them
        mem._fetcher = fetcher  # type: ignore[attr-defined]
        mem._rag = rag          # type: ignore[attr-defined]

    # ── Entry point ────────────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "=" * 60)
        print("  PLANNING AGENT — Workflow Design & Component Selection")
        print("=" * 60)

        # Seed RAG context from initial prompt if available
        if self.rag and getattr(self.rag, "is_built", False):
            if self.mem.user_prompt:
                snippets = self.rag.query(self.mem.user_prompt, top_k=5)
                if snippets:
                    self._rag_snippets = snippets
                    self.mem.mark_knowledge_use("planning", "rag_context", 1)
                    print("[Planning] Retrieved initial RAG context.")

        # Phase 1: Get initial description and classify
        self._understand()

        # Phase 2: Ask clarifying questions
        if _no_questions():
            print("[Planning] Non-interactive mode: skipping clarification questions.")
            self._extract_requirements_from_conversation()
        else:
            self._clarify()

        # Phase 3: Select components from catalog
        self._select_components()
        if not self.mem.selected_components:
            self.mem.errors.append("Component selection returned empty.")
            return

        # Phase 4: Ask user to fill required fields per component
        if _no_questions():
            print("[Planning] Non-interactive mode: auto-filling component fields.")
            self._llm_fill_remaining_params()
        else:
            self._fill_component_fields_interactively()

        # Phase 5: Enforce prompt template variables for connectivity
        self._enforce_connectivity_from_content()

        # Phase 6: Store prompt variable summary for Graph agent
        self._store_prompt_variables()

        print(f"\n[Planning] Done: {len(self.mem.selected_components)} components selected and configured.")
        self._print_component_summary()

    # ── Phase 1: Understand ────────────────────────────────────────────────

    def _understand(self) -> None:
        print("\nHello! I'm your Langflow workflow designer.")
        print("I'll help you create a perfect workflow step by step.\n")

        if not self.mem.user_prompt:
            if _no_questions():
                raise ValueError("No workflow description provided. Use --prompt with --no-questions.")
            self.mem.user_prompt = input(
                "What kind of workflow do you want to build?\n"
                "(Describe what it should do, e.g. 'memory chatbot', 'RAG Q&A', 'agent with tools')\n> "
            ).strip()
            if not self.mem.user_prompt:
                raise ValueError("No workflow description provided.")

        self.mem.add_message("user", self.mem.user_prompt)
        print(f"\n[Planning] Analyzing: {self.mem.user_prompt}")

        # Re-query RAG with initial prompt
        if self.rag and getattr(self.rag, "is_built", False):
            self._rag_snippets = self.rag.query(self.mem.user_prompt, top_k=6) or ""

        examples_list = "\n".join(
            f"- {ex.get('name','')}: {(ex.get('description') or '')[:120]}"
            for ex in self.mem.example_flows[:30]
        )
        source_hint = ", ".join(self.mem.get_source_types()[:6]) or "ChatInput, TextInput"
        sink_hint   = ", ".join(self.mem.get_sink_types()[:6])   or "ChatOutput, SaveToFile"

        system = f"""You are a Langflow workflow architect.

{LANGFLOW_GENERAL_RULES}

Available source components: {source_hint}
Available sink components: {sink_hint}

Example workflows in catalog:
{examples_list}

Relevant RAG context:
{self._rag_snippets[:1500] if self._rag_snippets else "(none)"}

Analyze the user request and return JSON ONLY:
{{
  "workflow_type": "short_label (e.g. memory_chatbot, rag_pipeline, agent_with_tools)",
  "needs_source_sink": true,
  "understood_requirements": {{}},
  "missing_info": ["specific gaps still needed"]
}}"""

        resp = self.mem.call_llm([
            {"role": "system", "content": system},
            {"role": "user", "content": f"User wants: {self.mem.user_prompt}"},
        ], temperature=0.1, label="understand")

        try:
            analysis = json.loads(_extract_json_block(resp))
            self.mem.requirements["workflow_type"]     = analysis.get("workflow_type", "workflow")
            self.mem.requirements["needs_source_sink"] = analysis.get("needs_source_sink", True)
            self.mem.requirements.update(analysis.get("understood_requirements", {}))
            self.mem.requirements["missing_info"]      = analysis.get("missing_info", [])
        except Exception:
            self.mem.requirements.setdefault("workflow_type", "workflow")
            self.mem.requirements.setdefault("needs_source_sink", True)
            self.mem.requirements.setdefault("missing_info", [])

        wf_type = self.mem.requirements.get("workflow_type", "")
        print(f"[Planning] Workflow type: {wf_type}")

        # Refresh RAG using detected workflow_type
        if self.rag and getattr(self.rag, "is_built", False) and wf_type:
            deep = self.rag.query(
                f"workflow_type={wf_type} {self.mem.user_prompt}", top_k=5
            )
            if deep:
                self._rag_snippets = (self._rag_snippets or "") + "\n\n=== DEEP ===\n" + deep
                self.mem.mark_knowledge_use("planning", "rag_context", 1)

    # ── Phase 2: Clarify ───────────────────────────────────────────────────

    def _clarify(self) -> None:
        """Ask targeted questions until the LLM decides it has enough information."""
        print("\n[Planning] Gathering requirements...\n")

        for round_num in range(7):
            question = self._next_clarification_question(round_num)
            if not question:
                break

            print(question)
            answer = input("> ").strip()
            if not answer:
                continue

            self.mem.add_message("assistant", question)
            self.mem.add_message("user", answer)
            self._integrate_answer(question, answer)

        print("\n[Planning] Requirements complete.")
        self._extract_requirements_from_conversation()

    def _next_clarification_question(self, round_num: int) -> Optional[str]:
        """
        Ask the LLM whether more information is needed.
        The LLM replies SUFFICIENT when it has enough to build the workflow,
        or with a single clarifying question when information is missing.
        No hardcoded done-phrases — the model reads the conversation and decides.
        """
        conv = self._fmt_conv(2000)

        system = f"""You are interviewing a user to build a Langflow workflow.
Workflow type: {self.mem.requirements.get('workflow_type', 'unknown')}

Requirements collected so far:
{json.dumps(self.mem.requirements, indent=2)[:600]}

Full conversation so far:
{conv}

Decide if you have enough information to proceed, or if one specific thing is still missing.

Reply SUFFICIENT (and ONLY that word) when:
- The user has indicated they are done, satisfied, or want to proceed
- The user has answered all critical questions
- You have enough to configure every component in the workflow
- Further questions would be redundant

Otherwise ask ONE specific, concise question about the single most important missing detail.
Ask about (in priority order):
1. LLM provider + model name + API key environment variable name
2. System prompt / agent behavior
3. Whether memory/history is needed
4. Whether file/document/web input is needed
5. Whether specific tools or APIs are needed

This is round {round_num + 1} of up to 6. Do not ask questions already answered."""

        resp = self.mem.call_llm([
            {"role": "system", "content": system},
            {"role": "user", "content": "Next question or SUFFICIENT?"},
        ], temperature=0.0, max_tokens=180, label="clarify")

        r = resp.strip()
        if "SUFFICIENT" in r.upper():
            return None
        if round_num >= 5:
            return None
        return r

    def _integrate_answer(self, question: str, answer: str) -> None:
        system = (
            "Update requirements JSON with the new Q&A info. "
            "Return ONLY valid JSON, no markdown, no explanation."
        )
        prompt = (
            f"Current requirements:\n{json.dumps(self.mem.requirements, indent=2)[:700]}\n\n"
            f"Q: {question}\nA: {answer}\n\nUpdated requirements JSON:"
        )
        resp = self.mem.call_llm([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ], temperature=0.1, max_tokens=512, label="integrate")
        try:
            updated = json.loads(_extract_json_block(resp))
            if isinstance(updated, dict):
                self.mem.requirements.update(updated)
        except Exception:
            key = re.sub(r"[^a-z0-9_]", "_", question.lower()[:40])
            self.mem.requirements[key] = answer

    def _extract_requirements_from_conversation(self) -> None:
        """
        Deterministically extract known knobs from conversation history.
        Latest user statement wins.
        """
        user_msgs = [
            m.get("content", "")
            for m in self.mem.conversation_history
            if m.get("role") == "user"
        ]

        provider = model = api_env = system_prompt = ""

        for msg in user_msgs:
            low = msg.lower()
            if "openrouter" in low:
                provider = "OpenRouter"
            elif "openai" in low:
                provider = "OpenAI"
            elif "anthropic" in low or "claude" in low:
                provider = "Anthropic"
            elif "ollama" in low:
                provider = "Ollama"

            m_match = re.search(
                r"\b([a-z0-9][a-z0-9_./:-]{2,}(?:qwen|llama|gpt|claude|mistral|gemini|falcon)[a-z0-9_./:-]*)\b",
                low,
            )
            if m_match:
                candidate = m_match.group(1).strip()
                if len(candidate) >= 4:
                    model = candidate

            k_match = re.search(r"\b([A-Z][A-Z0-9_]{3,}(?:KEY|TOKEN|SECRET))\b", msg)
            if k_match:
                api_env = k_match.group(1)

            sp_match = re.search(
                r"system\s*prompt\s*(?:as|is|=|:)?\s*['\"]?([^'\"\n]{5,})['\"]?",
                msg, flags=re.IGNORECASE,
            )
            if sp_match:
                candidate = sp_match.group(1).strip()
                if candidate:
                    system_prompt = candidate

        if provider:    self.mem.requirements["provider"] = provider
        if model:       self.mem.requirements["model_name"] = model
        if api_env:     self.mem.requirements["api_key_env_var"] = api_env
        if system_prompt: self.mem.requirements["system_prompt"] = system_prompt

    # ── Phase 3: Select components ─────────────────────────────────────────

    def _select_components(self) -> None:
        print("\n[Planning] Selecting components from live catalog...")

        catalog_text = self._build_catalog_text()
        best_ex = self._find_best_example()
        anchor  = self._format_example(best_ex)
        reqs    = json.dumps(self.mem.requirements, indent=2)
        conv    = self._fmt_conv(1800)

        source_types = self.mem.get_source_types()
        sink_types   = self.mem.get_sink_types()

        system = f"""You are a Langflow component selector.

{LANGFLOW_GENERAL_RULES}

{LANGFLOW_EDGE_PATTERNS}

Source/entry-point component types: {', '.join(source_types[:6]) or 'ChatInput'}
Sink/exit-point component types: {', '.join(sink_types[:6]) or 'ChatOutput'}

CLOSEST MATCHING EXAMPLE (use as primary reference):
{anchor}

AVAILABLE COMPONENTS (exact WorkflowType IDs from live catalog):
{catalog_text}

Relevant RAG context:
{self._rag_snippets[:1800] if self._rag_snippets else "(none)"}

SELECTION RULES:
1. Select 3-8 components that together fulfil the user's workflow.
2. Use ONLY IDs listed in the catalog above — exact strings required.
3. node_id format: WorkflowType-XXXXX (5 alphanumeric chars, no spaces).
4. For EVERY component include meaningful parameters matching the user's needs.
5. For Prompt nodes: include "template" parameter with actual text and {{variables}}.
6. For LLM nodes: include provider, model_name, api_key (use {{ENV_VAR}}).
7. Always include the required source (e.g. ChatInput) and sink (e.g. ChatOutput).
8. If memory is needed add a Memory component AND put {{memory}} in the Prompt template.

Return ONLY a JSON array — no markdown:
[
  {{
    "node_id": "WorkflowType-XXXXX",
    "component_type": "ExactWorkflowType",
    "display_name": "Human-readable label",
    "parameters": {{"field_name": "value", ...}}
  }}
]"""

        user_msg = f"Requirements:\n{reqs}\n\nConversation:\n{conv}\n\nSelect components:"

        raw = self._parse_retry([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])

        allowed = self._allowed_types()
        components = self._sanitize_components(raw, allowed)

        if not components:
            components = self._fallback_bootstrap(best_ex, allowed)
            if components:
                self.mem.warnings.append("Used fallback bootstrap for component selection.")

        if not components:
            self.mem.errors.append("Component selection failed.")
            return

        # Auto-add source/sink if flagged as needed
        if self.mem.requirements.get("needs_source_sink", True):
            components = self._ensure_source_sink(components)

        # Ensure a Prompt component if memory/system_prompt is needed
        components = self._ensure_prompt_if_needed(components, allowed)

        self.mem.selected_components = components

        # Resolve + hydrate from live catalog
        resolver = ComponentResolver(self.mem, rag=self.rag)
        resolver.resolve_all_components()

        print(f"[Planning] {len(self.mem.selected_components)} components selected.")

    # ── Phase 4: Fill fields interactively ────────────────────────────────

    def _fill_component_fields_interactively(self) -> None:
        """
        For each component, enumerate required + important fields.
        If a field has no value yet, ask the user.
        """
        print("\n[Planning] Checking required fields for each component...\n")

        for comp in self.mem.selected_components:
            ctype = comp.get("component_type", "")
            display = comp.get("display_name", ctype)
            params = comp.setdefault("parameters", {})
            if not isinstance(params, dict):
                comp["parameters"] = {}
                params = comp["parameters"]

            # Collect fields that need values
            missing = self._get_missing_fields(comp)
            if not missing:
                continue

            print(f"\n  Component: {display} ({ctype})")
            print(f"  The following fields need values:\n")

            for field_info in missing:
                fname   = field_info["name"]
                fdesc   = field_info.get("description", "")
                ftype   = field_info.get("type", "str")
                fopts   = field_info.get("options", [])
                fdefault = field_info.get("default", "")
                freq    = field_info.get("required", False)

                # Skip fields that are edge-fed (will be set by graph connections)
                if fname in {"input_value", "message", "prompt", "response", "system_message"}:
                    continue

                # Try to derive from requirements first
                auto_value = self._derive_field_from_requirements(ctype, fname)
                if auto_value:
                    params[fname] = auto_value
                    print(f"    ✓ {fname}: {str(auto_value)[:60]} (auto-filled)")
                    continue

                # Build prompt
                prompt_parts = [f"    [{fname}]"]
                if fdesc:
                    prompt_parts.append(f"  {fdesc}")
                if ftype == "float":
                    prompt_parts.append(f"  (number")
                    if fdefault:
                        prompt_parts.append(f", default: {fdefault}")
                    prompt_parts.append(")")
                elif fopts:
                    prompt_parts.append(f"  Options: {', '.join(str(o) for o in fopts[:8])}")
                    if fdefault:
                        prompt_parts.append(f"  (default: {fdefault})")
                elif fdefault:
                    prompt_parts.append(f"  (default: {fdefault})")
                if freq:
                    prompt_parts.append("  [REQUIRED]")

                print("".join(prompt_parts))
                user_val = input("    > ").strip()

                if user_val:
                    params[fname] = user_val
                elif fdefault:
                    params[fname] = fdefault
                    print(f"    Using default: {fdefault}")
                elif freq:
                    self.mem.warnings.append(
                        f"Required field '{fname}' on '{ctype}' was left blank."
                    )

        # Second pass — LLM fills any remaining blanks
        self._llm_fill_remaining_params()

    def _get_missing_fields(self, comp: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return list of fields that need user input."""
        ctype = comp.get("component_type", "")
        params = comp.get("parameters", {})
        live_template = comp.get("live_template", {})

        missing: List[Dict[str, Any]] = []
        ck = self.mem.get_component_knowledge(ctype)

        # Build field list from live_template
        fields_to_check: List[Dict[str, Any]] = []
        if isinstance(live_template, dict):
            for fname, fdef in live_template.items():
                if fname in ("_type", "code") or not isinstance(fdef, dict):
                    continue
                if not fdef.get("show", True):
                    continue
                if fdef.get("advanced", False):
                    continue
                fields_to_check.append({
                    "name": fname,
                    "description": fdef.get("info", "") or fdef.get("display_name", ""),
                    "type": fdef.get("type", "str"),
                    "options": fdef.get("options", []),
                    "default": str(fdef.get("value", fdef.get("default", ""))),
                    "required": bool(fdef.get("required", False)),
                })
        elif ck:
            # Fall back to knowledge_index fields
            for kf in ck.key_fields[:12]:
                fields_to_check.append({
                    "name": kf["name"],
                    "description": kf.get("info", ""),
                    "type": kf.get("type", "str"),
                    "options": kf.get("options", []),
                    "default": str(kf.get("default", "")),
                    "required": bool(kf.get("required", False)),
                })

        for field in fields_to_check:
            fname = field["name"]
            # Skip edge-fed and internal fields
            _SKIP = {"input_value", "message", "prompt", "response", "system_message",
                     "code", "_type", "session_id", "sender", "sender_name"}
            if fname in _SKIP:
                continue
            # Skip if already has a value
            existing = params.get(fname)
            if existing not in (None, "", [], {}):
                continue
            # Include required fields and key visible fields
            if field.get("required") or field.get("options") or field.get("type") in ("str", "float", "int"):
                missing.append(field)

        return missing

    def _derive_field_from_requirements(self, ctype: str, fname: str) -> Optional[str]:
        """
        Return a value for a field ONLY if it is a deterministic fact from requirements
        — something the user explicitly told us, not a guess based on field or type names.

        The LLM (called in _llm_fill_remaining_params) handles all semantic decisions.
        This function only short-circuits the interactive user prompt for values we
        already know with certainty.
        """
        r   = self.mem.requirements
        fl  = fname.lower()

        # If the user explicitly gave us the value for this exact field name, use it.
        if fname in r and r[fname] not in (None, "", [], {}):
            return str(r[fname])

        # Requirements carry an explicit API key env-var name → always safe to fill.
        # We only apply it to fields that look like credential fields.
        env = r.get("api_key_env_var", "")
        if env and ("key" in fl or "token" in fl or "secret" in fl):
            return "{" + env + "}"

        return None

    def _llm_fill_remaining_params(self) -> None:
        """
        Ask the LLM to fill missing parameter values for every component.

        The LLM receives, for each component:
          - Its exact live template (field name, type, required, options, description)
          - Its current parameter values
          - The full user requirements and workflow context

        This means the LLM acts as a Langflow expert with direct knowledge of EACH
        component's actual fields — no hardcoded field names, no type-name guessing.
        """
        # Build a rich per-component payload including the live template fields
        payload = []
        for c in self.mem.selected_components:
            ctype   = c.get("component_type", "")
            params  = c.get("parameters", {}) or {}
            live_tmpl = c.get("live_template", {})

            # Summarise live template fields that are empty or missing
            template_fields: List[Dict[str, Any]] = []
            if isinstance(live_tmpl, dict):
                for fname, fdef in live_tmpl.items():
                    if fname in ("_type", "code") or not isinstance(fdef, dict):
                        continue
                    if not fdef.get("show", True) or fdef.get("advanced", False):
                        continue
                    current_val = params.get(fname)
                    template_fields.append({
                        "name":         fname,
                        "display_name": fdef.get("display_name", fname),
                        "type":         fdef.get("type", "str"),
                        "required":     bool(fdef.get("required", False)),
                        "options":      (fdef.get("options") or [])[:8],
                        "info":         (fdef.get("info") or "")[:80],
                        "current_value": current_val if current_val not in (None, "", [], {}) else None,
                    })
            else:
                # Fall back to knowledge_index
                ck = self.mem.get_component_knowledge(ctype)
                if ck:
                    for kf in ck.key_fields[:15]:
                        current_val = params.get(kf["name"])
                        template_fields.append({
                            "name":         kf["name"],
                            "display_name": kf.get("name", ""),
                            "type":         kf.get("type", "str"),
                            "required":     bool(kf.get("required", False)),
                            "options":      (kf.get("options") or [])[:8],
                            "info":         (kf.get("info") or "")[:80],
                            "current_value": current_val if current_val not in (None, "", [], {}) else None,
                        })

            if not template_fields:
                continue

            payload.append({
                "node_id":         c.get("node_id"),
                "component_type":  ctype,
                "display_name":    c.get("display_name", ctype),
                "fields":          template_fields,
            })

        if not payload:
            return

        system = f"""You are a Langflow workflow configuration expert.

For each component below, fill in appropriate values for empty fields.
You are given the EXACT live template for each component — field names, types,
required flags, allowed options, and description. Use this to make correct decisions.

Workflow goal: {self.mem.user_prompt}

Requirements collected from user:
{json.dumps(self.mem.requirements, indent=2, ensure_ascii=True)[:800]}

{LANGFLOW_GENERAL_RULES}

Rules:
- Fill ONLY fields where current_value is null.
- For any field accepting an API key / secret / token: use {{ENV_VAR_NAME}} format, never hardcode.
- For fields with options: choose only from the provided options list.
- For template fields on Prompt components: write meaningful text with {{variable}} placeholders
  that match the actual workflow needs (memory → {{memory}}, document → {{context}}, etc).
- Skip fields that will be fed by edges at runtime (e.g. input_value, message, prompt, response).
- If a field value cannot be determined from the requirements, omit it (don't guess).

Return ONLY a JSON array — no markdown, no explanation:
[
  {{
    "node_id": "...",
    "parameters": {{"field_name": "value", ...}}
  }}
]
Only include components where you actually have values to fill."""

        resp = self.mem.call_llm([
            {"role": "system", "content": system},
            {"role": "user",   "content": json.dumps(payload, indent=2, ensure_ascii=True)[:3000]},
        ], temperature=0.0, max_tokens=2500, label="fill_params")

        try:
            cleaned = re.sub(r"```(?:json)?|```", "", resp or "").strip()
            m = re.search(r"\[[\s\S]*\]", cleaned)
            if not m:
                return
            items = json.loads(m.group(0))
            for item in items:
                if not isinstance(item, dict):
                    continue
                nid        = item.get("node_id", "")
                new_params = item.get("parameters", {})
                if not isinstance(new_params, dict):
                    continue
                for comp in self.mem.selected_components:
                    if comp.get("node_id") == nid:
                        for k, v in new_params.items():
                            if v not in (None, "", [], {}):
                                comp["parameters"].setdefault(k, v)
        except Exception:
            pass

    # ── Phase 5: Enforce connectivity variables ────────────────────────────

    def _enforce_connectivity_from_content(self) -> None:
        """
        SYSTEM RULE: Connectivity derives from {variable} placeholders in content fields.
        Universal: detects which components are present in the workflow and injects the
        correct {variables} into every Prompt template so edges can be wired automatically.
        Works for memory chatbots, RAG pipelines, agent workflows, and any other type.
        """
        component_types = {c.get("component_type", "") for c in self.mem.selected_components}

        # ── Detect what's in the workflow ─────────────────────────────────
        wants_memory = (
            "memory" in self.mem.user_prompt.lower()
            or "memory" in str(self.mem.requirements).lower()
            or "history" in self.mem.user_prompt.lower()
            or any("memory" in t.lower() for t in component_types)
        )

        # Detect any selected component that produces Data or list[Data] output.
        # These are document/content providers (file loaders, scrapers, vector stores,
        # API clients, parsers, etc.) that need a Prompt {context} port to land on.
        # We use the LIVE knowledge_index output types — no hardcoded type names.
        wants_rag_context = False
        for t in component_types:
            ck = self.mem.get_component_knowledge(t)
            if ck and any("Data" in ot for ot in ck.output_types):
                wants_rag_context = True
                break

        system_prompt = str(self.mem.requirements.get("system_prompt", "")).strip()

        for comp in self.mem.selected_components:
            ctype = comp.get("component_type", "")
            if "prompt" not in ctype.lower():
                continue

            params = comp.setdefault("parameters", {})
            if not isinstance(params, dict):
                comp["parameters"] = {}
                params = comp["parameters"]

            template = str(params.get("template") or "")

            # Inject system prompt text if not yet in template
            if system_prompt and system_prompt.lower() not in template.lower():
                template = (system_prompt + "\n\n" + template).strip()

            existing_vars = {v.lower() for v in _extract_template_vars(template)}

            # Inject {memory} if a Memory component is present
            if wants_memory and not (existing_vars & {"memory", "history", "chat_history"}):
                template = (template + "\n\nConversation history:\n{memory}").strip()
                existing_vars.add("memory")

            # Inject {context} if a RAG/document component is present
            if wants_rag_context and not (
                existing_vars & {"context", "document", "documents", "content", "rag_context"}
            ):
                template = (template + "\n\nContext:\n{context}").strip()
                existing_vars.add("context")

            # Ensure at least one dynamic variable so that at least one input port exists.
            # Only add {input} as a last resort when the template has NO variables at all.
            if not existing_vars:
                template = (template + "\n\n{input}").strip()

            params["template"] = template

    # ── Phase 6: Store prompt variables ───────────────────────────────────

    def _store_prompt_variables(self) -> None:
        """Record prompt template variables in requirements so Graph agent can wire them."""
        all_vars: Dict[str, List[str]] = {}
        for comp in self.mem.selected_components:
            ctype = comp.get("component_type", "")
            if "prompt" not in ctype.lower():
                continue
            template = str(comp.get("parameters", {}).get("template") or "")
            vars_found = _extract_template_vars(template)
            if vars_found:
                all_vars[comp["node_id"]] = vars_found

        if all_vars:
            self.mem.requirements["prompt_template_variables"] = all_vars

    # ── Helpers: catalog ──────────────────────────────────────────────────

    def _build_catalog_text(self) -> str:
        """Build a compact component list for the LLM."""
        fetcher = getattr(self.mem, "_fetcher", None)
        if fetcher and hasattr(fetcher, "_type_to_raw"):
            types = list(fetcher._type_to_raw.keys())
        else:
            types = [c.get("id", "") for c in self.mem.live_components if c.get("id")]

        ck = self.mem.knowledge_index
        lines = []
        for t in types[:280]:
            info = ck.get(t)
            if info:
                roles = []
                if info.is_source:        roles.append("SOURCE")
                if info.is_sink:          roles.append("SINK")
                if info.is_llm:           roles.append("LLM")
                if info.is_tool_provider: roles.append("TOOL")
                role_str = f" [{','.join(roles)}]" if roles else ""
                lines.append(f"  {t}{role_str}: {info.description[:70]}")
            else:
                lines.append(f"  {t}")
        return "\n".join(lines)

    def _allowed_types(self) -> Set[str]:
        fetcher = getattr(self.mem, "_fetcher", None)
        if fetcher and hasattr(fetcher, "_type_to_raw"):
            return set(fetcher._type_to_raw.keys())
        return {c.get("id", "") for c in self.mem.live_components if c.get("id")}

    def _find_best_example(self) -> Optional[Dict[str, Any]]:
        if not self.mem.example_flows:
            return None
        user_lower = self.mem.user_prompt.lower()
        wf_type = self.mem.requirements.get("workflow_type", "").lower()

        scored = []
        selected_types = {c.get("component_type", "") for c in self.mem.selected_components}
        for ex in self.mem.example_flows:
            name = (ex.get("name") or "").lower()
            desc = (ex.get("description") or "").lower()
            nodes, _ = SharedMemory._unwrap_flow_data(ex)
            ex_types = {n.get("data", {}).get("type", "") for n in nodes}

            score = 0
            for word in user_lower.split():
                if len(word) >= 4 and word in name:
                    score += 3
                if len(word) >= 4 and word in desc:
                    score += 1
            if wf_type:
                for part in wf_type.replace("_", " ").split():
                    if part in name or part in desc:
                        score += 2
            overlap = len(selected_types & ex_types)
            score += overlap * 2

            scored.append((score, ex))

        scored.sort(key=lambda x: -x[0])
        return scored[0][1] if scored else self.mem.example_flows[0]

    def _format_example(self, ex: Optional[Dict]) -> str:
        if not ex:
            return "(none)"
        nodes, edges = SharedMemory._unwrap_flow_data(ex)
        nmap = {n.get("id"): n.get("data", {}).get("type", "") for n in nodes}
        lines = [
            f"Name: {ex.get('name','')}",
            f"Description: {(ex.get('description') or '')[:150]}",
            f"Components: {', '.join(dict.fromkeys(nmap.values()))}",
        ]
        for e in edges[:10]:
            sh = e.get("data", {}).get("sourceHandle", {})
            th = e.get("data", {}).get("targetHandle", {})
            src_type = nmap.get(e.get("source", ""), "?")
            tgt_type = nmap.get(e.get("target", ""), "?")
            if sh.get("name") and th.get("fieldName"):
                lines.append(f"  Edge: {src_type}.{sh['name']} → {tgt_type}.{th['fieldName']}")
        return "\n".join(lines)

    # ── Helpers: parsing ───────────────────────────────────────────────────

    def _parse_retry(self, messages: List[Dict], max_retries: int = 3) -> Any:
        for attempt in range(max_retries):
            resp = self.mem.call_llm(messages, temperature=0.1 + attempt * 0.05,
                                     max_tokens=2500, label="select")
            try:
                cleaned = _extract_json_block(resp)
                result = json.loads(cleaned)
                if isinstance(result, (list, dict)):
                    return result
            except Exception:
                pass
        return []

    def _sanitize_components(self, raw: Any, allowed: Set[str]) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []

        resolver = ComponentResolver(self.mem, rag=self.rag)
        seen_ids: Set[str] = set()
        result = []

        for item in raw:
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("component_type", "")).strip()
            resolved = resolver.resolve(raw_type) if raw_type else None
            if not resolved or resolved not in allowed:
                continue

            nid = str(item.get("node_id", "")).strip()
            if not nid or nid in seen_ids or "-" not in nid:
                nid = f"{resolved}-{_rand(5)}"
            seen_ids.add(nid)

            params = item.get("parameters", {})
            if not isinstance(params, dict):
                params = {}

            result.append({
                "node_id":        nid,
                "component_type": resolved,
                "display_name":   item.get("display_name", resolved),
                "parameters":     params,
            })

        return result

    def _fallback_bootstrap(self, best_ex: Optional[Dict], allowed: Set[str]) -> List[Dict]:
        """If selection failed, build from best example."""
        if not best_ex:
            return []
        wf    = best_ex.get("workflow") or best_ex.get("data") or {}
        nodes = wf.get("data", {}).get("nodes", []) if isinstance(wf, dict) else []
        result = []
        seen: Set[str] = set()
        for n in nodes:
            t = n.get("data", {}).get("type", "")
            if not t or t in ("note", "noteNode") or t not in allowed or t in seen:
                continue
            seen.add(t)
            result.append({
                "node_id":        f"{t}-{_rand(5)}",
                "component_type": t,
                "display_name":   n.get("data", {}).get("node", {}).get("display_name", t),
                "parameters":     {},
            })
        if result:
            self.mem.mark_knowledge_use("planning", "bootstrap_fallback", 1)
        return result

    def _ensure_source_sink(self, components: List[Dict]) -> List[Dict]:
        types_present = {c.get("component_type", "") for c in components}

        has_source = any(self.mem.is_source_component(t) for t in types_present)
        has_sink   = any(self.mem.is_sink_component(t) for t in types_present)

        if not has_source:
            source_types = self.mem.get_source_types()
            if source_types:
                t = source_types[0]
                components.insert(0, {
                    "node_id": f"{t}-{_rand(5)}",
                    "component_type": t,
                    "display_name": t,
                    "parameters": {},
                })
                self.mem.warnings.append(f"Auto-added missing source component: {t}")

        if not has_sink:
            sink_types = self.mem.get_sink_types()
            if sink_types:
                t = sink_types[0]
                components.append({
                    "node_id": f"{t}-{_rand(5)}",
                    "component_type": t,
                    "display_name": t,
                    "parameters": {},
                })
                self.mem.warnings.append(f"Auto-added missing sink component: {t}")

        return components

    def _ensure_prompt_if_needed(self, components: List[Dict], allowed: Set[str]) -> List[Dict]:
        """
        Add a Prompt component when a system prompt or dynamic variable injection is needed
        but no Prompt was selected.

        Universal: detects need from requirements (any system_prompt/instructions key)
        and from presence of Memory or document-loader components that require a Prompt
        node to inject their output as template variables.
        """
        has_prompt = any("prompt" in c.get("component_type", "").lower() for c in components)
        if has_prompt:
            return components

        # Detect what's already selected
        selected_types = {c.get("component_type", "") for c in components}

        has_memory = any("memory" in t.lower() for t in selected_types)
        # A doc-provider is any component whose live output types include Data.
        # This is read from the knowledge_index — no hardcoded type names.
        has_doc_loader = any(
            (lambda ck: ck is not None and any("Data" in ot for ot in ck.output_types))(
                self.mem.get_component_knowledge(t)
            )
            for t in selected_types
        )

        needs_prompt = bool(
            self.mem.requirements.get("system_prompt")
            or self.mem.requirements.get("instructions")
            or self.mem.requirements.get("prompt_template")
            or has_memory      # Memory output needs a Prompt.{memory} port to land on
            or has_doc_loader  # Document output needs a Prompt.{context} port to land on
        )

        if not needs_prompt:
            return components

        # Find the Prompt component type in the catalog (exact match preferred)
        prompt_type = None
        for t in allowed:
            if _norm(t) == "promptcomponent":
                prompt_type = t
                break
        if not prompt_type:
            for t in allowed:
                if _norm(t) == "prompt":
                    prompt_type = t
                    break
        if not prompt_type:
            return components  # no Prompt component in this Langflow instance

        sp = self.mem.requirements.get("system_prompt", "You are a helpful assistant.")
        template = sp
        if has_memory:
            template += "\n\nConversation history:\n{memory}"
        if has_doc_loader:
            template += "\n\nContext:\n{context}"
        if not _extract_template_vars(template):
            template += "\n\n{input}"

        components.append({
            "node_id":        f"{prompt_type}-{_rand(5)}",
            "component_type": prompt_type,
            "display_name":   "Prompt",
            "parameters":     {"template": template},
        })
        self.mem.warnings.append(f"Auto-added Prompt component: {prompt_type}")
        return components

    # ── Formatting helpers ─────────────────────────────────────────────────

    def _fmt_conv(self, max_chars: int = 2000) -> str:
        lines = []
        for m in self.mem.conversation_history:
            role = m.get("role", "").upper()
            content = (m.get("content") or "")[:300]
            lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    def _print_component_summary(self) -> None:
        print("\n[Planning] Selected components:")
        for c in self.mem.selected_components:
            ctype = c.get("component_type", "?")
            display = c.get("display_name", ctype)
            params = c.get("parameters", {})
            param_preview = ", ".join(
                f"{k}={str(v)[:30]}" for k, v in list(params.items())[:3] if v
            )
            print(f"  • {display} ({ctype})" + (f"  [{param_preview}]" if param_preview else ""))
