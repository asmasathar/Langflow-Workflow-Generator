from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from shared_memory import SharedMemory
from knowledge_fetcher import KnowledgeFetcher
from langflow_guidelines import (
    LANGFLOW_GENERAL_RULES,
    LANGFLOW_EDGE_PATTERNS,
    LANGFLOW_COMPONENT_FIELD_GUIDE,
)


def _extract_json_block(text: str) -> Optional[str]:
    """Extract the last valid JSON object from text."""
    text = re.sub(r"```(?:json)?|```", "", (text or ""), flags=re.IGNORECASE).strip()
    candidates: List[str] = []
    stack: List[str] = []
    start: Optional[int] = None
    in_str = escape = False

    for i, ch in enumerate(text):
        if in_str:
            if escape:   escape = False
            elif ch == "\\": escape = True
            elif ch == '"':  in_str = False
            continue
        if ch == '"':
            in_str = True; continue
        if ch in "[{":
            if not stack: start = i
            stack.append(ch)
        elif ch in "]}" and stack:
            opener = stack[-1]
            if (opener == "[" and ch == "]") or (opener == "{" and ch == "}"):
                stack.pop()
                if not stack and start is not None:
                    candidates.append(text[start: i + 1])
                    start = None

    for cand in reversed(candidates):
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    return None


def _extract_template_vars(template: str) -> List[str]:
    return re.findall(r"\{([^}]+)\}", template or "")


# ── WorkflowRefiner ────────────────────────────────────────────────────────────

class WorkflowRefiner:
    """
    Fixes workflow issues using LLM reasoning over the FULL shared context.

    Full context passed to LLM every call:
      - user_prompt
      - requirements
      - current components (with all parameters)
      - current edges
      - validation report
      - RAG component facts for problematic components
      - Langflow construction rules
    """

    def __init__(
        self,
        mem: SharedMemory,
        fetcher: Optional[KnowledgeFetcher] = None,
        rag: Optional[Any] = None,
    ):
        self.mem     = mem
        self.fetcher = fetcher
        self.rag     = rag

    # ── Entry point ────────────────────────────────────────────────────────

    def run(self, validation_report: str) -> bool:
        """
        Fix issues described in validation_report.
        Returns True if any changes were made, False otherwise.
        """
        # First do deterministic fixes (fast, no LLM)
        det_changed = self._deterministic_fixes(validation_report)

        # Then LLM-reasoned fixes for remaining issues
        llm_changed = self._llm_reasoned_refinement(validation_report)

        return det_changed or llm_changed

    # ── Deterministic fixes ────────────────────────────────────────────────

    def _deterministic_fixes(self, report: str) -> bool:
        """
        Apply only structurally-certain fixes that require zero semantic judgment.

        Everything that requires understanding what a field means, what a component does,
        or which value to use is handled by the LLM in _llm_reasoned_refinement.
        """
        changed = False

        # Fix 1: Remove duplicate target-field edges
        # (same (target, field) can only have one source — this is a structural rule)
        edges = self.mem.abstract_graph.get("edges", [])
        seen: Dict[Tuple[str, str], str] = {}
        cleaned: List[Dict] = []
        removed = 0
        for e in edges:
            tgt   = e.get("target", "")
            field = e.get("target_field_name", "")
            key   = (tgt, field)
            if key in seen:
                removed += 1
                continue
            seen[key] = e.get("source", "")
            cleaned.append(e)

        if removed > 0:
            self.mem.abstract_graph["edges"] = cleaned
            self.mem.warnings.append(f"[Refiner] Removed {removed} duplicate target-field edge(s).")
            changed = True
            self.mem.graph_dirty = True

        # Fix 2: Remove self-loop edges (src == tgt is always invalid)
        edges = self.mem.abstract_graph.get("edges", [])
        no_self = [e for e in edges if e.get("source") != e.get("target")]
        if len(no_self) < len(edges):
            self.mem.abstract_graph["edges"] = no_self
            self.mem.graph_dirty = True
            changed = True

        # Fix 3: Template variable → edge enforcement
        # If a Prompt template has {var} and no edge lands on that port → add one.
        # This is deterministic because the rule "every {var} needs an incoming edge"
        # is absolute in Langflow regardless of workflow type.
        changed |= self._fix_unmatched_template_vars()

        return changed

    def _fix_unmatched_template_vars(self) -> bool:
        """Add edges for unconnected {variable} ports in Prompt nodes."""
        changed = False
        edges = self.mem.abstract_graph.get("edges", [])
        nodes = self.mem.abstract_graph.get("nodes", [])
        node_map = {n["node_id"]: n for n in nodes}

        incoming: Dict[str, Set[str]] = {}
        for e in edges:
            incoming.setdefault(e.get("target", ""), set()).add(e.get("target_field_name", ""))

        existing_quads = {
            (e.get("source"), e.get("source_output_name"),
             e.get("target"), e.get("target_field_name"))
            for e in edges
        }

        for node in nodes:
            nid   = node.get("node_id", "")
            ctype = node.get("component_type", "")
            if "prompt" not in ctype.lower():
                continue

            params = node.get("parameters", {})
            template = str(params.get("template") or "")
            if not template:
                lt = node.get("live_template", {})
                if isinstance(lt, dict) and isinstance(lt.get("template"), dict):
                    template = str(lt["template"].get("value") or "")

            for var_name in _extract_template_vars(template):
                if var_name in incoming.get(nid, set()):
                    continue  # already connected

                # Find best source for this variable
                best = self._find_source_for_var(var_name, nid, nodes, existing_quads)
                if best:
                    edges.append(best)
                    existing_quads.add((
                        best["source"], best["source_output_name"],
                        best["target"], best["target_field_name"],
                    ))
                    incoming.setdefault(nid, set()).add(var_name)
                    changed = True
                    self.mem.graph_dirty = True
                    print(f"[Refiner][Det] Fixed template var '{{{var_name}}}' on '{nid}'")

        if changed:
            self.mem.abstract_graph["edges"] = edges
        return changed

    def _find_source_for_var(
        self,
        var_name: str,
        target_id: str,
        nodes: List[Dict],
        existing_quads: Set[Tuple],
    ) -> Optional[Dict]:
        """
        Find the best source node/output for a given template variable.
        Universal: covers memory, input, document/RAG, system, tool, and unknown variables.
        """
        vl = var_name.lower()

        # ── Semantic variable categories ───────────────────────────────────
        _MEM_VARS   = {"memory", "history", "chat_history", "conversation_history", "messages", "past_messages"}
        _INPUT_VARS = {"input", "query", "question", "user_input", "message", "human_input", "user_message", "user_query"}
        _DOC_VARS   = {"context", "document", "documents", "content", "rag_context", "retrieved",
                       "results", "data", "file", "file_content", "passage", "passages"}
        _SYS_VARS   = {"system", "instructions", "persona", "behavior", "system_prompt", "role"}
        _TOOL_VARS  = {"tools", "tool"}

        candidates: List[Tuple[int, Dict]] = []

        for src in nodes:
            if src["node_id"] == target_id:
                continue
            ctype_l = src["component_type"].lower()
            ck = self.mem.get_component_knowledge(src["component_type"])

            # Default output name; will be overridden per category
            out_name = "message"
            score = 0

            # ── Memory/history → Memory component ──────────────────────────
            if vl in _MEM_VARS:
                if "memory" in ctype_l:
                    score = 20
                    out_name = "messages_text"

            # ── Input/query → source component (ChatInput, TextInput) ───────
            elif vl in _INPUT_VARS:
                if self.mem.is_source_component(src["component_type"]):
                    score = 15
                    out_name = "message"

            # ── Document/context/RAG → File, VectorStore, ParseData ─────────
            elif vl in _DOC_VARS:
                if any(t in ctype_l for t in ("file", "vectorstore", "vector", "document", "loader", "parsedata", "parse")):
                    score = 20
                    # ParseData outputs parsed_text; others output message
                    out_name = "parsed_text" if "parse" in ctype_l else "message"

            # ── System/instructions → TextInput or source ───────────────────
            elif vl in _SYS_VARS:
                if any(t in ctype_l for t in ("textinput", "input")):
                    score = 10
                    out_name = "text"
                elif self.mem.is_source_component(src["component_type"]):
                    score = 5
                    out_name = "message"

            # ── Tools → Tool-providing components ───────────────────────────
            elif vl in _TOOL_VARS:
                if ck and ck.is_tool_provider:
                    score = 20
                    out_name = "component_as_tool"

            # ── Unknown variable: semantic fallback ─────────────────────────
            else:
                # Component type name contains the variable name → strong match
                norm_vl = re.sub(r"[^a-z0-9]", "", vl)
                norm_ct = re.sub(r"[^a-z0-9]", "", ctype_l)
                if norm_vl and norm_vl in norm_ct:
                    score = 15
                    out_name = (ck.output_names[0] if ck and ck.output_names else "message")
                elif self.mem.is_source_component(src["component_type"]):
                    score = 5
                    out_name = "message"

            if score == 0:
                continue

            quad = (src["node_id"], out_name, target_id, var_name)
            if quad in existing_quads:
                continue

            candidates.append((score, {
                "source":             src["node_id"],
                "source_output_name": out_name,
                "target":             target_id,
                "target_field_name":  var_name,
            }))

        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]


    # ── LLM-reasoned refinement ────────────────────────────────────────────

    def _llm_reasoned_refinement(self, validation_report: str) -> bool:
        """Ask LLM to fix remaining issues with full context."""

        # Build catalog / allowed types
        catalog_ids = self._catalog_ids()
        allowed_types = set(catalog_ids)

        # RAG facts for issue components
        issue_types = self._extract_issue_types(validation_report)
        rag_facts: List[str] = []
        if self.rag and getattr(self.rag, "is_built", False):
            for ctype in issue_types:
                fact = self.rag.query(f"{ctype} component fields edges", top_k=2,
                                       kinds=["component_fact", "edge_pattern"])
                if fact:
                    rag_facts.append(fact)
            if rag_facts:
                self.mem.mark_knowledge_use("refiner", "component_fact", len(rag_facts))

        # Build component context with live template fields so the LLM knows
        # exactly what fields each component has — no guessing from type names.
        current_components = []
        for c in self.mem.selected_components:
            ctype   = c.get("component_type", "")
            live_tmpl = c.get("live_template", {})

            # Summarise live template: name, type, required, options, current value
            field_summary: List[Dict] = []
            if isinstance(live_tmpl, dict):
                for fname, fdef in live_tmpl.items():
                    if fname in ("_type", "code") or not isinstance(fdef, dict):
                        continue
                    if not fdef.get("show", True):
                        continue
                    current_val = c.get("parameters", {}).get(fname)
                    field_summary.append({
                        "name":     fname,
                        "type":     fdef.get("type", "str"),
                        "required": bool(fdef.get("required", False)),
                        "options":  (fdef.get("options") or [])[:6],
                        "info":     (fdef.get("info") or "")[:60],
                        "value":    current_val if current_val not in (None, "", [], {}) else None,
                    })
            else:
                ck = self.mem.get_component_knowledge(ctype)
                if ck:
                    for kf in ck.key_fields[:12]:
                        current_val = c.get("parameters", {}).get(kf["name"])
                        field_summary.append({
                            "name":     kf["name"],
                            "type":     kf.get("type", "str"),
                            "required": bool(kf.get("required", False)),
                            "options":  (kf.get("options") or [])[:6],
                            "info":     (kf.get("info") or "")[:60],
                            "value":    current_val if current_val not in (None, "", [], {}) else None,
                        })

            current_components.append({
                "node_id":        c.get("node_id", ""),
                "component_type": ctype,
                "display_name":   c.get("display_name", ""),
                "parameters":     c.get("parameters", {}),
                "live_fields":    field_summary,  # actual field schema from Langflow catalog
            })
        current_edges = self.mem.abstract_graph.get("edges", [])

        # Snapshot to detect changes
        before_sig = json.dumps(
            {"comps": current_components, "edges": current_edges},
            sort_keys=True, ensure_ascii=True
        )

        system = f"""You are a Universal Langflow Workflow Architect and Repair Agent.

Your role: fix ALL issues in the validation report for ANY kind of Langflow workflow.
You must be an expert in every workflow type: chatbots, RAG pipelines, agents with tools,
market research, document analysis, web scraping, YouTube analysis, data extraction,
structured output, sequential LLM chains, embeddings, vector stores, and anything else.

{LANGFLOW_GENERAL_RULES}

{LANGFLOW_EDGE_PATTERNS}

{LANGFLOW_COMPONENT_FIELD_GUIDE}

## Your Authority
1. Add, remove, or replace ANY component.
2. Add, remove, or replace ANY edge.
3. Set or update ANY component parameter field.
4. Inject {{variable}} into Prompt templates to create new input ports.

## Universal Absolute Rules
- Field-Edge Parity: every {{var}} in ANY Prompt template MUST have exactly one incoming edge.
  The port only exists because the {{var}} is in the template text.
- Target-field exclusivity: each (target_node, field_name) can only receive ONE edge.
- No orphan nodes: every non-source node needs at least one incoming edge.
- No self-loops.
- Use ONLY component_type values from the provided catalog IDs.
- Keep node_ids stable unless intentionally replacing a component.
- API keys: ALWAYS use {{ENV_VAR}} format, never hardcode secrets.

## Prompt Template Variable → Edge Rule (applies to ALL workflow types)
- If a Prompt template contains {{memory}} → wire: MemoryComponent.messages_text → Prompt.memory
- If a Prompt template contains {{context}} → wire: ParseData.parsed_text → Prompt.context
  OR VectorStore.search_results → ParseData.data → ParseData.parsed_text → Prompt.context
- If a Prompt template contains {{document}} → wire: File.message → Prompt.document
- If a Prompt template contains {{input}} → wire: SourceComponent.message → Prompt.input
- For ANY other {{variable}} → find the component whose output best semantically matches
  that variable name and wire it.

## Context
User goal: {self.mem.user_prompt}

Requirements:
{json.dumps(self.mem.requirements, indent=2, ensure_ascii=True)[:600]}

RAG component facts for problem areas:
{chr(10).join(rag_facts) if rag_facts else "(none)"}

Return ONLY this JSON (no markdown, no explanation):
{{
  "replace_components": true,
  "replace_edges": true,
  "components": [
    {{"node_id":"...","component_type":"ExactType","display_name":"...","parameters":{{...}}}}
  ],
  "edges": [
    {{"source":"...","source_output_name":"...","target":"...","target_field_name":"..."}}
  ]
}}"""

        # Trim live_fields from display for readability, but keep for LLM
        user_msg = f"""Validation report:
{validation_report}

Current components (with live field schema from catalog):
{json.dumps(current_components, indent=2, ensure_ascii=True)[:3000]}

Current edges:
{json.dumps(current_edges, indent=2, ensure_ascii=True)[:1200]}

Catalog IDs (first 300):
{json.dumps(catalog_ids[:300], ensure_ascii=True)}"""

        response = self.mem.call_llm(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=3000,
            label="refine",
        )

        payload_str = _extract_json_block(response or "")
        if not payload_str:
            return False

        try:
            payload = json.loads(payload_str)
        except Exception:
            return False

        if not isinstance(payload, dict):
            return False

        # Apply component changes
        from component_resolver import ComponentResolver
        resolver = ComponentResolver(self.mem, rag=self.rag)

        comps_list = payload.get("components")
        if isinstance(comps_list, list) and payload.get("replace_components", True):
            new_comps: List[Dict] = []
            seen_ids:  Set[str]  = set()

            for item in comps_list:
                if not isinstance(item, dict):
                    continue
                raw_type = str(item.get("component_type", "")).strip()
                ctype    = resolver.resolve(raw_type) if raw_type else None
                nid      = str(item.get("node_id", "")).strip()

                if not nid or not ctype or ctype not in allowed_types or nid in seen_ids:
                    continue
                seen_ids.add(nid)

                params = item.get("parameters", {})
                if not isinstance(params, dict):
                    params = {}

                # Preserve existing parameters not overwritten
                existing = next(
                    (c for c in self.mem.selected_components if c.get("node_id") == nid),
                    None,
                )
                if existing:
                    merged_params = dict(existing.get("parameters", {}))
                    merged_params.update({k: v for k, v in params.items() if v not in (None, "", [], {})})
                    params = merged_params

                new_comps.append({
                    "node_id":        nid,
                    "component_type": ctype,
                    "display_name":   item.get("display_name", ctype),
                    "parameters":     params,
                })

            if new_comps:
                self.mem.selected_components = new_comps
                # Re-hydrate from live catalog
                try:
                    resolver.resolve_all_components()
                except Exception:
                    pass
                self.mem.graph_rebuild_from_selection = True

        # Apply edge changes
        edges_list = payload.get("edges")
        if isinstance(edges_list, list) and payload.get("replace_edges", True):
            new_edges: List[Dict] = []
            valid_ids = {c.get("node_id", "") for c in self.mem.selected_components}
            seen_quads: Set[Tuple] = set()
            field_owner: Dict[Tuple, str] = {}

            for item in edges_list:
                if not isinstance(item, dict):
                    continue
                src     = str(item.get("source", "")).strip()
                src_out = str(item.get("source_output_name", "output")).strip()
                tgt     = str(item.get("target", "")).strip()
                tgt_fld = str(item.get("target_field_name", "input_value")).strip()

                if not (src and tgt and src in valid_ids and tgt in valid_ids):
                    continue
                if src == tgt:
                    continue

                quad = (src, src_out, tgt, tgt_fld)
                if quad in seen_quads:
                    continue

                # Enforce target-field exclusivity
                field_key = (tgt, tgt_fld)
                if field_key in field_owner:
                    continue
                field_owner[field_key] = src

                seen_quads.add(quad)
                new_edges.append({
                    "source":             src,
                    "source_output_name": src_out,
                    "target":             tgt,
                    "target_field_name":  tgt_fld,
                })

            if new_edges:
                self.mem.abstract_graph["edges"] = new_edges
                self.mem.graph_dirty = True

        # Detect changes
        after_comps = [
            {
                "node_id":        c.get("node_id", ""),
                "component_type": c.get("component_type", ""),
                "parameters":     c.get("parameters", {}),
            }
            for c in self.mem.selected_components
        ]
        after_sig = json.dumps(
            {"comps": after_comps, "edges": self.mem.abstract_graph.get("edges", [])},
            sort_keys=True, ensure_ascii=True,
        )

        changed = before_sig != after_sig
        if changed:
            self.mem.mark_knowledge_use("refiner", "llm_field_derivation", 1)
        return changed

    # ── Helpers ────────────────────────────────────────────────────────────

    def _catalog_ids(self) -> List[str]:
        if self.fetcher and hasattr(self.fetcher, "_type_to_raw"):
            return list(self.fetcher._type_to_raw.keys())
        return [c.get("id", "") for c in self.mem.live_components if c.get("id")]

    def _extract_issue_types(self, report: str) -> List[str]:
        """Extract component types mentioned in validation report."""
        # Look for patterns like "node 'ComponentType-xxxxx'" or "'ComponentType'"
        types_found: List[str] = []
        by_id = {c.get("node_id", ""): c.get("component_type", "")
                 for c in self.mem.selected_components}

        # Find node IDs mentioned in report
        for nid, ctype in by_id.items():
            if nid in report and ctype:
                types_found.append(ctype)

        # Also find component types mentioned directly
        for comp in self.mem.selected_components:
            ctype = comp.get("component_type", "")
            if ctype and ctype in report:
                types_found.append(ctype)

        return list(dict.fromkeys(types_found))[:6]
