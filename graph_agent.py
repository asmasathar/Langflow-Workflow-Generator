from __future__ import annotations

import json
import os
import re
import string
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from shared_memory import SharedMemory
from langflow_guidelines import LANGFLOW_GENERAL_RULES, LANGFLOW_EDGE_PATTERNS


def _rand(n: int = 5) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _extract_template_vars(template: str) -> List[str]:
    """Extract {variable} names from a template string."""
    return re.findall(r"\{([^}]+)\}", template or "")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _debug() -> bool:
    return os.getenv("WF_DEBUG", "1").strip().lower() not in {"0", "false", "no"}


def _allow_llm() -> bool:
    return os.getenv("WF_ALLOW_LLM_GAP_FILL", "1").strip().lower() in {"1", "true", "yes", "on"}


# ── Type compatibility ─────────────────────────────────────────────────────────

def _types_compatible(src_types: List[str], tgt_types: List[str], tgt_field_type: str = "str") -> bool:
    """Returns True if at least one src type is accepted by the target."""
    if not src_types or not tgt_types:
        return True  # unknown → assume compatible

    src_set = {t.lower() for t in src_types}
    tgt_set = {t.lower() for t in tgt_types}

    # Direct overlap
    if src_set & tgt_set:
        return True

    # Message is compatible with str inputs
    if "message" in src_set and tgt_field_type in ("str", ""):
        return True

    # Data → Data
    if "data" in src_set and "data" in tgt_set:
        return True

    # Text/Message cross-compatibility
    if {"message", "text", "str"} & src_set and {"message", "text", "str"} & tgt_set:
        return True

    return False


# ── GraphAgent ─────────────────────────────────────────────────────────────────

class GraphAgent:
    """
    Builds the abstract edge graph using:
      - LLM global planning with full workflow context
      - Template variable → port enforcement
      - Role-based flow integrity
    """

    def __init__(self, mem: SharedMemory, rag: Optional[Any] = None):
        self.mem = mem
        self.rag = rag

    # ── Entry point ────────────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "=" * 60)
        print("  GRAPH AGENT — Edge Design & Topology")
        print("=" * 60)

        if not self.mem.selected_components:
            raise RuntimeError("No components selected. Run PlanningAgent first.")

        nodes = self._build_nodes()
        nodes = self._expand_if_needed(nodes)

        # If refiner already set corrected edges, just re-position
        if (
            self.mem.graph_dirty
            and self.mem.abstract_graph.get("edges")
            and not self.mem.graph_rebuild_from_selection
        ):
            print("[Graph] Refiner-corrected edges present — repositioning nodes only.")
            existing   = self.mem.abstract_graph["edges"]
            valid_ids  = {n["node_id"] for n in nodes}
            clean      = self._validate_edges(existing, valid_ids, nodes)
            nodes      = self._assign_positions(nodes, clean)
            self.mem.abstract_graph = {"nodes": nodes, "edges": clean}
            self.mem.graph_dirty    = False
            self.mem.graph_rebuild_from_selection = False
            print(f"[Graph] Done: {len(nodes)} nodes, {len(clean)} edges.")
            return

        # Full rebuild
        edges = self._build_edges(nodes)
        nodes = self._assign_positions(nodes, edges)
        self.mem.abstract_graph = {"nodes": nodes, "edges": edges}
        self.mem.graph_dirty    = False
        self.mem.graph_rebuild_from_selection = False
        print(f"\n[Graph] Built: {len(nodes)} nodes, {len(edges)} edges.")
        self._print_edge_summary(edges, nodes)

    # ── Node assembly ──────────────────────────────────────────────────────

    def _build_nodes(self) -> List[Dict[str, Any]]:
        return [
            {
                "node_id":        c.get("node_id", ""),
                "component_type": c.get("component_type", ""),
                "display_name":   c.get("display_name", c.get("component_type", "")),
                "parameters":     c.get("parameters", {}),
                "live_template":  c.get("live_template", {}),
                "live_inputs":    c.get("live_inputs", []),
                "live_outputs":   c.get("live_outputs", []),
                "base_classes":   c.get("base_classes", []),
                "description":    c.get("description", ""),
            }
            for c in self.mem.selected_components
        ]

    def _expand_if_needed(self, nodes: List[Dict]) -> List[Dict]:
        """Ask LLM if bridge/adapter components are needed."""
        if not nodes or not _allow_llm():
            return nodes

        catalog_ids = self._catalog_ids()
        existing_types = {n["component_type"] for n in nodes}

        system = """You are a Langflow component set reviewer.
Decide if any REQUIRED bridge or adapter components are missing for a valid workflow.
Return [] if current components are already sufficient.
Return JSON array of additions ONLY if absolutely necessary:
[{"component_type":"ExactType","display_name":"...","parameters":{}}]
Rules:
- Minimal additions only
- Use only types from the catalog
- Do not add ChatInput/ChatOutput if already present"""

        user = (
            f"User goal: {self.mem.user_prompt}\n\n"
            f"Current components: {json.dumps([{'type': n['component_type'], 'params': list(n.get('parameters', {}).keys())} for n in nodes], ensure_ascii=True)}\n\n"
            f"Catalog IDs (first 200): {json.dumps(catalog_ids[:200], ensure_ascii=True)}"
        )

        resp = self.mem.call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=600, label="expand",
        )

        try:
            cleaned = re.sub(r"```(?:json)?|```", "", resp or "").strip()
            m = re.search(r"\[[\s\S]*\]", cleaned)
            additions = json.loads(m.group(0)) if m else []
        except Exception:
            additions = []

        if not isinstance(additions, list) or not additions:
            return nodes

        catalog_set = set(catalog_ids)
        from component_resolver import ComponentResolver
        resolver = ComponentResolver(self.mem, rag=self.rag)
        added = 0

        for item in additions:
            if not isinstance(item, dict):
                continue
            raw_type = str(item.get("component_type", "")).strip()
            ctype = resolver.resolve(raw_type)
            if not ctype or ctype not in catalog_set or ctype in existing_types:
                continue
            params = item.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            self.mem.selected_components.append({
                "node_id":        f"{ctype}-{_rand(5)}",
                "component_type": ctype,
                "display_name":   item.get("display_name", ctype),
                "parameters":     params,
            })
            existing_types.add(ctype)
            added += 1

        if added > 0:
            print(f"[Graph] Added {added} bridge component(s).")
            # Hydrate new components with live metadata
            try:
                from component_resolver import ComponentResolver as CR
                CR(self.mem, rag=self.rag).resolve_all_components()
            except Exception:
                pass
            return self._build_nodes()

        return nodes

    # ── Edge building ──────────────────────────────────────────────────────

    def _build_edges(self, nodes: List[Dict]) -> List[Dict]:
        """Build edges using LLM global plan + template variable enforcement + integrity checks."""
        prompt_vars = self._collect_all_prompt_vars(nodes)
        existing_edges: List[Dict] = []

        # Stage A: LLM global topology plan
        if _allow_llm():
            seed_edges = self._llm_plan_edges(nodes, prompt_vars)
            if seed_edges:
                existing_edges.extend(seed_edges)
                if _debug():
                    print(f"[Graph] LLM planned {len(seed_edges)} edges.")
                self.mem.mark_knowledge_use("graph", "llm_gap_fill", len(seed_edges))

        # Stage B: LLM gap-fill for uncovered nodes
        covered_targets = {e.get("target") for e in existing_edges}
        uncovered = [
            n for n in nodes
            if not self.mem.is_source_component(n["component_type"])
            and n["node_id"] not in covered_targets
        ]
        if uncovered and _allow_llm():
            gap_edges = self._llm_fill_gaps(nodes, existing_edges, uncovered, prompt_vars)
            existing_edges.extend(gap_edges)

        # Stage C: Template variable enforcement (critical for Prompt nodes)
        existing_edges = self._enforce_template_var_edges(nodes, existing_edges, prompt_vars)

        # Stage D: Flow integrity
        existing_edges = self._ensure_flow_integrity(nodes, existing_edges)

        # Deduplicate
        existing_edges = self._dedupe_edges(existing_edges)

        # Validate
        valid_ids = {n["node_id"] for n in nodes}
        return self._validate_edges(existing_edges, valid_ids, nodes)

    def _llm_plan_edges(self, nodes: List[Dict], prompt_vars: Dict[str, List[str]]) -> List[Dict]:
        """Ask LLM to plan the complete edge topology in one shot."""
        node_summary = [
            {
                "node_id":        n["node_id"],
                "component_type": n["component_type"],
                "outputs":        [o.get("name", "") for o in self._get_outputs(n)][:4],
                "inputs":         [i.get("name", "") for i in self._get_inputs(n)][:8],
                "template_vars":  prompt_vars.get(n["node_id"], []),
            }
            for n in nodes
        ]

        # RAG edge patterns
        rag_context = ""
        if self.rag and getattr(self.rag, "is_built", False):
            comp_types = [n["component_type"] for n in nodes]
            rag_context = self.rag.query_edge_patterns(comp_types, self.mem.user_prompt) or ""
            if rag_context:
                self.mem.mark_knowledge_use("graph", "rag_edge_context", 1)

        system = f"""You are a Langflow edge wiring expert.

{LANGFLOW_GENERAL_RULES}

{LANGFLOW_EDGE_PATTERNS}

CRITICAL RULE for template variables:
- If a Prompt node has template_vars = ["memory"], you MUST wire:
    Memory.messages_text → Prompt.memory
  The port "memory" on the Prompt node ONLY exists because {{memory}} is in the template.
- If template_vars = ["input"], wire ChatInput.message → Prompt.input
- Match every template variable to an appropriate source output.

Type compatibility:
  Message/text outputs → Message/str inputs (compatible)
  LanguageModel output (model_output) → Agent.llm / StructuredOutput.llm ONLY
  text_output (Message) → ChatOutput.input_value (use this, NOT model_output)
  Tool outputs → Agent.tools only

RAG retrieved edge patterns:
{rag_context[:1000] if rag_context else "(none)"}

Return ONLY a JSON array of edges:
[
  {{
    "source": "NodeId",
    "source_output_name": "output_name",
    "target": "NodeId",
    "target_field_name": "field_name",
    "description": "why this edge exists"
  }}
]
Every non-source node MUST be a target of at least one edge.
Every non-sink node MUST be a source of at least one edge."""

        user = (
            f"User goal: {self.mem.user_prompt}\n\n"
            f"Requirements: {json.dumps(self.mem.requirements, ensure_ascii=True)[:600]}\n\n"
            f"Nodes:\n{json.dumps(node_summary, indent=2, ensure_ascii=True)[:2500]}"
        )

        resp = self.mem.call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=2000, label="plan_edges",
        )

        return self._parse_edges(resp)

    def _llm_fill_gaps(
        self,
        nodes: List[Dict],
        existing: List[Dict],
        uncovered: List[Dict],
        prompt_vars: Dict[str, List[str]],
    ) -> List[Dict]:
        """Ask LLM to fill edges for uncovered nodes."""
        system = f"""You are filling MISSING edges in a Langflow workflow.
{LANGFLOW_GENERAL_RULES}

Return ONLY a JSON array of additional edges needed:
[{{"source":"...","source_output_name":"...","target":"...","target_field_name":"..."}}]"""

        all_nodes_summary = json.dumps(
            [{"node_id": n["node_id"], "component_type": n["component_type"],
              "outputs": [o.get("name") for o in self._get_outputs(n)][:3],
              "inputs": [i.get("name") for i in self._get_inputs(n)][:6],
              "template_vars": prompt_vars.get(n["node_id"], [])}
             for n in nodes],
            indent=2, ensure_ascii=True
        )[:2000]

        user = (
            f"All nodes:\n{all_nodes_summary}\n\n"
            f"Existing edges:\n{json.dumps(existing, ensure_ascii=True)[:800]}\n\n"
            f"Uncovered nodes (need incoming edges): "
            f"{json.dumps([n['node_id'] for n in uncovered], ensure_ascii=True)}"
        )

        resp = self.mem.call_llm(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=800, label="fill_gaps",
        )
        return self._parse_edges(resp)

    # ── Template variable enforcement ──────────────────────────────────────

    def _enforce_template_var_edges(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        prompt_vars: Dict[str, List[str]],
    ) -> List[Dict]:
        """
        CRITICAL: For every {variable} in a Prompt template, ensure there is an
        incoming edge to that variable's port from an appropriate source.

        This is the edge-case handler the user specifically asked for:
        'prompt node doesn't have a memory field initially but {memory} in the template
         creates a memory field so the memory can be connected to the prompt easily'
        """
        if not prompt_vars:
            return edges

        node_map = {n["node_id"]: n for n in nodes}
        existing_targets: Dict[str, Set[str]] = defaultdict(set)
        for e in edges:
            tgt = e.get("target", "")
            fld = e.get("target_field_name", "")
            if tgt and fld:
                existing_targets[tgt].add(fld)

        added: List[Dict] = []
        existing_quads = {
            (e.get("source"), e.get("source_output_name"),
             e.get("target"), e.get("target_field_name"))
            for e in edges
        }

        for node_id, var_names in prompt_vars.items():
            if node_id not in node_map:
                continue

            for var_name in var_names:
                # Skip if already covered
                if var_name in existing_targets.get(node_id, set()):
                    continue

                # Find the best source node for this variable
                best = self._find_best_source_for_var(
                    var_name, node_id, nodes, existing_quads
                )
                if best:
                    edge, quad = best
                    existing_quads.add(quad)
                    existing_targets[node_id].add(var_name)
                    added.append(edge)
                    if _debug():
                        print(
                            f"[Graph][TemplateVar] {var_name}: "
                            f"{edge['source']}.{edge['source_output_name']} → "
                            f"{node_id}.{var_name}"
                        )
                    self.mem.mark_knowledge_use("graph", "prompt_vars", 1)
                else:
                    self.mem.warnings.append(
                        f"Template variable '{{{var_name}}}' on '{node_id}' "
                        f"has no matching source component. Add an appropriate component."
                    )

        return edges + added

    def _find_best_source_for_var(
        self,
        var_name: str,
        target_id: str,
        nodes: List[Dict],
        existing_quads: Set[Tuple],
    ) -> Optional[Tuple[Dict, Tuple]]:
        """
        Find the best (source_node, output_name) pair for a given template variable.
        Universal: covers memory, input, document/RAG, system, tool, and any unknown variable.
        """
        vl = var_name.lower()

        # ── Semantic variable categories ───────────────────────────────────
        _MEM_VARS   = {"memory", "history", "chat_history", "conversation_history", "messages", "past_messages"}
        _INPUT_VARS = {"input", "query", "question", "user_input", "message", "human_input", "user_message", "user_query"}
        _DOC_VARS   = {"context", "document", "documents", "content", "rag_context", "retrieved",
                       "results", "data", "file", "file_content", "passage", "passages"}
        _SYS_VARS   = {"system", "instructions", "persona", "behavior", "system_prompt", "role"}
        _TOOL_VARS  = {"tools", "tool"}

        candidates: List[Tuple[int, Dict, str, str]] = []  # (score, node, output_name, out_type)

        for src_node in nodes:
            if src_node["node_id"] == target_id:
                continue

            src_type_l = _norm(src_node["component_type"])
            ck = self.mem.get_component_knowledge(src_node["component_type"])

            for out_def in self._get_outputs(src_node):
                out_name   = out_def.get("name", "output")
                out_types  = out_def.get("types") or out_def.get("output_types") or ["Message"]
                out_name_l = out_name.lower()
                score = 0

                # ── Memory/history → Memory component ──────────────────────
                if vl in _MEM_VARS:
                    if "memory" in src_type_l:
                        score += 20
                    if any(tok in out_name_l for tok in ("messages_text", "memory", "history")):
                        score += 15

                # ── Input/query → source component (ChatInput, TextInput) ───
                elif vl in _INPUT_VARS:
                    if self.mem.is_source_component(src_node["component_type"]):
                        score += 15
                    if any(tok in out_name_l for tok in ("message", "text", "output")):
                        score += 5

                # ── Document/context/RAG → File, VectorStore, ParseData ─────
                elif vl in _DOC_VARS:
                    if any(tok in src_type_l for tok in
                           ("file", "vectorstore", "vector", "document", "loader", "parsedata", "parse")):
                        score += 20
                    if any(tok in out_name_l for tok in ("text", "message", "result", "parsed", "data")):
                        score += 5
                    # ParseData parsed_text is the best RAG output
                    if "parsed" in out_name_l:
                        score += 8

                # ── System/instructions → TextInput or source ───────────────
                elif vl in _SYS_VARS:
                    if any(tok in src_type_l for tok in ("textinput", "input")):
                        score += 10
                    if self.mem.is_source_component(src_node["component_type"]):
                        score += 5
                    if any(tok in out_name_l for tok in ("text", "message")):
                        score += 3

                # ── Tools → Tool-providing components ──────────────────────
                elif vl in _TOOL_VARS:
                    if ck and ck.is_tool_provider:
                        score += 20
                    if "tool" in out_name_l:
                        score += 10

                # ── Unknown variable: semantic fallback ─────────────────────
                else:
                    # Component type contains the variable name → strong match
                    if vl in src_type_l or _norm(vl) in src_type_l:
                        score += 15
                        # Prefer first output of that component
                    # Output name is similar to variable name
                    if vl in out_name_l or out_name_l in vl:
                        score += 10
                    # Generic Message-producing sources as last resort
                    if "message" in out_types or "Message" in out_types:
                        score += 2
                    if self.mem.is_source_component(src_node["component_type"]):
                        score += 3

                # Never route LanguageModel type to a prompt template port
                if "LanguageModel" in out_types:
                    score = 0

                if score > 0:
                    candidates.append((score, src_node, out_name, out_types[0] if out_types else "Message"))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[0])
        _, best_node, best_out, _ = candidates[0]

        quad = (best_node["node_id"], best_out, target_id, var_name)
        if quad in existing_quads:
            # Try next best
            for _, node, out, _ in candidates[1:]:
                q = (node["node_id"], out, target_id, var_name)
                if q not in existing_quads:
                    edge = self._make_edge(node["node_id"], out, target_id, var_name, nodes)
                    return (edge, q) if edge else None
            return None

        edge = self._make_edge(best_node["node_id"], best_out, target_id, var_name, nodes)
        return (edge, quad) if edge else None

    # ── Flow integrity ─────────────────────────────────────────────────────

    def _ensure_flow_integrity(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """
        Guarantee:
          - Every source component has at least one outgoing edge
          - Every sink component has at least one incoming edge
          - No orphan non-source/non-sink nodes
        """
        all_sources = {e.get("source") for e in edges}
        all_targets = {e.get("target") for e in edges}
        node_map    = {n["node_id"]: n for n in nodes}
        existing_quads = {
            (e.get("source"), e.get("source_output_name"),
             e.get("target"), e.get("target_field_name"))
            for e in edges
        }

        added: List[Dict] = []

        # Source nodes must have outgoing edges
        for n in nodes:
            if not self.mem.is_source_component(n["component_type"]):
                continue
            if n["node_id"] in all_sources:
                continue
            # Find best non-source target
            for tgt in nodes:
                if tgt["node_id"] == n["node_id"]:
                    continue
                if self.mem.is_source_component(tgt["component_type"]):
                    continue
                for out_def in self._get_outputs(n):
                    out_name = out_def.get("name", "output")
                    for inp_def in self._get_inputs(tgt):
                        inp_name = inp_def.get("name", "input_value")
                        q = (n["node_id"], out_name, tgt["node_id"], inp_name)
                        if q in existing_quads:
                            continue
                        edge = self._make_edge(n["node_id"], out_name, tgt["node_id"], inp_name, nodes)
                        if edge:
                            added.append(edge)
                            existing_quads.add(q)
                            all_sources.add(n["node_id"])
                            if _debug():
                                print(f"[Graph][Integrity] Source {n['node_id']} → {tgt['node_id']}.{inp_name}")
                            break
                    if n["node_id"] in all_sources:
                        break
                if n["node_id"] in all_sources:
                    break

        # Sink nodes must have incoming edges
        for n in nodes:
            if not self.mem.is_sink_component(n["component_type"]):
                continue
            if n["node_id"] in all_targets:
                continue
            # Find best non-sink source
            for src in nodes:
                if src["node_id"] == n["node_id"]:
                    continue
                if self.mem.is_sink_component(src["component_type"]):
                    continue
                for out_def in self._get_outputs(src):
                    out_name = out_def.get("name", "output")
                    for inp_def in self._get_inputs(n):
                        inp_name = inp_def.get("name", "input_value")
                        q = (src["node_id"], out_name, n["node_id"], inp_name)
                        if q in existing_quads:
                            continue
                        edge = self._make_edge(src["node_id"], out_name, n["node_id"], inp_name, nodes)
                        if edge:
                            added.append(edge)
                            existing_quads.add(q)
                            all_targets.add(n["node_id"])
                            if _debug():
                                print(f"[Graph][Integrity] {src['node_id']} → Sink {n['node_id']}.{inp_name}")
                            break
                    if n["node_id"] in all_targets:
                        break
                if n["node_id"] in all_targets:
                    break

        if added:
            self.mem.mark_knowledge_use("graph", "flow_integrity", len(added))

        return edges + added

    # ── Edge helpers ───────────────────────────────────────────────────────

    def _make_edge(
        self,
        src_id: str,
        src_out: str,
        tgt_id: str,
        tgt_field: str,
        nodes: List[Dict],
    ) -> Optional[Dict]:
        """Build an abstract edge dict."""
        if not (src_id and src_out and tgt_id and tgt_field):
            return None
        src_node = next((n for n in nodes if n["node_id"] == src_id), None)
        tgt_node = next((n for n in nodes if n["node_id"] == tgt_id), None)

        src_out_types = ["Message"]
        tgt_in_types  = ["Message"]
        tgt_fld_type  = "str"

        if src_node:
            for o in self._get_outputs(src_node):
                if o.get("name") == src_out:
                    src_out_types = o.get("types") or o.get("output_types") or ["Message"]
                    break

        if tgt_node:
            live_tmpl = tgt_node.get("live_template", {})
            if isinstance(live_tmpl, dict) and tgt_field in live_tmpl:
                fdef = live_tmpl[tgt_field]
                if isinstance(fdef, dict):
                    tgt_in_types = fdef.get("input_types", ["Message"])
                    tgt_fld_type = fdef.get("type", "str")
            else:
                for inp in self._get_inputs(tgt_node):
                    if inp.get("name") == tgt_field:
                        tgt_in_types = inp.get("input_types") or ["Message"]
                        tgt_fld_type = inp.get("type", "str")
                        break

        return {
            "source":             src_id,
            "source_output_name": src_out,
            "source_output_types": src_out_types,
            "target":             tgt_id,
            "target_field_name":  tgt_field,
            "target_input_types": tgt_in_types,
            "target_field_type":  tgt_fld_type,
        }

    def _dedupe_edges(self, edges: List[Dict]) -> List[Dict]:
        """Remove duplicate (source, source_output_name, target, target_field_name) quads."""
        seen: Set[Tuple] = set()
        result = []
        # Target-field exclusivity: each (target, field) can have only one source
        field_owner: Dict[Tuple, str] = {}

        for e in edges:
            quad = (
                e.get("source"),
                e.get("source_output_name"),
                e.get("target"),
                e.get("target_field_name"),
            )
            if any(v is None for v in quad):
                continue
            if quad in seen:
                continue
            # Enforce target-field exclusivity
            field_key = (e.get("target"), e.get("target_field_name"))
            if field_key in field_owner:
                # Keep existing, warn on duplicate
                if _debug():
                    print(
                        f"[Graph] Duplicate target field dropped: "
                        f"{e.get('source')} → {e.get('target')}.{e.get('target_field_name')}"
                    )
                continue
            field_owner[field_key] = e.get("source", "")
            seen.add(quad)
            result.append(e)
        return result

    def _validate_edges(
        self, edges: List[Dict], valid_ids: Set[str], nodes: List[Dict]
    ) -> List[Dict]:
        """Filter out edges referencing non-existent nodes or self-loops."""
        valid = []
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            if not src or not tgt:
                continue
            if src == tgt:
                continue
            if src not in valid_ids or tgt not in valid_ids:
                continue
            valid.append(e)
        return valid

    # ── I/O helpers ────────────────────────────────────────────────────────

    def _get_outputs(self, node: Dict) -> List[Dict]:
        outputs = node.get("live_outputs", [])
        if outputs:
            return outputs
        # Try live_template outputs section
        live_tmpl = node.get("live_template", {})
        if isinstance(live_tmpl, dict):
            # Template doesn't directly contain outputs — use knowledge_index
            pass
        ck = self.mem.get_component_knowledge(node.get("component_type", ""))
        if ck:
            return [
                {"name": n, "types": ck.output_types[:2] or ["Message"]}
                for n in ck.output_names
            ]
        return [{"name": "output", "types": ["Message"]}]

    def _get_inputs(self, node: Dict) -> List[Dict]:
        inputs = node.get("live_inputs", [])
        if inputs:
            return inputs
        live_tmpl = node.get("live_template", {})
        if isinstance(live_tmpl, dict):
            result = []
            for fname, fdef in live_tmpl.items():
                if fname in ("_type", "code") or not isinstance(fdef, dict):
                    continue
                it = fdef.get("input_types", [])
                if it:
                    result.append({
                        "name": fname,
                        "input_types": it,
                        "type": fdef.get("type", "str"),
                    })
            if result:
                return result
        ck = self.mem.get_component_knowledge(node.get("component_type", ""))
        if ck:
            return [
                {"name": n, "input_types": ck.input_types_map.get(n, ["Message"])}
                for n in ck.input_names
            ]
        return [{"name": "input_value", "input_types": ["Message"]}]

    # ── Prompt variable collection ──────────────────────────────────────────

    def _collect_all_prompt_vars(self, nodes: List[Dict]) -> Dict[str, List[str]]:
        """
        Gather {variable} names from all Prompt nodes.
        Returns {node_id: [var1, var2, ...]}
        """
        result: Dict[str, List[str]] = {}

        # First check requirements (set by PlanningAgent)
        saved_vars = self.mem.requirements.get("prompt_template_variables", {})
        if isinstance(saved_vars, dict):
            result.update(saved_vars)

        for node in nodes:
            nid   = node.get("node_id", "")
            ctype = node.get("component_type", "")
            if "prompt" not in ctype.lower():
                continue

            # Extract from parameters.template
            params = node.get("parameters", {})
            template = str(params.get("template") or "")

            # Also check live_template
            if not template:
                lt = node.get("live_template", {})
                if isinstance(lt, dict) and isinstance(lt.get("template"), dict):
                    template = str(lt["template"].get("value") or "")

            vars_found = _extract_template_vars(template)
            if vars_found:
                result[nid] = list(dict.fromkeys(vars_found))  # dedup preserving order

        return result

    # ── Position assignment ────────────────────────────────────────────────

    def _assign_positions(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Assign x,y positions using a simple layered layout."""
        if not nodes:
            return nodes

        # Topological sort by edge connectivity
        outgoing: Dict[str, List[str]] = defaultdict(list)
        incoming: Dict[str, int] = defaultdict(int)
        node_ids = {n["node_id"] for n in nodes}

        for e in edges:
            s = e.get("source", "")
            t = e.get("target", "")
            if s in node_ids and t in node_ids and s != t:
                outgoing[s].append(t)
                incoming[t] += 1

        # BFS layers
        queue = [nid for nid in node_ids if incoming[nid] == 0]
        layers: List[List[str]] = []
        visited: Set[str] = set()

        while queue:
            layers.append(queue[:])
            visited.update(queue)
            next_queue = []
            for nid in queue:
                for tgt in outgoing[nid]:
                    if tgt not in visited:
                        incoming[tgt] -= 1
                        if incoming[tgt] <= 0:
                            next_queue.append(tgt)
            queue = next_queue

        # Nodes not reachable by BFS (isolated)
        remaining = [nid for nid in node_ids if nid not in visited]
        if remaining:
            layers.append(remaining)

        # Map node_id → layer index
        layer_map: Dict[str, int] = {}
        for li, layer in enumerate(layers):
            for nid in layer:
                layer_map[nid] = li

        # Assign positions
        layer_counts: Dict[int, int] = defaultdict(int)
        layer_positions: Dict[str, Tuple[int, int]] = {}

        for node in nodes:
            nid = node["node_id"]
            layer = layer_map.get(nid, len(layers))
            idx = layer_counts[layer]
            layer_counts[layer] += 1
            layer_positions[nid] = (layer, idx)

        for node in nodes:
            nid = node["node_id"]
            layer, idx = layer_positions.get(nid, (0, 0))
            node["position"] = {
                "x": 150 + layer * 380,
                "y": 80 + idx * 300,
            }

        return nodes

    # ── Misc helpers ───────────────────────────────────────────────────────

    def _catalog_ids(self) -> List[str]:
        fetcher = getattr(self.mem, "_fetcher", None)
        if fetcher and hasattr(fetcher, "_type_to_raw"):
            return list(fetcher._type_to_raw.keys())
        return [c.get("id", "") for c in self.mem.live_components if c.get("id")]

    def _parse_edges(self, response: str) -> List[Dict]:
        """Parse edge list from LLM response."""
        cleaned = re.sub(r"```(?:json)?|```", "", response or "").strip()
        try:
            m = re.search(r"\[[\s\S]*\]", cleaned)
            if m:
                items = json.loads(m.group(0))
                if isinstance(items, list):
                    result = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        src = item.get("source", "")
                        src_out = item.get("source_output_name", "output")
                        tgt = item.get("target", "")
                        tgt_fld = item.get("target_field_name", "input_value")
                        if src and tgt:
                            result.append({
                                "source":             src,
                                "source_output_name": src_out,
                                "target":             tgt,
                                "target_field_name":  tgt_fld,
                                "description":        item.get("description", ""),
                            })
                    return result
        except Exception:
            pass
        return []

    def _print_edge_summary(self, edges: List[Dict], nodes: List[Dict]) -> None:
        nmap = {n["node_id"]: n["component_type"] for n in nodes}
        print("\n[Graph] Edge summary:")
        for e in edges:
            src = nmap.get(e.get("source", ""), e.get("source", "?"))
            tgt = nmap.get(e.get("target", ""), e.get("target", "?"))
            src_out = e.get("source_output_name", "?")
            tgt_fld = e.get("target_field_name", "?")
            print(f"  {src}.{src_out} → {tgt}.{tgt_fld}")
