from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class ComponentKnowledge:
    workflow_type:    str
    display_name:     str
    description:      str
    required_fields:  List[str]
    key_fields:       List[Dict]
    output_names:     List[str]
    input_names:      List[str]
    output_types:     List[str]
    input_types_map:  Dict[str, List[str]]
    is_source:        bool = False
    is_sink:          bool = False
    is_llm:           bool = False
    is_tool_provider: bool = False


@dataclass
class SharedMemory:
    user_prompt:          str = ""
    conversation_history: List[Dict[str, str]]  = field(default_factory=list)
    live_components:      List[Dict[str, Any]]  = field(default_factory=list)
    example_flows:        List[Dict[str, Any]]  = field(default_factory=list)
    mcp_tools:            List[Dict[str, Any]]  = field(default_factory=list)
    starter_projects:     List[Dict[str, Any]]  = field(default_factory=list)
    knowledge_index:      Dict[str, ComponentKnowledge] = field(default_factory=dict)
    requirements:         Dict[str, Any]        = field(default_factory=dict)
    selected_components:  List[Dict[str, Any]]  = field(default_factory=list)
    abstract_graph:       Dict[str, Any]        = field(default_factory=dict)
    final_workflow:       Dict[str, Any]        = field(default_factory=dict)
    workflow_path:        str = "langflow_workflow.json"
    graph_dirty:               bool = False
    graph_rebuild_from_selection: bool = False
    errors:               List[str] = field(default_factory=list)
    warnings:             List[str] = field(default_factory=list)
    # ── Agent knowledge tracking ───────────────────────────────────────────
    # Tracks which knowledge sources each agent actually used (for debugging)
    agent_knowledge_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})

    def call_llm(
        self,
        messages: "list[dict]",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: Optional[int] = None,
        label: str = "",
    ) -> str:
        """
        Central LLM call wrapper used by ALL agents.
        - Strips <think> blocks so they appear only once in terminal.
        - Enforces max_tokens to prevent truncated JSON.
        - Logs which agent made the call.
        """
        from llm_call import call_local_llm
        import os

        if timeout is None:
            timeout = int(os.getenv("LOCAL_LLM_TIMEOUT", "300"))
        if isinstance(timeout, int) and timeout <= 0:
            timeout = None

        if label:
            import sys
            print(f"  [LLM:{label}]", end="", flush=True, file=sys.stderr)

        result = call_local_llm(messages, temperature=temperature,
                                max_tokens=max_tokens, timeout=timeout)
        return result

    def track_knowledge(self, agent: str, key: str, count: int = 1) -> None:
        """Record that an agent used a specific knowledge source."""
        if agent not in self.agent_knowledge_usage:
            self.agent_knowledge_usage[agent] = {}
        self.agent_knowledge_usage[agent][key] = (
            self.agent_knowledge_usage[agent].get(key, 0) + count
        )

    # Alias used by agents (mark_knowledge_use == track_knowledge)
    def mark_knowledge_use(self, agent: str, key: str, count: int = 1) -> None:
        self.track_knowledge(agent, key, count)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "user_prompt": self.user_prompt,
            "requirements": self.requirements,
            "selected_components_count": len(self.selected_components),
            "abstract_graph_node_count": len(self.abstract_graph.get("nodes", [])),
            "abstract_graph_edge_count": len(self.abstract_graph.get("edges", [])),
            "errors": self.errors, "warnings": self.warnings,
        }

    def clone(self) -> "SharedMemory":
        return copy.deepcopy(self)

    def reset_transient_state(self, *, preserve_user_prompt: bool = True) -> None:
        user_prompt   = self.user_prompt if preserve_user_prompt else ""
        workflow_path = self.workflow_path
        self.user_prompt   = user_prompt
        self.workflow_path = workflow_path
        self.conversation_history.clear()
        self.requirements.clear()
        self.selected_components.clear()
        self.abstract_graph.clear()
        self.final_workflow.clear()
        self.graph_dirty = False
        self.graph_rebuild_from_selection = False
        self.errors.clear()
        self.warnings.clear()

    # ── Knowledge index ────────────────────────────────────────────────────

    def build_knowledge_index(self, fetcher: Any) -> None:
        """
        Build ComponentKnowledge from live fetcher._type_to_raw.
        Roles are derived structurally and from usage stats — NO hardcoded names.
        """
        if not fetcher or not hasattr(fetcher, "_type_to_raw"):
            return

        sink_types: Set[str]   = set()
        source_types: Set[str] = set()
        self._compute_role_stats(sink_types, source_types)

        index: Dict[str, ComponentKnowledge] = {}

        for wf_type, raw in fetcher._type_to_raw.items():
            if not isinstance(raw, dict):
                continue

            required: List[str] = []
            key_fields: List[Dict] = []
            tmpl = raw.get("template", {})
            if isinstance(tmpl, dict):
                for fname, fdef in tmpl.items():
                    if fname in ("_type", "code") or not isinstance(fdef, dict):
                        continue
                    if fdef.get("required"):
                        required.append(fname)
                    if fdef.get("show", True) is not False:
                        options = fdef.get("options", [])
                        key_fields.append({
                            "name":     fname,
                            "type":     fdef.get("type", "str"),
                            "default":  str(fdef.get("value", fdef.get("default", "")))[:40],
                            "options":  options[:6] if options else [],
                            "info":     (fdef.get("info") or "")[:60],
                            "required": bool(fdef.get("required")),
                        })

            output_names: List[str] = []
            output_types_all: List[str] = []
            for o in raw.get("outputs", []):
                if isinstance(o, dict) and o.get("name"):
                    output_names.append(o["name"])
                    for t in (o.get("types") or o.get("output_types") or []):
                        if t not in output_types_all:
                            output_types_all.append(t)

            input_names: List[str] = []
            input_types_map: Dict[str, List[str]] = {}
            if isinstance(tmpl, dict):
                for fname, fdef in tmpl.items():
                    if fname in ("_type", "code") or not isinstance(fdef, dict):
                        continue
                    it = fdef.get("input_types", [])
                    if it:
                        input_names.append(fname)
                        input_types_map[fname] = it

            # ── Role detection — STATISTICAL only (no structural guessing) ──────
            # Only components that CONSISTENTLY appear as sources in real workflows
            # are flagged as sources. Structural "no inputs" is NOT sufficient
            # because hundreds of config components (Notion, APIRequest, etc.) have
            # no connectable inputs but are NOT workflow entry points.
            is_source = wf_type in source_types

            # sink: statistically always a target and not a source.
            is_sink = (wf_type in sink_types) and (not is_source)
            # llm: produces LanguageModel type OR description indicates LLM/Agent
            desc_lower = (raw.get("description") or "").lower()
            dn_lower   = (raw.get("display_name")  or "").lower()
            is_llm = (
                "LanguageModel" in output_types_all
                or "language model" in desc_lower
                or "language model" in dn_lower
                or ("agent" in dn_lower and "Message" in output_types_all)
            )
            # tool provider: produces Tool type
            is_tool_provider = "Tool" in output_types_all

            index[wf_type] = ComponentKnowledge(
                workflow_type    = wf_type,
                display_name     = raw.get("display_name", wf_type),
                description      = (raw.get("description") or "")[:120].replace("\n", " "),
                required_fields  = required,
                key_fields       = key_fields[:20],
                output_names     = output_names,
                input_names      = input_names,
                output_types     = output_types_all,
                input_types_map  = input_types_map,
                is_source        = is_source,
                is_sink          = is_sink,
                is_llm           = is_llm,
                is_tool_provider = is_tool_provider,
            )

        self.knowledge_index = index

    @staticmethod
    def _unwrap_flow_data(ex: Dict[str, Any]) -> tuple:
        """
        Extract (nodes, edges) from a flow dict regardless of nesting format.

        Langflow flows come in two formats from different API endpoints:
          Format A (example flows):  {"data": {"nodes": [...], "edges": [...]}}
          Format B (starter/dataset): {"workflow": {"data": {"nodes": [...], "edges": [...]}}}

        Both are handled by unwrapping one level with get("workflow") or get("data"),
        then checking both the top level AND a nested "data" key.
        """
        top = ex.get("workflow") or ex.get("data") or {}
        if not isinstance(top, dict):
            return [], []

        # Format A: top IS {nodes, edges}
        nodes = top.get("nodes", [])
        edges = top.get("edges", [])

        # Format B: top has a nested "data" key containing {nodes, edges}
        if not nodes and not edges:
            inner = top.get("data", {})
            if isinstance(inner, dict):
                nodes = inner.get("nodes", [])
                edges = inner.get("edges", [])

        return (nodes or []), (edges or [])

    def _compute_role_stats(self, sink_types: Set[str], source_types: Set[str]) -> None:
        """
        Derive source/sink roles from REAL workflow usage patterns.
        A component is a source if it consistently has no incoming edges (>=70%).
        A component is a sink if it consistently has no outgoing edges (>=70%).
        Falls back to structural detection when examples provide insufficient signal.
        """
        type_is_source: Dict[str, int] = {}
        type_is_sink:   Dict[str, int] = {}
        type_total:     Dict[str, int] = {}

        for ex in self.example_flows:
            nodes, edges = self._unwrap_flow_data(ex)
            if not edges:
                continue

            all_srcs = {e.get("source", "") for e in edges}
            all_tgts = {e.get("target", "") for e in edges}

            for n in nodes:
                nid = n.get("id") or n.get("data", {}).get("id", "")
                t   = n.get("data", {}).get("type", "")
                if not t or t in ("note", "noteNode"):
                    continue
                type_total[t] = type_total.get(t, 0) + 1
                if nid in all_srcs and nid not in all_tgts:
                    type_is_source[t] = type_is_source.get(t, 0) + 1
                if nid not in all_srcs and nid in all_tgts:
                    type_is_sink[t] = type_is_sink.get(t, 0) + 1

        for t, total in type_total.items():
            if total >= 2:
                if type_is_source.get(t, 0) / total >= 0.7:
                    source_types.add(t)
                if type_is_sink.get(t, 0) / total >= 0.7:
                    sink_types.add(t)

    # ── Role-based query helpers (NO hardcoded type name strings) ──────────

    def get_source_types(self) -> List[str]:
        return [wt for wt, ck in self.knowledge_index.items() if ck.is_source]

    def get_sink_types(self) -> List[str]:
        return [wt for wt, ck in self.knowledge_index.items() if ck.is_sink]

    def get_llm_types(self) -> List[str]:
        return [wt for wt, ck in self.knowledge_index.items() if ck.is_llm]

    def is_source_component(self, workflow_type: str) -> bool:
        ck = self.get_component_knowledge(workflow_type)
        return ck.is_source if ck else False

    def is_sink_component(self, workflow_type: str) -> bool:
        ck = self.get_component_knowledge(workflow_type)
        return ck.is_sink if ck else False

    def is_llm_component(self, workflow_type: str) -> bool:
        ck = self.get_component_knowledge(workflow_type)
        return ck.is_llm if ck else False

    def get_component_knowledge(self, workflow_type: str) -> Optional[ComponentKnowledge]:
        if workflow_type in self.knowledge_index:
            return self.knowledge_index[workflow_type]
        norm = re.sub(r"[^a-z0-9]", "", workflow_type.lower())
        for wt, ck in self.knowledge_index.items():
            if re.sub(r"[^a-z0-9]", "", wt.lower()) == norm:
                return ck
        return None

    def knowledge_summary_for(self, workflow_types: List[str], max_fields: int = 8) -> str:
        lines = []
        for wt in workflow_types:
            ck = self.get_component_knowledge(wt)
            if not ck:
                lines.append(f"{wt}: (no live knowledge)")
                continue
            req_str = ",".join(ck.required_fields) if ck.required_fields else "none"
            out_str = ",".join(f"{n}[{'/'.join(ck.output_types[:2])}]" for n in ck.output_names[:4])
            in_str  = ",".join(
                f"{n}[{'/'.join(ck.input_types_map.get(n, ['?'])[:2])}]"
                for n in ck.input_names[:6]
            )
            roles = []
            if ck.is_source:         roles.append("source")
            if ck.is_sink:           roles.append("sink")
            if ck.is_llm:            roles.append("llm")
            if ck.is_tool_provider:  roles.append("tool")
            lines.append(f"## {wt} [{','.join(roles)}] — {ck.description[:80]}")
            lines.append(f"   required={req_str}")
            lines.append(f"   outputs: {out_str or 'none'}")
            lines.append(f"   inputs:  {in_str or 'none'}")
            for f in ck.key_fields[:max_fields]:
                opts = f" options={f['options']}" if f["options"] else ""
                req  = " [REQ]" if f["required"] else ""
                lines.append(
                    f"   field {f['name']}({f['type']}){req} "
                    f"default={f['default']!r}{opts}  {f['info']}"
                )
        return "\n".join(lines)

    def edge_io_summary(self, node_ids: Optional[List[str]] = None) -> str:
        nodes = self.abstract_graph.get("nodes") or []
        if not nodes:
            nodes = [
                {"node_id": c.get("node_id",""), "component_type": c.get("component_type","")}
                for c in self.selected_components
            ]
        lines = []
        for n in nodes:
            nid = n.get("node_id", "")
            wt  = n.get("component_type", "")
            if node_ids and nid not in node_ids:
                continue
            ck = self.get_component_knowledge(wt)
            if not ck:
                lines.append(f"{nid} ({wt}): no live I/O data")
                continue
            out_str = ", ".join(
                f"{on}[{'/'.join(ck.output_types[:2])}]" for on in ck.output_names[:4]
            )
            in_str = ", ".join(
                f"{fn}[{'/'.join(ck.input_types_map.get(fn,['?'])[:2])}]"
                for fn in ck.input_names[:6]
            )
            lines.append(f"{nid} ({wt})")
            if out_str: lines.append(f"  OUT: {out_str}")
            if in_str:  lines.append(f"  IN:  {in_str}")
        return "\n".join(lines)