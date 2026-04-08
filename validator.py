from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from shared_memory import SharedMemory


_EDGE_SATISFIED_FIELDS = {"input_value", "message", "prompt", "response", "system_message"}
_SKIP_TEMPLATE_FIELDS  = {"code", "_type"}


@dataclass
class ValidationResult:
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def report(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"ERRORS ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        if not self.errors and not self.warnings:
            lines.append("✓ All checks passed.")
        return "\n".join(lines)


def _decode_handle(s: str) -> Dict[str, Any]:
    try:
        return json.loads((s or "").replace("\u0153", '"'))
    except Exception:
        return {}


def _extract_template_vars(text: str) -> List[str]:
    return re.findall(r"\{([^}]+)\}", text or "")


# ── WorkflowValidator ──────────────────────────────────────────────────────────

class WorkflowValidator:
    def __init__(self, mem: Optional[SharedMemory] = None):
        self.mem = mem

    def validate(self, workflow: Dict[str, Any]) -> ValidationResult:
        result = ValidationResult()

        # ── Top-level structure ──────────────────────────────────────────
        for key in ("data", "name"):
            if key not in workflow:
                result.errors.append(f"Missing top-level key: '{key}'")

        data = workflow.get("data", {})
        if not isinstance(data, dict):
            result.errors.append("'data' must be a dict")
            return result

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        if not isinstance(nodes, list):
            result.errors.append("'data.nodes' must be a list")
        if not isinstance(edges, list):
            result.errors.append("'data.edges' must be a list")
        if not nodes:
            result.errors.append("Workflow has no nodes.")
            return result
        if len(nodes) == 1:
            result.errors.append(
                "Only 1 node — not functional. Need at least source, processor, sink."
            )

        # ── Role completeness ─────────────────────────────────────────────
        node_types = [
            n.get("data", {}).get("type", "")
            for n in nodes if isinstance(n, dict)
        ]

        if self.mem:
            sources  = [t for t in node_types if self.mem.is_source_component(t)]
            sinks    = [t for t in node_types if self.mem.is_sink_component(t)]
            llm_nodes = [t for t in node_types if self.mem.is_llm_component(t)]

            if not sources and len(nodes) > 1:
                result.warnings.append(
                    "No source/entry-point component detected. "
                    "Add a component like ChatInput or TextInput."
                )
            if not sinks and len(nodes) > 1:
                result.warnings.append(
                    "No sink/exit-point component detected. "
                    "Add a component like ChatOutput."
                )
            if not llm_nodes and len(nodes) > 1:
                result.warnings.append(
                    "No LLM or Agent component detected — workflow may not produce AI responses."
                )
        else:
            types_l = [t.lower() for t in node_types]
            if not any("input" in t for t in types_l) and len(nodes) > 1:
                result.warnings.append("No input component detected.")
            if not any("output" in t for t in types_l) and len(nodes) > 1:
                result.warnings.append("No output component detected.")

        # ── Node structure ────────────────────────────────────────────────
        node_ids:     Set[str]             = set()
        node_type_map: Dict[str, str]      = {}
        node_by_id:   Dict[str, Dict]      = {}
        template_vars_by_node: Dict[str, List[str]] = {}

        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                result.errors.append(f"Node[{i}] is not a dict")
                continue

            nid = node.get("id", "")
            typ = node.get("data", {}).get("type", "")

            if not nid:
                result.errors.append(f"Node[{i}] missing 'id'")
            else:
                if nid in node_ids:
                    result.errors.append(f"Duplicate node id: '{nid}'")
                node_ids.add(nid)
                node_type_map[nid] = typ
                node_by_id[nid]    = node

            nd = node.get("data", {})
            if not isinstance(nd, dict):
                result.errors.append(f"Node '{nid}': missing 'data' dict")
                continue

            for fn in ("id", "type", "node"):
                if fn not in nd:
                    result.errors.append(f"Node '{nid}': data.{fn} missing")

            inner = nd.get("node", {})
            if not isinstance(inner, dict):
                result.errors.append(f"Node '{nid}': data.node not a dict")
                continue

            for fn in ("template", "outputs", "display_name"):
                if fn not in inner:
                    result.warnings.append(f"Node '{nid}': data.node.{fn} missing")

            tmpl = inner.get("template", {})
            if isinstance(tmpl, dict):
                self._check_template(nid, typ, tmpl, result)
                # Collect template variables for parity check
                template_field = tmpl.get("template", {})
                if isinstance(template_field, dict):
                    tval = template_field.get("value", "")
                    vars_found = _extract_template_vars(str(tval))
                    if vars_found:
                        template_vars_by_node[nid] = vars_found

            if not inner.get("outputs"):
                result.warnings.append(f"Node '{nid}' has no outputs defined")

            if "position" not in node:
                result.warnings.append(f"Node '{nid}' has no position")

        # ── Edge integrity ────────────────────────────────────────────────
        edge_ids:    Set[str]         = set()
        edge_srcs:   Set[str]         = set()
        edge_tgts:   Set[str]         = set()
        field_owner: Dict[Tuple[str, str], str] = {}   # (tgt, field) → source
        incoming_fields: Dict[str, Set[str]] = {}
        successors:  Dict[str, Set[str]] = {nid: set() for nid in node_ids}

        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                result.errors.append(f"Edge[{i}] is not a dict")
                continue

            eid = edge.get("id", f"edge_{i}")
            if eid in edge_ids:
                result.warnings.append(f"Duplicate edge id: '{eid}'")
            edge_ids.add(eid)

            src = edge.get("source", "")
            tgt = edge.get("target", "")

            # Extract target field name
            tgt_field = edge.get("data", {}).get("targetHandle", {}).get("fieldName", "")
            if not tgt_field:
                tgt_field = _decode_handle(edge.get("targetHandle", "")).get("fieldName", "")

            edge_srcs.add(src)
            edge_tgts.add(tgt)

            if not src:
                result.errors.append(f"Edge[{i}] missing 'source'")
            elif src not in node_ids:
                result.errors.append(f"Edge source '{src}' is not a valid node id")
            if not tgt:
                result.errors.append(f"Edge[{i}] missing 'target'")
            elif tgt not in node_ids:
                result.errors.append(f"Edge target '{tgt}' is not a valid node id")
            if src and tgt and src == tgt:
                result.errors.append(f"Self-loop on node '{src}'")

            if src and tgt and tgt_field:
                key = (tgt, tgt_field)
                prior = field_owner.get(key)
                if prior and prior != src:
                    result.errors.append(
                        f"Target-field exclusivity violation: '{tgt}.{tgt_field}' "
                        f"receives edges from both '{prior}' and '{src}'"
                    )
                else:
                    field_owner[key] = src
                incoming_fields.setdefault(tgt, set()).add(tgt_field)

            if src and tgt and src in successors:
                successors[src].add(tgt)

        # ── Connectivity — orphan detection ───────────────────────────────
        if node_ids and edges:
            for nid in node_ids:
                typ = node_type_map.get(nid, "")
                is_src  = self.mem.is_source_component(typ) if self.mem else "input" in typ.lower()
                is_sink = self.mem.is_sink_component(typ)   if self.mem else "output" in typ.lower()

                if not is_src and nid not in edge_tgts:
                    result.errors.append(
                        f"Orphan node '{nid}' ({typ}): no incoming edges. "
                        "Connect it or remove it."
                    )
                if not is_sink and nid not in edge_srcs:
                    result.warnings.append(
                        f"Node '{nid}' ({typ}) has no outgoing edges. "
                        "Ensure it feeds into a downstream component."
                    )

        # ── Reachability from source(s) ────────────────────────────────────
        if node_ids and edges and self.mem:
            source_ids = {
                nid for nid, typ in node_type_map.items()
                if self.mem.is_source_component(typ)
            }
            if source_ids:
                reachable: Set[str] = set()
                queue = deque(source_ids)
                while queue:
                    nid = queue.popleft()
                    if nid in reachable:
                        continue
                    reachable.add(nid)
                    for suc in successors.get(nid, set()):
                        queue.append(suc)
                unreachable = node_ids - reachable - source_ids
                for nid in unreachable:
                    result.warnings.append(
                        f"Node '{nid}' ({node_type_map.get(nid,'?')}) "
                        "is not reachable from any source node."
                    )

        # ── Template variable parity ───────────────────────────────────────
        for nid, var_names in template_vars_by_node.items():
            node_incoming = incoming_fields.get(nid, set())
            for var in var_names:
                if var not in node_incoming:
                    result.warnings.append(
                        f"Template variable '{{{var}}}' in node '{nid}' "
                        f"has no incoming edge to field '{var}'. "
                        "Add the matching connection."
                    )

        return result

    def _check_template(
        self,
        nid: str,
        comp_type: str,
        tmpl: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Check required template fields have values."""
        ck = self.mem.get_component_knowledge(comp_type) if self.mem else None

        for fname, fdef in tmpl.items():
            if fname in _SKIP_TEMPLATE_FIELDS:
                continue
            if not isinstance(fdef, dict):
                continue
            if not fdef.get("required", False):
                continue
            val = fdef.get("value")
            if val in (None, "", [], {}):
                # Skip edge-fed fields — they get their value at runtime
                if fname in _EDGE_SATISFIED_FIELDS:
                    continue
                # Check if this field gets an incoming edge
                # (we can't know here — warn softly)
                result.warnings.append(
                    f"Node '{nid}' ({comp_type}): required field '{fname}' has no value. "
                    "Ensure it is filled or fed by an edge."
                )
