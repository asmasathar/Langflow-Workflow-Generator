from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from llm_call import call_local_llm
from shared_memory import SharedMemory


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _is_placeholder(name: str) -> bool:
    n = _norm(name or "")
    return n in {
        "", "typeid", "workflowtype", "workflowtypeid",
        "componenttype", "componentid", "exactworkflowtype",
        "unknowntype", "none", "null",
    }


def _token_overlap(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z]+", a.lower()))
    tb = set(re.findall(r"[a-z]+", b.lower()))
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0


class ComponentResolver:
    """
    Resolves user/LLM component type names to exact Langflow WorkflowType IDs
    and hydrates SharedMemory.selected_components with live catalog metadata.
    """

    def __init__(self, mem: SharedMemory, rag: Optional[Any] = None):
        self.mem = mem
        self.rag = rag
        self._wf_types: List[str] = []
        self._norm_map: Dict[str, str] = {}  # normalised -> WorkflowType

        # Try fetcher's type index first (most reliable source)
        fetcher = getattr(mem, "_fetcher", None)
        if fetcher and hasattr(fetcher, "_type_to_raw") and fetcher._type_to_raw:
            self._wf_types = list(fetcher._type_to_raw.keys())
        else:
            # Fall back: collect from example flows + live_components
            seen: set = set()
            for ex in mem.example_flows:
                wf = ex.get("workflow") or ex.get("data") or {}
                nodes = wf.get("data", {}).get("nodes", []) if isinstance(wf, dict) else []
                for n in nodes:
                    t = n.get("data", {}).get("type", "")
                    if t and t not in ("note", "noteNode") and t not in seen:
                        seen.add(t)
                        self._wf_types.append(t)
            for comp in mem.live_components:
                cid = comp.get("id", "")
                if cid and cid not in seen:
                    seen.add(cid)
                    self._wf_types.append(cid)

        for wt in self._wf_types:
            self._norm_map[_norm(wt)] = wt

    # ── Public API ─────────────────────────────────────────────────────────

    def resolve(self, component_type: str) -> Optional[str]:
        """Return the canonical WorkflowType ID for the given name, or None."""
        if not component_type:
            return None
        if _is_placeholder(component_type):
            return None

        # 1. Exact
        if component_type in self._norm_map.values():
            return component_type

        norm_ct = _norm(component_type)

        # 2. Normalised exact
        if norm_ct in self._norm_map:
            return self._norm_map[norm_ct]

        # 3. Substring + token-overlap scoring
        candidates: List[Tuple[str, float]] = []
        for n, wt in self._norm_map.items():
            # Substring containment
            if norm_ct in n or n in norm_ct:
                score = 0.7 + 0.2 * (len(min(n, norm_ct, key=len)) / max(len(n), len(norm_ct), 1))
                candidates.append((wt, score))

        if not candidates:
            for wt in self._wf_types:
                score = max(
                    _token_overlap(component_type, wt),
                    _token_overlap(component_type, wt.replace("Component", "").replace("Model", "")),
                )
                if score >= 0.4:
                    candidates.append((wt, score))

        if candidates:
            return max(candidates, key=lambda x: x[1])[0]

        # 4. RAG-guided fuzzy
        rag_guess = self._rag_resolve(component_type)
        if rag_guess:
            return rag_guess

        # 5. LLM fallback
        return self._llm_resolve(component_type)

    def resolve_all_components(self) -> None:
        """
        Resolve all component types in mem.selected_components and hydrate
        each component with live template/input/output metadata from the catalog.
        """
        resolved = []
        fetcher = getattr(self.mem, "_fetcher", None)

        for comp in self.mem.selected_components:
            original = comp.get("component_type", "")
            resolved_type = self.resolve(original)

            if resolved_type is None:
                self.mem.errors.append(
                    f"Cannot resolve component type '{original}' to any catalog type — skipped."
                )
                continue

            comp = dict(comp)
            if resolved_type != original:
                self.mem.warnings.append(
                    f"Component type '{original}' resolved to '{resolved_type}'"
                )
                comp["component_type"] = resolved_type
                # Sync node_id prefix
                old_id = comp.get("node_id", "")
                if old_id and "-" in old_id:
                    suffix = old_id.split("-", 1)[1]
                    comp["node_id"] = f"{resolved_type}-{suffix}"

            # Hydrate with live catalog metadata
            if fetcher and hasattr(fetcher, "get_component_by_type"):
                raw = fetcher.get_component_by_type(resolved_type)
                if raw and isinstance(raw, dict):
                    comp["live_template"] = raw.get("template", {})
                    comp["live_inputs"]   = raw.get("inputs", []) or raw.get("parameters", [])
                    comp["live_outputs"]  = raw.get("outputs", [])
                    comp["base_classes"]  = raw.get("base_classes", [])
                    comp["description"]   = comp.get("description") or raw.get("description", "")
                    comp["display_name"]  = comp.get("display_name") or raw.get("display_name", resolved_type)

            resolved.append(comp)

        self.mem.selected_components = resolved

        if not resolved:
            self.mem.errors.append(
                "No valid components after resolution. "
                "Verify that component names match your Langflow catalog."
            )

    # ── Private helpers ────────────────────────────────────────────────────

    def _rag_resolve(self, component_type: str) -> Optional[str]:
        if not self.rag or not getattr(self.rag, "is_built", False):
            return None
        try:
            context = self.rag.query(component_type, top_k=5, kinds=["component_fact"]) or ""
        except Exception:
            return None
        if not context:
            return None

        found = re.findall(r"\[COMPONENT:\s*([^\]]+)\]", context)
        candidates: List[Tuple[str, float]] = []
        for name in found:
            wt = name.strip()
            if wt in self._norm_map.values():
                score = max(
                    _token_overlap(component_type, wt),
                    _token_overlap(component_type, wt.replace("Component", "")),
                )
                candidates.append((wt, score))

        if not candidates:
            return None
        best, score = max(candidates, key=lambda x: x[1])
        return best if score >= 0.2 else None

    def _llm_resolve(self, component_type: str) -> Optional[str]:
        catalog_list = "\n".join(f"- {t}" for t in self._wf_types[:250])
        system = f"""You are a Langflow component name resolver.
These are ALL valid workflow component types:
{catalog_list}

Return ONLY the single best-matching type string from the list above.
If nothing is close, return: NONE
No explanation, no markdown, no quotes."""

        response = call_local_llm([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Resolve this component name: {component_type}"},
        ], temperature=0.0, max_tokens=80)

        resolved = (response or "").strip().strip('"').strip("'")
        if not resolved or resolved.upper() == "NONE" or _is_placeholder(resolved):
            return None
        if resolved in self._norm_map.values():
            return resolved
        return self._norm_map.get(_norm(resolved))
