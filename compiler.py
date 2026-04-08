from __future__ import annotations

import json
import os
import re
import string
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from shared_memory import SharedMemory


def _rand(n: int = 5) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _encode_handle(handle_dict: Dict[str, Any]) -> str:
    """
    Encode a handle dict to Langflow's edge handle format.
    Langflow uses œ (U+0153) instead of " in handle strings.
    """
    raw = json.dumps(handle_dict, separators=(",", ":"), ensure_ascii=False)
    return raw.replace('"', "\u0153")


def _extract_template_vars(text: str) -> List[str]:
    """Extract {variable} names from a prompt template string."""
    import re
    return re.findall(r"\{([^}]+)\}", text or "")


# ── WorkflowCompiler ───────────────────────────────────────────────────────────

class WorkflowCompiler:
    def __init__(self, mem: SharedMemory):
        self.mem = mem
        self.base_url   = os.getenv("LANGFLOW_BASE_URL", "http://localhost:7860").rstrip("/")
        self.api_key    = os.getenv("LANGFLOW_API_KEY")
        self.timeout    = int(os.getenv("LANGFLOW_TIMEOUT", "30"))
        self.verify_ssl = os.getenv("LANGFLOW_VERIFY_SSL", "true").lower() not in {"0","false","no"}
        self._raw_catalog: Optional[Dict[str, Any]] = None

    # ── Public entry point ────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "=" * 60)
        print("  COMPILER — Langflow JSON Assembly")
        print("=" * 60)

        if not self.mem.abstract_graph:
            raise RuntimeError("No abstract graph. Run GraphAgent first.")

        print("[Compiler] Fetching live component catalog...", end=" ", flush=True)
        try:
            self._raw_catalog = self._fetch_raw_catalog()
            print("OK")
        except Exception as exc:
            print(f"WARN: {exc}")
            self._raw_catalog = {}

        abstract_nodes = self.mem.abstract_graph.get("nodes", [])
        abstract_edges = self.mem.abstract_graph.get("edges", [])

        print("[Compiler] Building nodes...", end=" ", flush=True)
        compiled_nodes = [self._compile_node(n) for n in abstract_nodes]
        print(f"{len(compiled_nodes)} nodes")

        print("[Compiler] Building edges...", end=" ", flush=True)
        compiled_edges = self._compile_edges(abstract_edges, compiled_nodes)
        print(f"{len(compiled_edges)} edges")

        workflow = self._assemble_workflow(compiled_nodes, compiled_edges)
        self.mem.final_workflow = workflow

        self._save(workflow, self.mem.workflow_path)
        print(f"\n[Compiler] ✓ Saved to: {self.mem.workflow_path}")

    # ── Node compilation ──────────────────────────────────────────────────

    def _compile_node(self, abstract_node: Dict[str, Any]) -> Dict[str, Any]:
        node_id        = abstract_node["node_id"]
        component_type = abstract_node["component_type"]
        position       = abstract_node.get("position", {"x": 100, "y": 100})
        user_params    = self._collect_params(abstract_node)

        live_comp = self._get_live_component(component_type)

        if live_comp:
            template     = self._build_template_from_live(live_comp, user_params)
            outputs      = self._build_outputs_from_live(live_comp)
            base_classes = live_comp.get("base_classes", [])
            description  = live_comp.get("description", abstract_node.get("description", ""))
            display_name = (
                abstract_node.get("display_name")
                or live_comp.get("display_name")
                or component_type
            )
            output_types = [
                o.get("selected", (o.get("types") or ["Message"])[0])
                for o in outputs if isinstance(o, dict)
            ]
        else:
            template     = self._build_template_from_abstract(abstract_node, user_params)
            outputs      = self._build_outputs_from_abstract(abstract_node)
            base_classes = abstract_node.get("base_classes", [])
            description  = abstract_node.get("description", "")
            display_name = abstract_node.get("display_name", component_type)
            output_types = []
            self.mem.warnings.append(
                f"Component '{component_type}' not in live catalog — using abstract fallback."
            )

        node_data = {
            "base_classes":     base_classes,
            "beta":             False,
            "conditional_paths": [],
            "custom_fields":    {},
            "description":      description,
            "display_name":     display_name,
            "documentation":    "",
            "edited":           False,
            "field_order":      list(user_params.keys()),
            "frozen":           False,
            "icon":             self._icon_for(component_type),
            "legacy":           False,
            "lf_version":       self._lf_version(),
            "metadata":         {},
            "output_types":     output_types,
            "outputs":          outputs,
            "pinned":           False,
            "template":         template,
        }

        return {
            "data": {
                "description":   description,
                "display_name":  display_name,
                "id":            node_id,
                "node":          node_data,
                "selected_output": outputs[0]["name"] if outputs else "",
                "type":          component_type,
            },
            "dragging":         False,
            "height":           400,
            "id":               node_id,
            "measured":         {"height": 400, "width": 320},
            "position":         position,
            "positionAbsolute": position,
            "selected":         False,
            "type":             "genericNode",
            "width":            320,
        }

    def _collect_params(self, abstract_node: Dict[str, Any]) -> Dict[str, Any]:
        """Merge agent parameters with non-empty live_template values."""
        merged: Dict[str, Any] = {}

        live_template = abstract_node.get("live_template", {})
        if isinstance(live_template, dict):
            for fname, fdef in live_template.items():
                if fname in ("_type", "code") or not isinstance(fdef, dict):
                    continue
                val = fdef.get("value")
                if val not in (None, "", [], {}):
                    merged[fname] = val

        params = abstract_node.get("parameters", {})
        if isinstance(params, dict):
            for k, v in params.items():
                if v not in (None, "", [], {}):
                    merged[k] = v

        return merged

    def _build_template_from_live(
        self, live_comp: Dict[str, Any], user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        template = {"_type": "Component"}
        live_template = live_comp.get("template", {})

        if isinstance(live_template, dict):
            for field_name, field_def in live_template.items():
                if field_name == "_type":
                    continue
                if not isinstance(field_def, dict):
                    continue
                field_copy = dict(field_def)
                if field_name in user_params:
                    field_copy["value"] = user_params[field_name]
                template[field_name] = field_copy

            # Preserve dynamic user fields (e.g., prompt template variables)
            for pname, pvalue in user_params.items():
                if pname in template or pname in ("_type", "code"):
                    continue
                template[pname] = {
                    "name":         pname,
                    "type":         "str",
                    "value":        pvalue,
                    "display_name": pname,
                    "show":         True,
                    "required":     False,
                    "dynamic":      True,
                    "advanced":     False,
                    "input_types":  ["Message", "Text", "Data"],
                }

        # CRITICAL: Pre-populate dynamic fields from {variable} placeholders in
        # the template text. Langflow does NOT auto-generate these at import time —
        # they must be present in the JSON for the input ports to exist on the node.
        template_text_field = template.get("template", {})
        if isinstance(template_text_field, dict):
            template_text = str(template_text_field.get("value") or "")
            for var_name in _extract_template_vars(template_text):
                if var_name in template or var_name in ("_type", "code"):
                    continue
                template[var_name] = {
                    "name":         var_name,
                    "type":         "str",
                    "value":        "",
                    "display_name": var_name,
                    "show":         True,
                    "required":     False,
                    "dynamic":      True,
                    "advanced":     False,
                    "field_type":   "str",
                    "input_types":  ["Message", "Text", "Data"],
                }
        else:
            # Build from parameters / inputs list
            params = live_comp.get("parameters") or live_comp.get("inputs") or []
            for param in params:
                if not isinstance(param, dict):
                    continue
                fname = param.get("name", "")
                if not fname or fname == "code":
                    continue
                template[fname] = {
                    "advanced":     param.get("advanced", False),
                    "display_name": param.get("display_name", fname),
                    "dynamic":      param.get("dynamic", False),
                    "info":         param.get("info", ""),
                    "list":         param.get("list", False),
                    "load_from_db": param.get("load_from_db", False),
                    "multiline":    param.get("multiline", False),
                    "name":         fname,
                    "password":     param.get("password", False),
                    "placeholder":  param.get("placeholder", ""),
                    "required":     param.get("required", False),
                    "show":         param.get("show", True) if param.get("show") is not None else True,
                    "title_case":   param.get("title_case", False),
                    "type":         param.get("type", "str"),
                    "value":        user_params.get(fname, param.get("value", param.get("default", ""))),
                }
                if param.get("options"):
                    template[fname]["options"] = param["options"]
                if param.get("input_types"):
                    template[fname]["input_types"] = param["input_types"]

        return template

    def _build_template_from_abstract(
        self, abstract_node: Dict[str, Any], user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        template = {"_type": "Component"}
        live_template = abstract_node.get("live_template", {})
        if isinstance(live_template, dict):
            for fname, fdef in live_template.items():
                if fname in ("_type", "code"):
                    continue
                if isinstance(fdef, dict):
                    field_copy = dict(fdef)
                    if fname in user_params:
                        field_copy["value"] = user_params[fname]
                    template[fname] = field_copy
        for k, v in user_params.items():
            if k not in template:
                template[k] = {
                    "name": k, "type": "str", "value": v,
                    "display_name": k, "show": True, "required": False,
                    "dynamic": True, "input_types": ["Message", "Text", "Data"],
                }
        # Pre-populate dynamic fields from {variable} placeholders in template text
        template_text_field = template.get("template", {})
        if isinstance(template_text_field, dict):
            template_text = str(template_text_field.get("value") or "")
            for var_name in _extract_template_vars(template_text):
                if var_name in template or var_name in ("_type", "code"):
                    continue
                template[var_name] = {
                    "name":         var_name,
                    "type":         "str",
                    "value":        "",
                    "display_name": var_name,
                    "show":         True,
                    "required":     False,
                    "dynamic":      True,
                    "advanced":     False,
                    "field_type":   "str",
                    "input_types":  ["Message", "Text", "Data"],
                }
        return template

    def _build_outputs_from_live(self, live_comp: Dict[str, Any]) -> List[Dict[str, Any]]:
        outputs = live_comp.get("outputs", [])
        if isinstance(outputs, list) and outputs:
            result = []
            for o in outputs:
                if not isinstance(o, dict):
                    continue
                types = o.get("types") or o.get("output_types") or ["Message"]
                result.append({
                    "allows_loop":   o.get("allows_loop", False),
                    "cache":         o.get("cache", True),
                    "display_name":  o.get("display_name", o.get("name", "")),
                    "group_outputs": o.get("group_outputs", False),
                    "method":        o.get("method", ""),
                    "name":          o.get("name", ""),
                    "selected":      o.get("selected", types[0] if types else "Message"),
                    "tool_mode":     o.get("tool_mode", False),
                    "types":         types,
                    "value":         o.get("value", "__UNDEFINED__"),
                })
            return result
        return self._default_output()

    def _build_outputs_from_abstract(self, abstract_node: Dict[str, Any]) -> List[Dict[str, Any]]:
        outputs = abstract_node.get("live_outputs", [])
        if outputs:
            return self._build_outputs_from_live({"outputs": outputs})
        ck = self.mem.get_component_knowledge(abstract_node.get("component_type", ""))
        if ck and ck.output_names:
            result = []
            for name in ck.output_names:
                types = ck.output_types[:2] or ["Message"]
                result.append({
                    "allows_loop": False, "cache": True,
                    "display_name": name, "group_outputs": False,
                    "method": "", "name": name,
                    "selected": types[0], "tool_mode": False,
                    "types": types, "value": "__UNDEFINED__",
                })
            return result
        return self._default_output()

    def _default_output(self) -> List[Dict[str, Any]]:
        return [{
            "allows_loop": False, "cache": True, "display_name": "Output",
            "group_outputs": False, "method": "", "name": "output",
            "selected": "Message", "tool_mode": False, "types": ["Message"],
            "value": "__UNDEFINED__",
        }]

    # ── Edge compilation ──────────────────────────────────────────────────

    def _compile_edges(
        self, abstract_edges: List[Dict], compiled_nodes: List[Dict]
    ) -> List[Dict]:
        node_map = {n["id"]: n for n in compiled_nodes}
        compiled = []

        for edge in abstract_edges:
            src_id       = edge.get("source", "")
            tgt_id       = edge.get("target", "")
            src_out_name = edge.get("source_output_name", "output")
            tgt_fld_name = edge.get("target_field_name", "input_value")

            if src_id not in node_map or tgt_id not in node_map:
                self.mem.warnings.append(
                    f"Skipping edge {src_id}→{tgt_id}: node(s) not in compiled set"
                )
                continue

            src_node = node_map[src_id]
            tgt_node = node_map[tgt_id]
            src_comp_type = src_node["data"]["type"]
            tgt_comp_type = tgt_node["data"]["type"]

            # Find actual output from source
            src_outputs = src_node["data"]["node"].get("outputs", [])
            src_output  = next((o for o in src_outputs if o.get("name") == src_out_name), None)
            if not src_output and src_outputs:
                src_output = src_outputs[0]
            if not src_output:
                src_output = {"name": src_out_name, "types": edge.get("source_output_types", ["Message"])}

            src_out_types = src_output.get("types") or src_output.get("output_types") or ["Message"]

            # Find target field
            tgt_template = tgt_node["data"]["node"].get("template", {})
            tgt_field    = tgt_template.get(tgt_fld_name, {})
            tgt_in_types = (
                tgt_field.get("input_types")
                or edge.get("target_input_types")
                or ["Message"]
            ) if isinstance(tgt_field, dict) else ["Message"]
            tgt_fld_type = tgt_field.get("type", "str") if isinstance(tgt_field, dict) else "str"

            source_handle = {
                "dataType":    src_comp_type,
                "id":          src_id,
                "name":        src_output.get("name", src_out_name),
                "output_types": src_out_types,
            }
            target_handle = {
                "fieldName":  tgt_fld_name,
                "id":         tgt_id,
                "inputTypes": tgt_in_types,
                "type":       tgt_fld_type,
            }

            src_handle_str = _encode_handle(source_handle)
            tgt_handle_str = _encode_handle(target_handle)
            edge_id = f"reactflow__edge-{src_id}{src_handle_str}-{tgt_id}{tgt_handle_str}"

            compiled.append({
                "animated":   False,
                "className":  "",
                "data": {
                    "sourceHandle": source_handle,
                    "targetHandle": target_handle,
                },
                "id":           edge_id,
                "selected":     False,
                "source":       src_id,
                "sourceHandle": src_handle_str,
                "target":       tgt_id,
                "targetHandle": tgt_handle_str,
            })

        return compiled

    # ── Workflow assembly ─────────────────────────────────────────────────

    def _assemble_workflow(
        self, nodes: List[Dict], edges: List[Dict]
    ) -> Dict[str, Any]:
        name = str(self.mem.requirements.get("workflow_type", "Generated Workflow")).replace("_", " ").title()
        return {
            "data": {
                "edges":    edges,
                "nodes":    nodes,
                "viewport": {"x": 0, "y": 0, "zoom": 0.75},
            },
            "description":          self.mem.user_prompt[:200],
            "endpoint_name":        None,
            "id":                   str(uuid.uuid4()),
            "is_component":         False,
            "last_tested_version":  self._lf_version(),
            "name":                 name,
            "tags":                 [],
            "webhook":              False,
        }

    # ── Save ──────────────────────────────────────────────────────────────

    def _save(self, workflow: Dict[str, Any], path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(workflow, f, indent=2, ensure_ascii=False)

    # ── Live catalog helpers ──────────────────────────────────────────────

    def _fetch_raw_catalog(self) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json", "accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        for endpoint in [f"{self.base_url}/api/v1/all", f"{self.base_url}/v1/all"]:
            try:
                resp = requests.get(
                    endpoint, headers=headers,
                    timeout=self.timeout, verify=self.verify_ssl,
                )
                resp.raise_for_status()
                payload = resp.json()
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
        raise RuntimeError("Cannot fetch live catalog for compiler.")

    def _get_live_component(self, component_type: str) -> Optional[Dict[str, Any]]:
        """Look up component in raw catalog by WorkflowType key."""
        # First try fetcher (already loaded)
        fetcher = getattr(self.mem, "_fetcher", None)
        if fetcher and hasattr(fetcher, "get_component_by_type"):
            raw = fetcher.get_component_by_type(component_type)
            if raw:
                return raw

        if not self._raw_catalog:
            return None

        for category, comps in self._raw_catalog.items():
            if not isinstance(comps, dict):
                continue
            if component_type in comps:
                return comps[component_type]
            for key, comp in comps.items():
                if key.lower() == component_type.lower():
                    return comp
                if isinstance(comp, dict):
                    dn = comp.get("display_name", "")
                    if dn == component_type or dn.lower() == component_type.lower():
                        return comp

        # Search live_components
        for comp in self.mem.live_components:
            if comp.get("id") == component_type:
                return comp.get("raw", comp)

        return None

    def _icon_for(self, component_type: str) -> str:
        t = component_type.lower()
        if "chatinput" in t:      return "message-circle"
        if "chatoutput" in t:     return "message-square"
        if "prompt" in t:         return "braces"
        if "languagemodel" in t:  return "bot"
        if "agent" in t:          return "bot"
        if "memory" in t:         return "database"
        if "vectorstore" in t:    return "database"
        if "embedding" in t:      return "layers"
        if "file" in t:           return "file-text"
        if "text" in t:           return "type"
        if "openai" in t:         return "openai"
        if "anthropic" in t:      return "anthropic"
        return "component"

    def _lf_version(self) -> str:
        return "1.0.0"
