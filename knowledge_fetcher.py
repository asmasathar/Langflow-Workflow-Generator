from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from langflow_catalog import normalize_langflow_components
from shared_memory import SharedMemory


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    h = {"Content-Type": "application/json", "accept": "application/json"}
    if api_key:
        h["x-api-key"] = api_key
    return h


def _get(url: str, api_key: Optional[str], timeout: int = 30, verify: bool = True) -> Any:
    resp = requests.get(url, headers=_headers(api_key), timeout=timeout, verify=verify)
    resp.raise_for_status()
    return resp.json()


class KnowledgeFetcher:
    def __init__(self):
        self.base_url = os.getenv("LANGFLOW_BASE_URL", "http://localhost:7860").rstrip("/")
        self.api_key  = os.getenv("LANGFLOW_API_KEY")
        self.timeout  = int(os.getenv("LANGFLOW_TIMEOUT", "30"))
        self.verify_ssl = os.getenv("LANGFLOW_VERIFY_SSL", "true").lower() not in {"0","false","no"}
        self.mcp_project_id = os.getenv("LANGFLOW_MCP_PROJECT_ID")
        self._dataset_path  = Path(__file__).parent / "workflow_dataset.json"
        self._core_only = os.getenv("WF_CORE_KNOWLEDGE_ONLY", "0").strip().lower() in {"1","true","yes","on"}
        self._core_snapshot_path = Path(__file__).parent / "core_knowledge.md"

        # Populated by fetch_all_components
        self._raw_all_payload: Dict[str, Any] = {}
        # workflow_type -> raw component dict  (e.g. "ChatInput" -> {...})
        self._type_to_raw:     Dict[str, Any] = {}
        # normalized(workflow_type) -> workflow_type  for fuzzy lookup
        self._norm_to_type:    Dict[str, str] = {}

    # ── Catalog ───────────────────────────────────────────────────────────

    def fetch_all_components(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        for endpoint in [f"{self.base_url}/api/v1/all", f"{self.base_url}/v1/all"]:
            try:
                raw = _get(endpoint, self.api_key, self.timeout, self.verify_ssl)
                if isinstance(raw, dict):
                    self._raw_all_payload = raw
                    self._build_type_index(raw)
                    normalized = normalize_langflow_components(raw)
                    return normalized, raw
            except Exception:
                continue
        raise RuntimeError(
            f"Cannot reach Langflow component catalog. "
            f"Is LANGFLOW_BASE_URL={self.base_url} correct?"
        )

    def _build_type_index(self, raw: Dict[str, Any]) -> None:
        """
        Walk the raw /api/v1/all payload and populate:
          _type_to_raw   : { "ChatInput": <component_dict>, … }
          _norm_to_type  : { "chatinput": "ChatInput", … }

        The dict KEY at the category level is the canonical workflow type.
        """
        import re
        self._type_to_raw  = {}
        self._norm_to_type = {}

        for category, comps in raw.items():
            if not isinstance(comps, dict):
                continue
            for wf_type, comp_dict in comps.items():
                if not isinstance(comp_dict, dict):
                    continue
                self._type_to_raw[wf_type] = comp_dict
                norm = re.sub(r"[^a-z0-9]", "", wf_type.lower())
                self._norm_to_type[norm] = wf_type

    def get_component_by_type(self, workflow_type: str) -> Optional[Dict[str, Any]]:
        """
        Lookup a raw component dict by its workflow type string.
        Tries exact match first, then normalised match.
        Returns None if not found.
        """
        import re

        # 1. Exact
        if workflow_type in self._type_to_raw:
            return self._type_to_raw[workflow_type]

        # 2. Normalised
        norm = re.sub(r"[^a-z0-9]", "", workflow_type.lower())
        exact_type = self._norm_to_type.get(norm)
        if exact_type:
            return self._type_to_raw.get(exact_type)

        # 3. Substring in normalised keys
        for k, t in self._norm_to_type.items():
            if norm in k or k in norm:
                return self._type_to_raw.get(t)

        return None

    # Keep old name as alias for backwards compat
    def get_component_by_id(
        self, component_id: str, raw_all_payload: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        if raw_all_payload and raw_all_payload is not self._raw_all_payload:
            # caller passed a different payload – rebuild index temporarily
            tmp = KnowledgeFetcher.__new__(KnowledgeFetcher)
            tmp._type_to_raw  = {}
            tmp._norm_to_type = {}
            tmp._build_type_index(raw_all_payload)
            return tmp.get_component_by_type(component_id)
        return self.get_component_by_type(component_id)

    def all_workflow_types(self) -> List[str]:
        """Return the list of ALL canonical workflow type strings."""
        return list(self._type_to_raw.keys())

    # ── Other endpoints ───────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/health_check",
                headers=_headers(self.api_key),
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            if resp.status_code >= 400:
                snippet = (resp.text or "").strip().replace("\n", " ")[:160]
                return {
                    "status": "error",
                    "error": f"HTTP {resp.status_code} from /health_check: {snippet or 'no body'}",
                }

            body = (resp.text or "").strip()
            if not body:
                return {"status": "warning", "error": "Empty response body from /health_check"}

            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    return payload
                return {"status": "ok", "detail": str(payload)[:160]}
            except ValueError:
                ctype = resp.headers.get("content-type", "unknown")
                snippet = body.replace("\n", " ")[:160]
                return {
                    "status": "warning",
                    "error": f"Non-JSON health response (content-type={ctype}): {snippet}",
                }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def fetch_config(self) -> Dict[str, Any]:
        try:
            return _get(f"{self.base_url}/api/v1/config",
                        self.api_key, self.timeout, self.verify_ssl)
        except Exception as exc:
            return {"error": str(exc)}

    def fetch_example_flows(self) -> List[Dict[str, Any]]:
        try:
            result = _get(
                f"{self.base_url}/api/v1/flows/"
                "?remove_example_flows=false&get_all=true&page=1&size=100",
                self.api_key, self.timeout, self.verify_ssl,
            )
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "items" in result:
                return result["items"]
        except Exception:
            pass
        return []

    def fetch_starter_projects(self) -> List[Dict[str, Any]]:
        try:
            result = _get(f"{self.base_url}/api/v1/starter-projects/",
                          self.api_key, self.timeout, self.verify_ssl)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "flows" in result:
                return result["flows"]
        except Exception:
            pass
        return []

    def fetch_mcp_tools(self) -> List[Dict[str, Any]]:
        if not self.mcp_project_id:
            return []
        try:
            result = _get(
                f"{self.base_url}/api/v1/mcp/project/{self.mcp_project_id}",
                self.api_key, self.timeout, self.verify_ssl,
            )
            tools = result.get("tools", []) if isinstance(result, dict) else []
            return tools if isinstance(tools, list) else []
        except Exception:
            return []

    def load_workflow_dataset(self) -> List[Dict[str, Any]]:
        if not self._dataset_path.exists():
            return []
        with open(self._dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    def load_all_into_memory(self, mem: SharedMemory) -> None:
        print("\n[Knowledge] Checking Langflow server health...", end=" ", flush=True)
        health = self.health_check()
        if "error" in health:
            print(f"WARNING: {health['error']}")
        else:
            print("OK")

        print("[Knowledge] Fetching component catalog from /api/v1/all...", end=" ", flush=True)
        components, _ = self.fetch_all_components()
        mem.live_components = components
        print(f"{len(components)} components, {len(self._type_to_raw)} workflow types")

        print("[Knowledge] Fetching example flows...", end=" ", flush=True)
        flows = self.fetch_example_flows()
        mem.example_flows = flows
        print(f"{len(flows)} flows")

        if not self._core_only:
            print("[Knowledge] Fetching starter projects...", end=" ", flush=True)
            starters = self.fetch_starter_projects()
            mem.starter_projects = starters
            print(f"{len(starters)} starter projects")

            print("[Knowledge] Fetching MCP tools...", end=" ", flush=True)
            tools = self.fetch_mcp_tools()
            mem.mcp_tools = tools
            print(f"{len(tools)} tools")

            print("[Knowledge] Loading workflow dataset...", end=" ", flush=True)
            dataset = self.load_workflow_dataset()
            mem.example_flows = (mem.example_flows or []) + dataset
            print(f"{len(dataset)} curated examples")
        else:
            mem.starter_projects = []
            mem.mcp_tools = []
            print("[Knowledge] Core-only mode: skipped starter projects, MCP tools, and local dataset.")

        self._export_core_snapshot(mem)

        print("[Knowledge] All knowledge loaded.\n")

    def _export_core_snapshot(self, mem: SharedMemory) -> None:
        try:
            comp_section = self.build_component_catalog_summary(mem.live_components)
            flow_section = self.build_workflow_examples_summary(mem.example_flows)
            content = "\n\n".join([
                "# Core Langflow Knowledge (live catalog + example flows)",
                "## Components",
                comp_section,
                "\n## Example Flows",
                flow_section,
            ])
            self._core_snapshot_path.write_text(content, encoding="utf-8")
            print(f"[Knowledge] Wrote core knowledge snapshot to {self._core_snapshot_path}")
        except Exception as exc:
            print(f"[Knowledge] Warning: could not write core knowledge snapshot ({exc})")

    # ── Utility builders ──────────────────────────────────────────────────

    def build_component_catalog_summary(self, components: List[Dict[str, Any]]) -> str:
        lines = ["# Available Langflow Components\n"]
        seen: set = set()
        for comp in components:
            ctype = comp.get("id") or ""
            if ctype in seen:
                continue
            seen.add(ctype)
            desc = (comp.get("description") or "").replace("\n", " ").strip()[:100]
            lines.append(f"- {ctype}: {desc}")
        return "\n".join(lines)

    def build_workflow_examples_summary(self, examples: List[Dict[str, Any]]) -> str:
        from shared_memory import SharedMemory
        lines = ["# Workflow Examples\n"]
        for ex in examples:
            name = ex.get("name") or ex.get("id", "")
            desc = (ex.get("description") or "")[:100]
            nodes, edges = SharedMemory._unwrap_flow_data(ex)
            node_types = list({n.get("data", {}).get("type", "") for n in nodes if isinstance(n, dict)})
            lines.append(f"- {name}: {desc}")
            if node_types:
                lines.append(f"  Uses: {', '.join(node_types[:10])}")
            if edges:
                lines.append(f"  Edges: {len(edges)}")
        return "\n".join(lines)