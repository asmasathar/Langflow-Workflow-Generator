from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


def _as_named_io(item: Any) -> Dict[str, Any] | None:
    if isinstance(item, dict):
        name = item.get("name") or item.get("id") or item.get("display_name")
        if not name:
            return None
        return {
            "name": str(name),
            "type": item.get("type") or item.get("field_type") or item.get("input_type") or "str",
            "required": bool(item.get("required", False)),
            "display_name": item.get("display_name") or str(name),
            "input_types": item.get("input_types") if isinstance(item.get("input_types"), list) else [],
            "output_types": item.get("output_types") if isinstance(item.get("output_types"), list) else [],
            "value": item.get("value"),
            "default": item.get("default"),
            "placeholder": item.get("placeholder"),
            "info": item.get("info"),
            "advanced": bool(item.get("advanced", False)),
            "tool_mode": bool(item.get("tool_mode", False)),
            "dynamic": bool(item.get("dynamic", False)),
            "options": item.get("options") if isinstance(item.get("options"), list) else [],
            "options_metadata": item.get("options_metadata") if isinstance(item.get("options_metadata"), list) else [],
            "combobox": bool(item.get("combobox", False)),
            "toggle": bool(item.get("toggle", False)),
            "multiline": bool(item.get("multiline", False)),
            "list": bool(item.get("list", False)),
            "list_add_label": item.get("list_add_label"),
            "fileTypes": item.get("fileTypes") if isinstance(item.get("fileTypes"), list) else [],
            "file_path": item.get("file_path"),
            "password": bool(item.get("password", False)),
            "trace_as_input": bool(item.get("trace_as_input", False)),
            "trace_as_metadata": bool(item.get("trace_as_metadata", False)),
            "show": item.get("show"),
            "load_from_db": bool(item.get("load_from_db", False)),
            "copy_field": bool(item.get("copy_field", False)),
            "title_case": bool(item.get("title_case", False)),
        }
    if isinstance(item, str) and item.strip():
        return {"name": item.strip(), "type": "str", "required": False, "display_name": item.strip()}
    return None


def _normalize_inputs(component: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_inputs = component.get("inputs")
    inputs: List[Dict[str, Any]] = []

    if isinstance(raw_inputs, list):
        for item in raw_inputs:
            normalized = _as_named_io(item)
            if normalized:
                inputs.append(normalized)

    if not inputs:
        field_config = component.get("field_config")
        if isinstance(field_config, dict):
            for field_name, field_meta in field_config.items():
                if isinstance(field_meta, dict):
                    inputs.append(
                        {
                            "name": str(field_name),
                            "type": field_meta.get("type") or field_meta.get("field_type") or "str",
                            "required": bool(field_meta.get("required", False)),
                            "display_name": field_meta.get("display_name") or str(field_name),
                        }
                    )
                elif isinstance(field_name, str):
                    inputs.append(
                        {
                            "name": field_name,
                            "type": "str",
                            "required": False,
                            "display_name": field_name,
                        }
                    )

    return inputs


def _normalize_outputs(component: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    raw_outputs = component.get("outputs")

    if isinstance(raw_outputs, list):
        for item in raw_outputs:
            normalized = _as_named_io(item)
            if normalized:
                outputs.append(normalized)

    output_type_map = component.get("output_type_map")
    if isinstance(output_type_map, dict):
        for out_name, out_type in output_type_map.items():
            if not any(existing.get("name") == out_name for existing in outputs):
                outputs.append(
                    {
                        "name": str(out_name),
                        "type": out_type if isinstance(out_type, str) else "str",
                        "required": False,
                        "display_name": str(out_name),
                        "input_types": [],
                        "output_types": [],
                        "value": None,
                        "default": None,
                        "placeholder": None,
                        "info": None,
                        "advanced": False,
                        "tool_mode": False,
                        "dynamic": False,
                        "options": [],
                        "options_metadata": [],
                        "combobox": False,
                        "toggle": False,
                        "multiline": False,
                        "list": False,
                        "list_add_label": None,
                        "fileTypes": [],
                        "file_path": None,
                        "password": False,
                        "trace_as_input": False,
                        "trace_as_metadata": False,
                        "show": None,
                        "load_from_db": False,
                        "copy_field": False,
                        "title_case": False,
                    }
                )

    if not outputs and isinstance(component.get("output_types"), list):
        for out_name in component.get("output_types", []):
            if isinstance(out_name, str):
                outputs.append(
                    {
                        "name": out_name,
                        "type": "str",
                        "required": False,
                        "display_name": out_name,
                        "input_types": [],
                        "output_types": [],
                        "value": None,
                        "default": None,
                        "placeholder": None,
                        "info": None,
                        "advanced": False,
                        "tool_mode": False,
                        "dynamic": False,
                        "options": [],
                        "options_metadata": [],
                        "combobox": False,
                        "toggle": False,
                        "multiline": False,
                        "list": False,
                        "list_add_label": None,
                        "fileTypes": [],
                        "file_path": None,
                        "password": False,
                        "trace_as_input": False,
                        "trace_as_metadata": False,
                        "show": None,
                        "load_from_db": False,
                        "copy_field": False,
                        "title_case": False,
                    }
                )

    return outputs


def _normalize_template_fields(component: Dict[str, Any]) -> List[Dict[str, Any]]:
    template = component.get("template")
    fields: List[Dict[str, Any]] = []
    if not isinstance(template, dict):
        return fields

    for field_name, field_meta in template.items():
        if field_name == "code":
            continue
        if not isinstance(field_meta, dict):
            continue
        name = str(field_meta.get("name") or field_name).strip()
        if not name:
            continue
        input_types = field_meta.get("input_types")
        if not isinstance(input_types, list):
            input_types = []
        fields.append(
            {
                "name": name,
                "type": field_meta.get("type") or "str",
                "required": bool(field_meta.get("required", False)),
                "display_name": field_meta.get("display_name") or name,
                "default": field_meta.get("value"),
                "input_types": input_types,
                "advanced": bool(field_meta.get("advanced", False)),
                        "tool_mode": bool(field_meta.get("tool_mode", False)),
                        "placeholder": field_meta.get("placeholder"),
                        "info": field_meta.get("info"),
                        "dynamic": bool(field_meta.get("dynamic", False)),
                        "options": field_meta.get("options") if isinstance(field_meta.get("options"), list) else [],
                        "options_metadata": field_meta.get("options_metadata") if isinstance(field_meta.get("options_metadata"), list) else [],
                        "combobox": bool(field_meta.get("combobox", False)),
                        "toggle": bool(field_meta.get("toggle", False)),
                        "multiline": bool(field_meta.get("multiline", False)),
                        "list": bool(field_meta.get("list", False)),
                        "list_add_label": field_meta.get("list_add_label"),
                        "fileTypes": field_meta.get("fileTypes") if isinstance(field_meta.get("fileTypes"), list) else [],
                        "file_path": field_meta.get("file_path"),
                        "password": bool(field_meta.get("password", False)),
                        "trace_as_input": bool(field_meta.get("trace_as_input", False)),
                        "trace_as_metadata": bool(field_meta.get("trace_as_metadata", False)),
                        "show": field_meta.get("show"),
                        "load_from_db": bool(field_meta.get("load_from_db", False)),
                        "copy_field": bool(field_meta.get("copy_field", False)),
                        "title_case": bool(field_meta.get("title_case", False)),
            }
        )

    return fields


def _extract_component_code(component: Dict[str, Any]) -> str:
    template = component.get("template")
    if isinstance(template, dict):
        code_field = template.get("code")
        if isinstance(code_field, dict):
            value = code_field.get("value")
            if isinstance(value, str):
                return value
        if isinstance(code_field, str):
            return code_field
    code = component.get("code")
    if isinstance(code, str):
        return code
    source_code = component.get("source_code")
    if isinstance(source_code, str):
        return source_code
    return ""


def _is_component_candidate(node: Dict[str, Any]) -> bool:
    keys = set(node.keys())
    signal_keys = {
        "inputs",
        "outputs",
        "field_config",
        "output_type_map",
        "output_types",
        "display_name",
        "base_classes",
        "template",
    }
    return bool(keys & signal_keys)


def _collect_component_dicts(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("components"), list):
        return [item for item in payload["components"] if isinstance(item, dict)]

    if isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]

    candidates: List[Dict[str, Any]] = []
    stack: List[Any] = [payload]
    seen_ids: Set[int] = set()

    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)

        if isinstance(current, dict):
            if _is_component_candidate(current):
                candidates.append(current)
            for value in current.values():
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            for value in current:
                if isinstance(value, (dict, list)):
                    stack.append(value)

    return candidates


def normalize_langflow_components(payload: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for raw in _collect_component_dicts(payload):
        component_id = raw.get("id") or raw.get("name") or raw.get("display_name")
        if not component_id:
            continue

        cid = str(component_id)
        if cid in seen:
            continue
        seen.add(cid)

        inputs = _normalize_inputs(raw)
        outputs = _normalize_outputs(raw)
        template_fields = _normalize_template_fields(raw)
        parameters = raw.get("parameters") if isinstance(raw.get("parameters"), list) else []
        merged_parameters = template_fields or parameters or inputs

        normalized.append(
            {
                "id": cid,
                "type": raw.get("type") or raw.get("category") or raw.get("component_type") or "generic",
                "role": raw.get("role") or "",
                "display_name": raw.get("display_name") or cid,
                "description": raw.get("description") or raw.get("documentation") or "",
                "inputs": inputs,
                "outputs": outputs,
                "parameters": merged_parameters,
                "template": raw.get("template") if isinstance(raw.get("template"), dict) else {},
                "field_order": raw.get("field_order") if isinstance(raw.get("field_order"), list) else [],
                "metadata": raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {},
                "code": _extract_component_code(raw),
                "base_classes": raw.get("base_classes") if isinstance(raw.get("base_classes"), list) else [],
                "output_type_map": raw.get("output_type_map") if isinstance(raw.get("output_type_map"), dict) else {},
                "raw": raw,
            }
        )

    return normalized
