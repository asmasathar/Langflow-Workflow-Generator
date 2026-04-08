from __future__ import annotations

import json
import re
import sys
import os
import glob
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


# -- Flow unwrapping -----------------------------------------------------------

def unwrap(flow_obj: Any) -> Tuple[List[Dict], List[Dict]]:
    if not isinstance(flow_obj, dict):
        return [], []
    candidate = flow_obj.get("workflow") or flow_obj.get("data") or flow_obj
    if not isinstance(candidate, dict):
        return [], []
    nodes = candidate.get("nodes", [])
    edges = candidate.get("edges", [])
    if not nodes and not edges:
        inner = candidate.get("data", {})
        if isinstance(inner, dict):
            nodes = inner.get("nodes", [])
            edges = inner.get("edges", [])
    return (nodes or []), (edges or [])


def node_type(n: Dict) -> str:
    return n.get("data", {}).get("type", "")


def node_id(n: Dict) -> str:
    return n.get("id", "") or n.get("data", {}).get("id", "")


def get_template(n: Dict) -> Dict:
    return n.get("data", {}).get("node", {}).get("template", {})


def get_field_value(n: Dict, fname: str) -> Any:
    tmpl = get_template(n)
    fdef = tmpl.get(fname, {})
    if isinstance(fdef, dict):
        return fdef.get("value")
    return None


def get_outputs(n: Dict) -> List[Dict]:
    return n.get("data", {}).get("node", {}).get("outputs", [])


def get_output_names(n: Dict) -> List[str]:
    return [o.get("name", "") for o in get_outputs(n)]


# -- Role classification -------------------------------------------------------
# 13 functional roles. Each node maps to exactly one primary role (or None).

def get_role(n: Dict) -> Optional[str]:
    """
    Map a node to its single primary functional role, or None for unknown/note nodes.

    Roles (in evaluation priority order):
      source       -- ChatInput, TextInput
      sink         -- ChatOutput, SaveToFile, TextOutput, GoogleDriveUpload
      ai_processor -- Any LLM model (OpenAI/Anthropic/LanguageModel/etc.) and Agents
      memory       -- Memory, MemoryComponent, ZepMemory, etc.
      retrieval    -- VectorStore, AstraDB, Chroma, FAISS, Pinecone, Qdrant, etc.
      embeddings   -- OpenAIEmbeddings, HuggingFaceEmbeddings, etc.
      file_data    -- File, DirectoryLoader, PDFLoader (produce data from disk)
      web_data     -- URLComponent, APIRequest, Tavily, AgentQL, Firecrawl
      tool         -- MCPTools and other tool-providing components
      prompt       -- Prompt template node
      splitter     -- SplitText, TextSplitter, CharacterTextSplitter
      parser       -- ParseData, ParserComponent, StructuredOutput, DataToText
      knowledge    -- KnowledgeIngestion, KnowledgeRetrieval
    """
    t = node_type(n)
    tl = t.lower()

    if not tl or tl in ("note", "notenode"):
        return None

    outputs = get_outputs(n)
    out_types: Set[str] = set()
    for o in outputs:
        for ot in (o.get("types") or o.get("output_types") or []):
            out_types.add(ot)

    # source -- user input entry points
    if "chatinput" in tl or ("textinput" in tl and "output" not in tl):
        return "source"

    # sink -- output delivery points
    if ("chatoutput" in tl or "savetofile" in tl or "textoutput" in tl
            or "googledriveupload" in tl):
        return "sink"

    # ai_processor -- LLM models and agents (agents ARE LLM-powered)
    if ("agent" in tl
            or "languagemodel" in tl
            or "openaimodel" in tl
            or "anthropicmodel" in tl
            or "openrouter" in tl
            or "groqmodel" in tl or tl == "groq"
            or "mistralmodel" in tl or "mistral" in tl
            or "geminimodel" in tl or "gemini" in tl
            or "ollamamodel" in tl or "ollama" in tl
            or "LanguageModel" in out_types):
        return "ai_processor"

    # memory -- conversation history storage
    if "memory" in tl and "embedding" not in tl:
        return "memory"

    # retrieval -- vector stores for semantic search / RAG
    if ("vectorstore" in tl or "astradb" in tl or "chroma" in tl
            or "faiss" in tl or "opensearch" in tl or "pinecone" in tl
            or "qdrant" in tl or "weaviate" in tl or "milvus" in tl):
        return "retrieval"

    # embeddings -- text-to-vector models
    if "embedding" in tl or "Embeddings" in out_types:
        return "embeddings"

    # file_data -- file/document loaders (no upstream dependencies)
    if (tl == "file" or "directoryloader" in tl or "pdfloader" in tl
            or ("loader" in tl and any(k in tl for k in ("file", "pdf", "directory", "csv", "doc")))):
        return "file_data"

    # web_data -- fetches from web/external APIs
    if ("urlcomponent" in tl or tl == "url" or "apirequest" in tl
            or "tavily" in tl or "agentql" in tl or "firecrawl" in tl
            or "browserless" in tl or "searxng" in tl or "serpapi" in tl):
        return "web_data"

    # tool -- provides tools/capabilities to agents
    if "mcptools" in tl or "Tool" in out_types:
        return "tool"

    # prompt -- prompt template construction
    if "prompt" in tl:
        return "prompt"

    # splitter -- text chunking
    if "splittext" in tl or "textsplitter" in tl or "charactertextsplitter" in tl:
        return "splitter"

    # parser -- data transformation/parsing
    if ("parsedata" in tl or ("parse" in tl and "data" in tl)
            or "parsercomponent" in tl
            or ("structured" in tl and "output" in tl)
            or "retrievalqa" in tl or "datatotext" in tl):
        return "parser"

    # knowledge
    if "knowledge" in tl:
        return "knowledge"

    return None


def llm_has_system_message_field(n: Dict) -> bool:
    """Return True if the node has a system_message or system_prompt field in its template."""
    tmpl = get_template(n)
    return "system_message" in tmpl or "system_prompt" in tmpl


def get_workflow_roles(nodes: List[Dict], edges: List[Dict]) -> Set[str]:
    """
    Return the set of functional roles present in a workflow.

    Special rule for 'prompt': satisfied by EITHER:
      (a) a Prompt template node connected via edge, OR
      (b) an ai_processor node that has system_message/system_prompt in its template
          (regardless of whether the field is filled — the capability is built-in).

    Note: whether the system_message is actually filled is evaluated separately
    as a runtime health check (empty field → small penalty, not a role miss).
    """
    roles: Set[str] = set()
    for n in nodes:
        r = get_role(n)
        if r:
            roles.add(r)

    # Prompt via system_message field (even empty) counts as role coverage
    if "prompt" not in roles:
        for n in nodes:
            if get_role(n) == "ai_processor":
                if llm_has_system_message_field(n):
                    roles.add("prompt")
                    break

    return roles


# -- Path reachability (BFS) ---------------------------------------------------

def path_exists(from_ids: Set[str], to_ids: Set[str], edges: List[Dict]) -> bool:
    """Can any node in from_ids reach any node in to_ids following directed edges?"""
    if not from_ids or not to_ids:
        return False
    fwd: Dict[str, List[str]] = {}
    for e in edges:
        src = e.get("source", "")
        tgt = e.get("target", "")
        if src and tgt:
            fwd.setdefault(src, []).append(tgt)

    visited: Set[str] = set(from_ids)
    queue: deque = deque(from_ids)
    while queue:
        curr = queue.popleft()
        if curr in to_ids:
            return True
        for nxt in fwd.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return False


# -- Logical equivalence helpers -----------------------------------------------

def system_message_delivered(n: Dict, edges: List[Dict]) -> bool:
    """Check if system_message/system_prompt is delivered to node n (field or edge)."""
    nid = node_id(n)
    for e in edges:
        th = e.get("data", {}).get("targetHandle", {})
        if isinstance(th, dict):
            if e.get("target") == nid and th.get("fieldName") in ("system_message", "system_prompt"):
                return True
    val = get_field_value(n, "system_message") or get_field_value(n, "system_prompt")
    return bool(val and str(val).strip())


def types_compatible(src_types: List[str], tgt_types: List[str]) -> bool:
    if not src_types or not tgt_types:
        return True
    s = {t.lower() for t in src_types}
    t = {t.lower() for t in tgt_types}
    if s & t:
        return True
    if {"message", "text", "str"} & s and {"message", "text", "str"} & t:
        return True
    if "message" in s and "data" in t:
        return True
    return False


# -- Required field analysis (edge-aware) -------------------------------------

# Fields that look required but are always satisfied at runtime
_SKIP_REQUIRED_FIELDS = {"input_value", "code", "_type"}
_IGNORE_COMPONENT_TYPES = {"note", "noteNode"}


def get_real_missing_required_fields(
    nodes: List[Dict], edges: List[Dict]
) -> List[Tuple[str, str, str]]:
    """
    Returns (node_id, component_type, field_name) for required fields that have
    no value AND no incoming edge feeding that field name.
    """
    fields_with_edge: Set[Tuple[str, str]] = set()
    for e in edges:
        th = e.get("data", {}).get("targetHandle", {})
        tgt = e.get("target", "")
        fn = th.get("fieldName", "") if isinstance(th, dict) else ""
        if tgt and fn:
            fields_with_edge.add((tgt, fn))

    missing = []
    for n in nodes:
        t = node_type(n)
        if t in _IGNORE_COMPONENT_TYPES:
            continue
        nid = node_id(n)
        tmpl = get_template(n)
        for fname, fdef in tmpl.items():
            if fname in _SKIP_REQUIRED_FIELDS:
                continue
            if not isinstance(fdef, dict):
                continue
            if not fdef.get("required", False):
                continue
            val = fdef.get("value")
            is_empty = val is None or val == "" or val == [] or val == {}
            if is_empty and (nid, fname) not in fields_with_edge:
                missing.append((nid, t, fname))
    return missing


# -- Handle validity check -----------------------------------------------------

def get_invalid_handles(
    nodes: List[Dict], edges: List[Dict]
) -> List[Tuple[str, str, str]]:
    """
    Returns (src_component_type, handle_name, tgt_component_type) for edges whose
    sourceHandle.name does not exist in the source node's declared outputs list.
    """
    nid_to_node = {node_id(n): n for n in nodes}
    nid_to_outputs = {node_id(n): get_output_names(n) for n in nodes}

    invalid = []
    for e in edges:
        sh = e.get("data", {}).get("sourceHandle", {})
        src = e.get("source", "")
        tgt = e.get("target", "")
        if not isinstance(sh, dict):
            continue
        handle_name = sh.get("name", "")
        valid_outputs = nid_to_outputs.get(src, [])
        if handle_name and valid_outputs and handle_name not in valid_outputs:
            src_node = nid_to_node.get(src, {})
            tgt_node = nid_to_node.get(tgt, {})
            invalid.append((node_type(src_node), handle_name, node_type(tgt_node)))
    return invalid


# -- Result dataclasses --------------------------------------------------------

@dataclass
class NodeCoverage:
    score: float             # 0.0 - 100.0
    ref_roles: List[str]     # functional roles required by the reference
    covered_roles: List[str] # ref roles present in the generated workflow
    missing_roles: List[str] # ref roles absent from the generated workflow
    extra_roles: List[str]   # generated roles not in the reference (bonus)
    notes: List[str]


@dataclass
class RuntimeHealth:
    total_penalty: float         # total points deducted (capped at 60)
    breakdown: Dict[str, float]  # category -> penalty points
    notes: List[str]


@dataclass
class EvaluationResult:
    workflow_name: str
    node_coverage: NodeCoverage
    runtime_health: RuntimeHealth
    final_score: float   # max(0, node_coverage.score - runtime_health.total_penalty)
    verdict: str
    detail_lines: List[str]


# -- Main evaluator ------------------------------------------------------------

class WorkflowEvaluator:
    """
    Two-component workflow evaluator.

    Final Score = max(0, Node_Coverage_Score - Runtime_Health_Penalty)
    """

    def evaluate(
        self,
        ref_flow: Any,
        gen_flow: Any,
        name: str = "workflow",
    ) -> EvaluationResult:

        ref_nodes, ref_edges = unwrap(ref_flow)
        gen_nodes, gen_edges = unwrap(gen_flow)

        # ── Component 1: Node Coverage ────────────────────────────────────────
        ref_roles = get_workflow_roles(ref_nodes, ref_edges)
        gen_roles = get_workflow_roles(gen_nodes, gen_edges)
        ref_roles_list = sorted(ref_roles)

        if not ref_roles_list:
            cov_score = 100.0
        else:
            covered = ref_roles & gen_roles
            cov_score = len(covered) / len(ref_roles) * 100.0

        covered_roles = sorted(ref_roles & gen_roles)
        missing_roles = sorted(ref_roles - gen_roles)
        extra_roles   = sorted(gen_roles - ref_roles)

        cov_notes: List[str] = []
        for r in covered_roles:
            cov_notes.append(f"Role '{r}': present in generated [OK]")
        for r in missing_roles:
            cov_notes.append(f"Role '{r}': required by reference but MISSING [MISS]")
        if extra_roles:
            cov_notes.append(
                f"Extra roles in generated (bonus, not penalized): {', '.join(extra_roles)}"
            )

        coverage = NodeCoverage(
            score=cov_score,
            ref_roles=ref_roles_list,
            covered_roles=covered_roles,
            missing_roles=missing_roles,
            extra_roles=extra_roles,
            notes=cov_notes,
        )

        # ── Component 2: Runtime Health Penalty ───────────────────────────────
        breakdown: Dict[str, float] = {}
        rh_notes: List[str] = []

        # 1. Orphan non-source nodes (5 pts each, cap 25)
        source_roles = {"source", "file_data", "web_data"}
        gen_targets = {e.get("target") for e in gen_edges}
        orphans: List[str] = []
        for n in gen_nodes:
            r = get_role(n)
            if r in source_roles:
                continue
            if node_type(n) in ("note", "noteNode"):
                continue
            if node_id(n) not in gen_targets:
                orphans.append(node_type(n))
        if orphans:
            p = min(25.0, len(orphans) * 5.0)
            breakdown["orphan_nodes"] = p
            rh_notes.append(
                f"Orphan node(s) ({len(orphans)}): {', '.join(orphans[:5])} -> -{p:.0f} pts [MISS]"
            )
        else:
            rh_notes.append("No orphan nodes [OK]")

        # 2. Invalid handle names (8 pts each, cap 24)
        invalid_handles = get_invalid_handles(gen_nodes, gen_edges)
        if invalid_handles:
            p = min(24.0, len(invalid_handles) * 8.0)
            breakdown["invalid_handles"] = p
            for src_t, h, tgt_t in invalid_handles[:3]:
                rh_notes.append(
                    f"Invalid handle: {src_t}.{h} -> {tgt_t} [MISS]"
                )
            if len(invalid_handles) > 3:
                rh_notes.append(f"  ...and {len(invalid_handles) - 3} more invalid handle(s)")
            rh_notes.append(f"Invalid handles: {len(invalid_handles)} total -> -{p:.0f} pts")
        else:
            rh_notes.append("All edge handles valid [OK]")

        # 3. Hard type errors (8 pts each, cap 16)
        nid_to_node    = {node_id(n): n for n in gen_nodes}
        nid_to_outputs = {node_id(n): get_output_names(n) for n in gen_nodes}
        hard_errs: List[str] = []
        for e in gen_edges:
            sh  = e.get("data", {}).get("sourceHandle", {})
            th  = e.get("data", {}).get("targetHandle", {})
            src = e.get("source", "")
            tgt = e.get("target", "")
            if not isinstance(sh, dict) or not isinstance(th, dict):
                continue
            handle_name   = sh.get("name", "")
            valid_outputs = nid_to_outputs.get(src, [])
            if handle_name and valid_outputs and handle_name not in valid_outputs:
                continue  # already counted as invalid handle, skip type check
            src_ot = sh.get("output_types") or []
            tgt_it = th.get("inputTypes") or []
            if not src_ot or not tgt_it:
                continue
            if not types_compatible(src_ot, tgt_it):
                safe_cross = "Message" in src_ot and tgt_it == ["Data"]
                if (not safe_cross
                        and "LanguageModel" in src_ot
                        and "LanguageModel" not in tgt_it):
                    src_t = node_type(nid_to_node.get(src, {}))
                    tgt_t = node_type(nid_to_node.get(tgt, {}))
                    hard_errs.append(f"{src_t}({src_ot}) -> {tgt_t}({tgt_it})")
        if hard_errs:
            p = min(16.0, len(hard_errs) * 8.0)
            breakdown["hard_type_errors"] = p
            for he in hard_errs[:2]:
                rh_notes.append(f"Hard type error: {he} [MISS]")
            rh_notes.append(f"Hard type errors: {len(hard_errs)} -> -{p:.0f} pts")
        else:
            rh_notes.append("No hard type errors [OK]")

        # 4. Source -> AI processor path (15 pts if broken when ref has it)
        ref_src_ids  = {node_id(n) for n in ref_nodes if get_role(n) == "source"}
        ref_ai_ids   = {node_id(n) for n in ref_nodes if get_role(n) == "ai_processor"}
        ref_sink_ids = {node_id(n) for n in ref_nodes if get_role(n) == "sink"}
        gen_src_ids  = {node_id(n) for n in gen_nodes if get_role(n) == "source"}
        gen_ai_ids   = {node_id(n) for n in gen_nodes if get_role(n) == "ai_processor"}
        gen_sink_ids = {node_id(n) for n in gen_nodes if get_role(n) == "sink"}

        if path_exists(ref_src_ids, ref_ai_ids, ref_edges):
            if not path_exists(gen_src_ids, gen_ai_ids, gen_edges):
                breakdown["source_ai_path"] = 15.0
                rh_notes.append("Source does not reach AI processor -> -15 pts [MISS]")
            else:
                rh_notes.append("Source -> AI processor path: connected [OK]")

        # 5. AI processor -> sink path (12 pts if broken when ref has it)
        if path_exists(ref_ai_ids, ref_sink_ids, ref_edges):
            if not path_exists(gen_ai_ids, gen_sink_ids, gen_edges):
                breakdown["ai_sink_path"] = 12.0
                rh_notes.append("AI processor does not reach sink -> -12 pts [MISS]")
            else:
                rh_notes.append("AI processor -> Sink path: connected [OK]")

        # 6. Empty required fields (8 pts per component, cap 24)
        missing_fields = get_real_missing_required_fields(gen_nodes, gen_edges)
        if missing_fields:
            by_comp: Dict[str, List[str]] = {}
            for _nid, ctype, fname in missing_fields:
                by_comp.setdefault(ctype, []).append(fname)
            for ctype, fnames in sorted(by_comp.items()):
                rh_notes.append(
                    f"Empty required field(s): {ctype}.{{{', '.join(fnames)}}} [MISS]"
                )
            p = min(24.0, len(by_comp) * 8.0)
            breakdown["missing_required_fields"] = p
            rh_notes.append(f"Empty required fields: {len(by_comp)} component(s) -> -{p:.0f} pts")
        else:
            rh_notes.append("All required fields satisfied [OK]")

        # 7. System message field -- NOT penalized.
        # An empty system_message is acceptable: natural-language descriptions rarely
        # specify the LLM system prompt, so penalising an empty field would unfairly
        # punish the system for a configuration value the user never provided.
        # The 'prompt' role coverage check (above) still uses the system_message field
        # as evidence that prompt-delivery capability is present.
        gen_ai_nodes = [n for n in gen_nodes if get_role(n) == "ai_processor"]
        if gen_ai_nodes:
            rh_notes.append("System message check skipped (not penalized -- user-supplied value) [OK]")

        # 8. Memory present but disconnected -- NOT separately penalized.
        # A disconnected Memory node is already caught by the orphan node check (check 1),
        # which applies -5 pts per orphan. Adding a separate memory penalty would
        # double-count the same structural issue.

        total_penalty = min(60.0, sum(breakdown.values()))

        runtime = RuntimeHealth(
            total_penalty=total_penalty,
            breakdown=breakdown,
            notes=rh_notes,
        )

        # ── Final Score ────────────────────────────────────────────────────────
        final_score = max(0.0, cov_score - total_penalty)

        if final_score >= 92:
            verdict = "EXCELLENT -- Structurally correct, expected to be runnable"
        elif final_score >= 80:
            verdict = "GOOD -- Functionally sound, minor issues"
        elif final_score >= 65:
            verdict = "ACCEPTABLE -- Core pattern present, notable issues"
        elif final_score >= 45:
            verdict = "NEEDS WORK -- Significant structural problems"
        else:
            verdict = "POOR -- Does not implement the required workflow"

        # ── Detail lines (for CLI display) ─────────────────────────────────────
        detail: List[str] = []
        detail.append(f"\n{'='*62}")
        detail.append(f"  EVALUATION: {name}")
        detail.append(f"{'='*62}")
        detail.append(f"Reference  : {len(ref_nodes)} nodes, {len(ref_edges)} edges")
        detail.append(f"Generated  : {len(gen_nodes)} nodes, {len(gen_edges)} edges")
        detail.append(f"Ref roles  : {', '.join(ref_roles_list)}")
        detail.append(f"Gen roles  : {', '.join(sorted(gen_roles))}")
        detail.append(f"")
        detail.append(f"Node Coverage Score   : {cov_score:.1f} / 100")
        detail.append(f"  Covered  : {', '.join(covered_roles) or '(none)'}")
        detail.append(f"  Missing  : {', '.join(missing_roles) or '(none)'}")
        detail.append(f"Runtime Health Penalty: -{total_penalty:.1f} pts")
        for cat, pts in breakdown.items():
            detail.append(f"  {cat}: -{pts:.0f} pts")
        detail.append(f"Final Score           : {final_score:.1f} / 100")
        detail.append(f"Verdict               : {verdict}")

        return EvaluationResult(
            workflow_name=name,
            node_coverage=coverage,
            runtime_health=runtime,
            final_score=final_score,
            verdict=verdict,
            detail_lines=detail,
        )


# -- Report printer (CLI) -------------------------------------------------------

def print_report(result: EvaluationResult) -> None:
    for line in result.detail_lines:
        print(line)

    print(f"\n{'-'*62}")
    print(f"  NODE COVERAGE  {result.node_coverage.score:.1f} / 100")
    print(f"{'-'*62}")
    for note in result.node_coverage.notes:
        print(f"  {note}")

    print(f"\n{'-'*62}")
    print(f"  RUNTIME HEALTH  Penalty: -{result.runtime_health.total_penalty:.1f} pts")
    print(f"{'-'*62}")
    for note in result.runtime_health.notes:
        print(f"  {note}")

    print(f"\n{'='*62}")
    print(f"  FINAL SCORE : {result.final_score:.1f} / 100")
    print(f"  VERDICT     : {result.verdict}")
    print(f"{'='*62}\n")


# -- CLI -----------------------------------------------------------------------

def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_dataset_entry(dataset: List[Dict], index: int) -> Optional[Dict]:
    if index and 1 <= index <= len(dataset):
        return dataset[index - 1]
    return None


def extract_index(filename: str) -> Optional[int]:
    m = re.match(r"^(\d+)_", os.path.basename(filename))
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Two-component Langflow workflow evaluator")
    parser.add_argument("--reference",     "-r", help="Reference workflow JSON")
    parser.add_argument("--generated",     "-g", help="Generated workflow JSON")
    parser.add_argument("--dataset",       "-d", help="Dataset JSON array")
    parser.add_argument("--generated-dir",       help="Directory of generated JSONs")
    parser.add_argument("--name",          "-n", default="", help="Workflow label")
    args = parser.parse_args()

    evaluator = WorkflowEvaluator()
    scores: List[float] = []

    if args.reference and args.generated:
        ref    = load_json(args.reference)
        gen    = load_json(args.generated)
        name   = args.name or os.path.basename(args.generated)
        result = evaluator.evaluate(ref, gen, name=name)
        print_report(result)
        scores.append(result.final_score)

    elif args.dataset and args.generated_dir:
        dataset   = load_json(args.dataset)
        gen_files = sorted(glob.glob(os.path.join(args.generated_dir, "*.json")))
        if not gen_files:
            print(f"No JSON files found in {args.generated_dir}")
            sys.exit(1)

        print(f"\nEvaluating {len(gen_files)} generated workflow(s)...\n")
        for gf in gen_files:
            idx       = extract_index(gf)
            ref_entry = find_dataset_entry(dataset, idx) if idx else None
            if ref_entry is None:
                print(f"[SKIP] {os.path.basename(gf)} -- no matching dataset entry")
                continue
            ref_flow = ref_entry.get("workflow") or ref_entry
            gen_flow = load_json(gf)
            name     = ref_entry.get("name", os.path.basename(gf))
            result   = evaluator.evaluate(ref_flow, gen_flow, name=name)
            print_report(result)
            scores.append(result.final_score)

        if scores:
            avg = sum(scores) / len(scores)
            print(f"\n{'='*62}")
            print(f"  BATCH SUMMARY")
            print(f"{'='*62}")
            for gf, s in zip(gen_files, scores):
                print(f"  {os.path.basename(gf):50s}  {s:5.1f}%")
            print(f"{'-'*62}")
            print(f"  AVERAGE SCORE : {avg:.1f}%")
            print(f"{'='*62}\n")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
