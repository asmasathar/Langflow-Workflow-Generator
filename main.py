from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_env(env_path: str = ".env") -> None:
    """Load .env file into os.environ (without overwriting existing vars)."""
    for p in [Path(env_path), Path(__file__).parent / ".env"]:
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())
            return


_load_env()

from shared_memory import SharedMemory
from knowledge_fetcher import KnowledgeFetcher
from knowledge_rag import KnowledgeRAG
from planning_agent import PlanningAgent
from graph_agent import GraphAgent
from compiler import WorkflowCompiler
from validator import WorkflowValidator
from refiner import WorkflowRefiner


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Langflow Multi-Agent Workflow Generator"
    )
    p.add_argument("--prompt", default="",
                   help="Initial workflow description (skip first prompt if provided)")
    p.add_argument("--output", default="langflow_workflow.json",
                   help="Output JSON file path")
    p.add_argument("--no-questions", action="store_true",
                   help="Skip interactive follow-up questions and auto-fill fields")
    return p.parse_args()


# ── Banner & summary ───────────────────────────────────────────────────────────

def _banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║       Langflow Multi-Agent Workflow Generator                ║
║  Plan → Graph → Compile → Validate → Refine → Save          ║
╚══════════════════════════════════════════════════════════════╝
""")


def _summary(mem: SharedMemory) -> None:
    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print("=" * 62)
    name  = mem.final_workflow.get("name", "?")
    nodes = mem.final_workflow.get("data", {}).get("nodes", [])
    edges = mem.final_workflow.get("data", {}).get("edges", [])
    print(f"  Workflow : {name}")
    print(f"  Nodes    : {len(nodes)}")
    print(f"  Edges    : {len(edges)}")
    print(f"  Saved to : {mem.workflow_path}")

    base = os.getenv("LANGFLOW_BASE_URL", "http://localhost:7860")
    print(f"\n  To import into Langflow:")
    print(f"    1. Open {base}")
    print(f"    2. Click '+ New Flow' → 'Import'")
    print(f"    3. Select: {mem.workflow_path}")

    unique_warnings = list(dict.fromkeys(mem.warnings))
    unique_errors   = list(dict.fromkeys(mem.errors))

    if unique_warnings:
        print(f"\n  Warnings ({len(unique_warnings)}):")
        for w in unique_warnings[:12]:
            print(f"    ⚠  {w}")
    if unique_errors:
        print(f"\n  Errors ({len(unique_errors)}):")
        for e in unique_errors[:12]:
            print(f"    ✗  {e}")
    print()


def _rag_summary(rag: KnowledgeRAG) -> None:
    report  = rag.usage_report()
    indexed = report.get("indexed_by_kind", {})
    used    = report.get("retrieved_by_kind", {})
    unused  = report.get("unused_by_kind", {})

    print("\n" + "=" * 62)
    print("  KNOWLEDGE USAGE")
    print("=" * 62)
    for k in sorted(set(indexed) | set(used)):
        print(f"  {k}: indexed={indexed.get(k,0)} retrieved={used.get(k,0)} unused={unused.get(k,0)}")


def _agent_context_summary(mem: SharedMemory) -> None:
    """Show which knowledge each agent actually used — proves full context sharing."""
    print("\n" + "=" * 62)
    print("  AGENT CONTEXT USAGE  (proves all agents share full context)")
    print("=" * 62)
    for agent, used in sorted(mem.agent_knowledge_usage.items()):
        keys = sorted(k for k, v in used.items() if v > 0)
        counts = ", ".join(f"{k}={v}" for k, v in sorted(used.items(), key=lambda x: -x[1])[:6])
        print(f"  {agent}: {counts or '(none)'}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    _banner()
    args = parse_args()

    # ── Validate env ──────────────────────────────────────────────────────
    missing_env = [v for v in ("LANGFLOW_BASE_URL", "LANGFLOW_API_KEY") if not os.getenv(v)]
    if missing_env:
        print(f"[ERROR] Missing environment variables: {', '.join(missing_env)}")
        print("       Set them in .env or your shell.")
        return 1
    if not os.getenv("LOCAL_LLM_API_URL") or not os.getenv("LOCAL_LLM_MODEL"):
        print("[ERROR] LOCAL_LLM_API_URL and LOCAL_LLM_MODEL must be set in .env")
        return 1

    print(f"[Config] Langflow: {os.getenv('LANGFLOW_BASE_URL')}")
    print(f"[Config] LLM:      {os.getenv('LOCAL_LLM_API_URL')} / {os.getenv('LOCAL_LLM_MODEL')}")
    print(f"[Config] Output:   {args.output}")
    print(f"[Config] No Q&A:   {args.no_questions}")

    if args.no_questions:
        os.environ["WF_NO_QUESTIONS"] = "1"

    # ── SharedMemory — the single source of truth shared by ALL agents ────
    mem = SharedMemory()
    mem.user_prompt   = args.prompt.strip()
    mem.workflow_path = args.output

    # ── Phase 1: Knowledge ─────────────────────────────────────────────────
    print("\n[Phase 1] Loading Langflow knowledge...")
    try:
        fetcher = KnowledgeFetcher()
        fetcher.load_all_into_memory(mem)
    except RuntimeError as exc:
        print(f"\n[ERROR] Knowledge loading failed: {exc}")
        print("        Is Langflow running at LANGFLOW_BASE_URL?")
        return 1

    if not mem.live_components:
        print("[ERROR] No components loaded from Langflow.")
        return 1

    print("[Phase 1] Building component knowledge index...", end=" ", flush=True)
    mem.build_knowledge_index(fetcher)
    print(f"{len(mem.knowledge_index)} components indexed")

    include_docs = _bool_env("WF_ENABLE_DOC_SCRAPE", False)
    rag = KnowledgeRAG(mem)
    rag.build(include_docs=include_docs)

    # Reset transient state (keep user_prompt / workflow_path)
    user_prompt   = mem.user_prompt
    workflow_path = mem.workflow_path
    mem.reset_transient_state()
    mem.user_prompt   = user_prompt
    mem.workflow_path = workflow_path

    # ── Phase 2: Planning ──────────────────────────────────────────────────
    print("\n[Phase 2] Starting planning session...")
    try:
        PlanningAgent(mem, fetcher, rag=rag).run()
    except KeyboardInterrupt:
        print("\n[Cancelled by user]")
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Planning failed: {exc}")
        traceback.print_exc()
        return 1

    if not mem.selected_components:
        print("[ERROR] No components were selected.")
        return 1

    # ── Phase 3: Graph ─────────────────────────────────────────────────────
    print("\n[Phase 3] Building edge topology...")
    graph    = GraphAgent(mem, rag=rag)
    compiler = WorkflowCompiler(mem)

    try:
        graph.run()
    except Exception as exc:
        print(f"\n[ERROR] Graph building failed: {exc}")
        traceback.print_exc()
        return 1

    # ── Phase 4: Compile ───────────────────────────────────────────────────
    print("\n[Phase 4] Compiling Langflow JSON...")
    try:
        compiler.run()
    except Exception as exc:
        print(f"\n[ERROR] Compilation failed: {exc}")
        traceback.print_exc()
        return 1

    # ── Phase 5: Proactive refinement ──────────────────────────────────────
    print("\n[Phase 5] Proactive pre-validation refinement pass...")
    refiner   = WorkflowRefiner(mem, fetcher, rag=rag)
    validator = WorkflowValidator(mem)

    try:
        proactive_changed = refiner.run(
            "PROACTIVE_PASS: "
            "Fill any missing required field values that cannot be edge-fed. "
            "Ensure every Prompt template variable has a matching incoming edge. "
            "Ensure no orphan nodes or duplicate target-field edges exist."
        )
        if proactive_changed:
            print("[Phase 5] Proactive changes applied — re-running graph + compile...")
            graph.run()
            compiler.run()
        else:
            print("[Phase 5] No proactive changes needed.")
    except Exception as exc:
        print(f"[Phase 5] Proactive pass failed (non-fatal): {exc}")

    # ── Phase 6: Validate + Refine loop (up to 3 passes) ──────────────────
    print("\n[Phase 6] Validate → Refine loop...")
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        print(f"\n[Validator] Pass {attempt}/{max_attempts}...")
        result = validator.validate(mem.final_workflow)
        print(result.report())

        if result.is_valid:
            print(f"[Validator] ✓ Workflow is valid (pass {attempt}).")
            mem.warnings.extend(result.warnings)
            break

        print(f"[Refiner] Applying corrections (attempt {attempt}/{max_attempts})...")

        try:
            before_n = len(mem.selected_components)
            before_e = len(mem.abstract_graph.get("edges", []))

            changed = refiner.run(result.report())

            after_n = len(mem.selected_components)
            after_e = len(mem.abstract_graph.get("edges", []))
            print(
                f"[Refiner] Delta: components {before_n}→{after_n}, "
                f"edges {before_e}→{after_e}"
            )

            if not changed:
                print("[Refiner] No further changes possible — accepting current state.")
                mem.errors.extend(result.errors)
                mem.warnings.extend(result.warnings)
                break

            graph.run()
            compiler.run()

        except Exception as exc:
            print(f"[Refiner] Error during attempt {attempt}: {exc}")
            traceback.print_exc()
            mem.errors.extend(result.errors)
            mem.warnings.extend(result.warnings)
            break
    else:
        print("\n[WARNING] Max correction attempts reached — saving latest state.")
        mem.errors.extend(result.errors)
        mem.warnings.extend(result.warnings)

    # ── Summaries ──────────────────────────────────────────────────────────
    _rag_summary(rag)
    _agent_context_summary(mem)
    _summary(mem)

    # Success if workflow was saved (even with warnings)
    return 0 if Path(mem.workflow_path).exists() else 1


if __name__ == "__main__":
    sys.exit(main())
