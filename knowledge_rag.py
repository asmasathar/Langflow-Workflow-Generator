from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from shared_memory import SharedMemory


# ── TF-IDF ────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_vectors(docs: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    n = len(docs)
    if n == 0:
        return [], {}
    tf_docs: List[Dict[str, float]] = []
    df: Dict[str, int] = defaultdict(int)
    for doc in docs:
        tokens = _tokenize(doc)
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        total = max(len(tokens), 1)
        tf = {t: c / total for t, c in tf.items()}
        tf_docs.append(tf)
        for t in tf:
            df[t] += 1
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    vectors: List[Dict[str, float]] = []
    for tf in tf_docs:
        vec = {t: tf_val * idf[t] for t, tf_val in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})
    return vectors, idf


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    return sum(a[t] * b[t] for t in set(a) & set(b))


def _query_vec(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    tokens = _tokenize(query)
    tf: Dict[str, float] = defaultdict(float)
    for t in tokens:
        tf[t] += 1.0
    total = max(len(tokens), 1)
    vec = {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {t: v / norm for t, v in vec.items()}


# ── Document ───────────────────────────────────────────────────────────────────

@dataclass
class RAGDocument:
    doc_id:  str
    kind:    str     # "workflow_example" | "edge_pattern" | "component_fact" | "documentation"
    text:    str     # searchable text
    payload: str     # LLM-ready content
    vector:  Dict[str, float] = field(default_factory=dict)


# ── Main class ─────────────────────────────────────────────────────────────────

class KnowledgeRAG:
    def __init__(self, mem: SharedMemory):
        self.mem   = mem
        self.docs: List[RAGDocument] = []
        self._idf: Dict[str, float] = {}
        self._retrieval_counts: Dict[str, int] = defaultdict(int)
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    def build(self, include_docs: bool = True) -> None:
        """Build the full knowledge index. Call once after all knowledge is loaded."""
        self.docs = []

        self._index_workflow_examples()
        self._index_edge_patterns()
        self._index_component_facts()

        if include_docs:
            self._index_documentation()

        texts = [d.text for d in self.docs]
        vectors, idf = _tfidf_vectors(texts)
        self._idf = idf
        for doc, vec in zip(self.docs, vectors):
            doc.vector = vec

        self._built = True
        by_kind = defaultdict(int)
        for d in self.docs:
            by_kind[d.kind] += 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items()))
        print(f"[RAG] Knowledge index: {len(self.docs)} docs ({summary})")

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        kinds: Optional[List[str]] = None,
    ) -> str:
        if not self._built or not self.docs:
            return ""
        qvec  = _query_vec(query_text, self._idf)
        pool  = [d for d in self.docs if kinds is None or d.kind in kinds]
        scored = sorted([(d, _cosine(qvec, d.vector)) for d in pool], key=lambda x: -x[1])
        top   = [d for d, s in scored[:top_k] if s > 0.01]
        if not top:
            return ""
        for d in top:
            self._retrieval_counts[d.doc_id] += 1
        return "\n".join([f"=== RETRIEVED (query: {query_text!r}) ==="] + [d.payload for d in top])

    def usage_report(self) -> Dict[str, Any]:
        """Return index/retrieval usage metrics for run summaries."""
        indexed_by_kind: Dict[str, int] = defaultdict(int)
        for d in self.docs:
            indexed_by_kind[d.kind] += 1

        retrieved_by_kind: Dict[str, int] = defaultdict(int)
        for d in self.docs:
            count = self._retrieval_counts.get(d.doc_id, 0)
            if count > 0:
                retrieved_by_kind[d.kind] += count

        unused_by_kind: Dict[str, int] = {}
        for kind, indexed in indexed_by_kind.items():
            used_docs = sum(
                1 for d in self.docs
                if d.kind == kind and self._retrieval_counts.get(d.doc_id, 0) > 0
            )
            unused_by_kind[kind] = max(indexed - used_docs, 0)

        top_used_docs = sorted(
            self._retrieval_counts.items(), key=lambda kv: -kv[1]
        )[:10]

        return {
            "indexed_by_kind": dict(indexed_by_kind),
            "retrieved_by_kind": dict(retrieved_by_kind),
            "unused_by_kind": dict(unused_by_kind),
            "top_used_docs": top_used_docs,
        }

    def format_for_prompt(
        self,
        query: str,
        top_k: int = 4,
        kinds: Optional[List[str]] = None,
        max_chars: int = 1200,
    ) -> str:
        raw = self.query(query, top_k=top_k, kinds=kinds)
        if len(raw) > max_chars:
            raw = raw[:max_chars] + "\n[truncated]"
        return raw

    def query_edge_patterns(self, component_types: List[str], workflow_query: str = "") -> str:
        q = " ".join(component_types) + " " + workflow_query
        return self.query(q, top_k=5, kinds=["edge_pattern", "workflow_example"])

    def query_component_facts(self, component_type: str) -> str:
        return self.query(component_type, top_k=3, kinds=["component_fact", "documentation"])

    # ── Indexers ──────────────────────────────────────────────────────────────

    def _index_workflow_examples(self) -> None:
        for ex in self.mem.example_flows:
            name = ex.get("name", "")
            desc = ex.get("description", "")
            nodes, edges = SharedMemory._unwrap_flow_data(ex)

            nmap = {n.get("id"): n.get("data", {}).get("type", "") for n in nodes}
            comp_types = list(dict.fromkeys(
                n.get("data", {}).get("type", "") for n in nodes
                if n.get("data", {}).get("type", "") not in ("note", "noteNode", "")
            ))

            edge_lines = []
            for e in edges:
                sh = e.get("data", {}).get("sourceHandle", {})
                th = e.get("data", {}).get("targetHandle", {})
                src = nmap.get(e.get("source", ""), "?")
                tgt = nmap.get(e.get("target", ""), "?")
                if sh.get("name") and th.get("fieldName") and src != "?" and tgt != "?":
                    edge_lines.append(f"  {src}.{sh['name']} -> {tgt}.{th['fieldName']}")

            if not comp_types:
                continue

            payload = "\n".join(filter(None, [
                f"[WORKFLOW: {name}]",
                f"Components: {', '.join(comp_types)}",
                f"Description: {desc[:100]}" if desc else "",
                "Connections:" if edge_lines else "",
                *edge_lines[:10],
            ]))

            self.docs.append(RAGDocument(
                doc_id  = f"wf_{name}",
                kind    = "workflow_example",
                text    = f"{name} {desc} {' '.join(comp_types)}",
                payload = payload,
            ))

    def _index_edge_patterns(self) -> None:
        seen: set = set()
        for ex in self.mem.example_flows:
            nodes, edges = SharedMemory._unwrap_flow_data(ex)
            nmap  = {n.get("id"): n.get("data", {}).get("type", "") for n in nodes}

            for e in edges:
                sh = e.get("data", {}).get("sourceHandle", {})
                th = e.get("data", {}).get("targetHandle", {})
                src_type = nmap.get(e.get("source", ""), "")
                tgt_type = nmap.get(e.get("target", ""), "")
                src_out  = sh.get("name", "")
                tgt_fld  = th.get("fieldName", "")
                tgt_it   = th.get("inputTypes", [])
                src_ot   = sh.get("output_types", [])

                if not (src_type and tgt_type and src_out and tgt_fld):
                    continue
                if src_type in ("note", "noteNode") or tgt_type in ("note", "noteNode"):
                    continue

                key = f"{src_type}.{src_out}->{tgt_type}.{tgt_fld}"
                if key in seen:
                    continue
                seen.add(key)

                payload = f"[EDGE PATTERN] {src_type}.{src_out} -> {tgt_type}.{tgt_fld}"
                if src_ot:
                    payload += f"  output_types={src_ot}"
                if tgt_it:
                    payload += f"  inputTypes={tgt_it}"

                self.docs.append(RAGDocument(
                    doc_id  = f"edge_{key}",
                    kind    = "edge_pattern",
                    text    = f"{src_type} {tgt_type} {src_out} {tgt_fld} edge connection {ex.get('name','')}",
                    payload = payload,
                ))

    def _index_component_facts(self) -> None:
        for wt, ck in self.mem.knowledge_index.items():
            roles = []
            if ck.is_source:        roles.append("source/entry-point")
            if ck.is_sink:          roles.append("sink/exit-point")
            if ck.is_llm:           roles.append("language-model/agent")
            if ck.is_tool_provider: roles.append("tool-provider")

            lines = [
                f"[COMPONENT: {wt}] {ck.description[:80]}",
                f"  roles: {', '.join(roles) or 'none'}",
                f"  outputs: {', '.join(f'{n}[{chr(47).join(ck.output_types[:2])}]' for n in ck.output_names[:4]) or 'none'}",
            ]
            connectable_in = [
                f"{fn}[{'/'.join(ck.input_types_map.get(fn, ['?'])[:2])}]"
                for fn in ck.input_names[:6]
            ]
            if connectable_in:
                lines.append(f"  connectable inputs: {', '.join(connectable_in)}")
            if ck.required_fields:
                lines.append(f"  required fields: {', '.join(ck.required_fields)}")
            for f in ck.key_fields[:8]:
                if f["options"]:
                    lines.append(f"  field {f['name']}({f['type']}) options={f['options']}")
                elif f["required"]:
                    lines.append(f"  field {f['name']}({f['type']}) REQUIRED  {f['info']}")

            self.docs.append(RAGDocument(
                doc_id  = f"comp_{wt}",
                kind    = "component_fact",
                text    = f"{wt} {ck.display_name} {ck.description} {' '.join(roles)}",
                payload = "\n".join(lines),
            ))

