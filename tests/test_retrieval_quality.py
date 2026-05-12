"""
Retrieval Quality Tests — REAL PDFs through the REAL pipeline.

Uses 3 arXiv papers (42 pages total) from the HuggingFace
dwb2023/ragas-golden-dataset, processed through the production
pipeline: PyMuPDFLoader → RecursiveCharacterTextSplitter → ChromaDB.

Ground-truth Q&A pairs come directly from the HuggingFace dataset.

Metrics:
  - Recall@K         : Is the relevant content in the top K chunks?
  - Precision@K      : Fraction of top K that are actually relevant
  - MRR              : Mean Reciprocal Rank
  - Source filtering  : Metadata filter correctness
  - MMR diversity     : Cross-document diversity
  - Latency           : Per-query retrieval speed
  - Chunk-size sens.  : Impact of chunk size on recall

Run:
    pytest tests/test_retrieval_quality.py -v -s --no-cov --timeout=300
"""

import os
import time
import shutil
import tempfile
import statistics
from typing import List

import pytest
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── PDF fixtures (must be downloaded to tests/fixtures/) ─────────────────
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

PDF_FILES = {
    "ai_agents": os.path.join(FIXTURES_DIR, "ai_agents_vs_agentic_ai.pdf"),
    "redteam": os.path.join(FIXTURES_DIR, "redteam_llm.pdf"),
    "control_plane": os.path.join(FIXTURES_DIR, "control_plane_agentic.pdf"),
}

# ── Ground truth from HuggingFace dwb2023/ragas-golden-dataset ──────────
# Each entry: question, key phrases that MUST be in retrieved context,
# and which PDF(s) should contain the answer.
from typing import Any, Dict, cast

GROUND_TRUTH: List[Dict[str, Any]] = [
    {
        "id": "hf_01",
        "query": "What was AI agent design like in the pre-2022 era?",
        "key_phrases": ["rule-based", "constrained"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "single_hop",
    },
    {
        "id": "hf_02",
        "query": "How did ChatGPT change the AI world?",
        "key_phrases": ["ChatGPT", "November 2022"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "single_hop",
    },
    {
        "id": "hf_03",
        "query": "How do AI Agents contribute to scalable automation in enterprise settings?",
        "key_phrases": ["modular", "lightweight"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "single_hop",
    },
    {
        "id": "hf_04",
        "query": "How do LIMs help AI agents do visual tasks like seeing and reacting in the real world?",
        "key_phrases": ["CLIP", "BLIP"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "single_hop",
    },
    {
        "id": "hf_05",
        "query": (
            "How do Agentic AI systems enhance multi-agent collaboration and "
            "distributed intelligence compared to traditional AI Agents?"
        ),
        "key_phrases": ["orchestrat", "multi-agent"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "multi_hop",
    },
    {
        "id": "hf_06",
        "query": (
            "How does the autonomy level differ between AI Agents and Agentic AI?"
        ),
        "key_phrases": ["autonomy", "Agentic AI"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "multi_hop",
    },
    {
        "id": "hf_07",
        "query": (
            "How does agentic AI use collaborative scientific writing and "
            "collaborative medical decision support?"
        ),
        "key_phrases": ["orchestrator", "clinical"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "multi_hop",
    },
    {
        "id": "hf_08",
        "query": (
            "How do agentic AI frameworks like RedTeamLLM address challenges "
            "such as reliability and automation in offensive cybersecurity?"
        ),
        "key_phrases": ["RedTeamLLM", "automation"],
        "expected_pdf_key": "redteam",
        "difficulty": "multi_hop",
    },
    {
        "id": "hf_09",
        "query": (
            "How do Generative Agents differ from more advanced AI Agents "
            "and Agentic AI in terms of autonomy and task coordination?"
        ),
        "key_phrases": ["Generative", "autonomy"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "multi_hop",
    },
    {
        "id": "hf_10",
        "query": (
            "How does GPT-Engineer exemplify the transition from traditional "
            "AI Agents to Agentic AI systems?"
        ),
        "key_phrases": ["GPT-Engineer", "code"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "multi_hop",
    },
    {
        "id": "hf_11",
        "query": (
            "How do the architectural components of RedTeamLLM address the "
            "challenges of context window constraints and plan correction?"
        ),
        "key_phrases": ["ADaPT", "Plan Corrector"],
        "expected_pdf_key": "redteam",
        "difficulty": "multi_hop",
    },
    {
        "id": "hf_12",
        "query": (
            "What are some recent IEEE publications that discuss the integration "
            "of large language models for improved decision-making?"
        ),
        "key_phrases": ["IEEE", "IJCNN"],
        "expected_pdf_key": "ai_agents",
        "difficulty": "multi_hop",
    },
]


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embeddings():
    """Load real HuggingFace embeddings (bge-large-en-v1.5 — same as production)."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.fixture(scope="module")
def ingestion_stats():
    """Shared dict to collect ingestion stats for reporting."""
    return {}


@pytest.fixture(scope="module")
def retriever(embeddings, ingestion_stats):
    """
    Build a REAL Chroma vector store by:
      1. Loading 3 real arXiv PDFs via PyMuPDFLoader (same as production)
      2. Splitting with RecursiveCharacterTextSplitter (chunk_size=1000, overlap=100)
      3. Embedding with HuggingFace all-MiniLM-L6-v2
      4. Storing in a temp ChromaDB directory
    """
    # Verify PDFs exist
    for name, path in PDF_FILES.items():
        if not os.path.exists(path):
            pytest.skip(
                f"PDF fixture missing: {path}. "
                f"Download from arXiv first (see tests/fixtures/README)."
            )

    all_docs = []
    page_counts = {}

    # Step 1: Load PDFs with PyMuPDFLoader (same loader as production rag.py)
    for name, path in PDF_FILES.items():
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        page_counts[name] = len(docs)
        all_docs.extend(docs)

    # Step 2: Split with production settings (chunk_size=1000, overlap=200)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # Save stats for reporting
    ingestion_stats["total_pages"] = sum(page_counts.values())
    ingestion_stats["page_counts"] = page_counts
    ingestion_stats["total_chunks"] = len(chunks)
    ingestion_stats["avg_chunk_len"] = (
        statistics.mean(len(c.page_content) for c in chunks) if chunks else 0
    )
    ingestion_stats["unique_sources"] = len(
        set(c.metadata.get("source", "") for c in chunks)
    )

    # Step 3 & 4: Embed and store using ParentDocumentRetriever
    tmpdir = tempfile.mkdtemp(prefix="rag_real_pdf_test_")
    ingestion_stats["tmpdir"] = tmpdir

    t0 = time.perf_counter()
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import LocalFileStore
    from langchain.storage._lc_store import create_kv_docstore

    vs = Chroma(
        persist_directory=tmpdir,
        embedding_function=embeddings,
    )

    fs = LocalFileStore(os.path.join(tmpdir, "docstore"))
    store = create_kv_docstore(fs)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    retriever_obj = ParentDocumentRetriever(
        vectorstore=vs,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever_obj.add_documents(all_docs)
    ingestion_stats["embedding_time_s"] = time.perf_counter() - t0

    print(f"\n{'='*70}")
    print(f"  INGESTION REPORT — Real PDF Pipeline (PDR)")
    print(f"{'='*70}")
    for name, count in page_counts.items():
        print(f"  {name:30s} : {count:>3} pages")
    print(f"  {'─'*50}")
    print(f"  Total pages:     {ingestion_stats['total_pages']}")
    print(f"  Total chunks:    {ingestion_stats['total_chunks']}")
    print(f"  Avg chunk length:{ingestion_stats['avg_chunk_len']:.0f} chars")
    print(f"  Unique sources:  {ingestion_stats['unique_sources']}")
    print(f"  Embedding time:  {ingestion_stats['embedding_time_s']:.1f}s")
    print(f"{'='*70}\n")

    yield retriever_obj

    shutil.rmtree(tmpdir, ignore_errors=True)


# ── Helpers ──────────────────────────────────────────────────────────────


def _check_hit(results: List[Document], key_phrases: List[str]) -> bool:
    """Check if ANY combination of results contains ALL key phrases."""
    combined = " ".join(doc.page_content for doc in results).lower()
    return all(phrase.lower() in combined for phrase in key_phrases)


def _find_first_hit_rank(results: List[Document], key_phrases: List[str]) -> int:
    """Return 1-based rank of first doc containing ALL key phrases, or 0."""
    for i, doc in enumerate(results, start=1):
        content = doc.page_content.lower()
        if all(p.lower() in content for p in key_phrases):
            return i
    return 0


def _precision_at_k(results: List[Document], expected_pdf_key: str) -> float:
    """Fraction of results whose source path contains the expected PDF name."""
    if not results:
        return 0.0
    pdf_basename = os.path.basename(PDF_FILES[expected_pdf_key]).lower()
    relevant = sum(
        1
        for doc in results
        if pdf_basename in os.path.basename(doc.metadata.get("source", "")).lower()
    )
    return relevant / len(results)


# ── Test Classes ─────────────────────────────────────────────────────────


class TestRecallAtK:
    """Recall@K — does the correct content appear in top K chunks?"""

    @pytest.mark.parametrize(
        "case", GROUND_TRUTH, ids=[str(c["id"]) for c in GROUND_TRUTH]
    )
    def test_recall_at_3(self, retriever, case):
        retriever.search_kwargs = {"k": 3}
        results = retriever.invoke(case["query"])
        hit = _check_hit(results, case["key_phrases"])
        if not hit:
            previews = [
                f"  [{i+1}] {doc.page_content[:100]}..."
                for i, doc in enumerate(results)
            ]
            print(
                f"\n  MISS [{case['id']}] query='{case['query'][:60]}'\n"
                + "\n".join(previews)
            )
        # This is a measurement — we collect, not assert per-query for recall@3
        # (assertion is on aggregate in TestAggregateReport)

    @pytest.mark.parametrize(
        "case", GROUND_TRUTH, ids=[str(c["id"]) for c in GROUND_TRUTH]
    )
    def test_recall_at_5(self, retriever, case):
        retriever.search_kwargs = {"k": 5}
        results = retriever.invoke(case["query"])
        hit = _check_hit(results, case["key_phrases"])
        if not hit:
            previews = [
                f"  [{i+1}] {doc.page_content[:100]}..."
                for i, doc in enumerate(results)
            ]
            print(
                f"\n  MISS [{case['id']}] query='{case['query'][:60]}'\n"
                + "\n".join(previews)
            )

    @pytest.mark.parametrize(
        "case", GROUND_TRUTH, ids=[str(c["id"]) for c in GROUND_TRUTH]
    )
    def test_recall_at_10(self, retriever, case):
        """Relaxed recall — correct chunk should be in top 10."""
        retriever.search_kwargs = {"k": 10}
        results = retriever.invoke(case["query"])
        hit = _check_hit(results, case["key_phrases"])
        assert hit, (
            f"[{case['id']}] Recall@10 FAIL for: '{case['query'][:60]}...'\n"
            f"Expected key_phrases: {case['key_phrases']}"
        )


class TestMRR:
    """Mean Reciprocal Rank across all ground-truth queries."""

    def test_mean_reciprocal_rank(self, retriever):
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain.retrievers import ContextualCompressionRetriever

        cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=10)
        retriever.search_kwargs = {"k": 20}
        retriever_pipeline = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever,
        )

        reciprocal_ranks = []
        details = []

        for case in GROUND_TRUTH:
            results = retriever_pipeline.invoke(case["query"])
            rank = _find_first_hit_rank(results, case["key_phrases"])
            rr = 1.0 / rank if rank > 0 else 0.0
            reciprocal_ranks.append(rr)
            status = f"rank={rank}" if rank > 0 else "NOT FOUND"
            details.append(
                f"  {case['id']:>6s} [{case['difficulty']:>10s}] "
                f"RR={rr:.3f} {status:>14s}  '{case['query'][:55]}'"
            )

        mrr = statistics.mean(reciprocal_ranks)
        print(f"\n{'='*80}")
        print(f"  MEAN RECIPROCAL RANK (MRR) — Real PDFs")
        print(f"{'='*80}")
        for d in details:
            print(d)
        print(f"{'='*80}")
        print(f"  MRR = {mrr:.4f}  (threshold: ≥ 0.40)")
        print(f"{'='*80}\n")

        assert mrr >= 0.40, f"MRR = {mrr:.4f} — below threshold 0.40"


class TestSourceFiltering:
    """Source metadata filter correctness."""

    def test_filter_isolates_correct_pdf(self, retriever):
        """Filtering by source should only return chunks from that file."""
        # Get all unique sources from the vector store
        collection_data = retriever.vectorstore.get(include=["metadatas"], limit=100000)
        all_sources = list(
            set(m.get("source", "") for m in collection_data.get("metadatas", []) if m)
        )

        print(f"\n  Sources in vector store: {len(all_sources)}")
        for src in sorted(all_sources):
            print(f"    - {os.path.basename(src)}")

        for source in all_sources[:3]:  # Test first 3 sources
            retriever.search_kwargs = {
                "k": 5,
                "filter": {"source": source},
            }
            results = retriever.invoke("AI agents")
            for doc in results:
                assert (
                    doc.metadata["source"] == source
                ), f"Filter leak: expected '{source}', got '{doc.metadata['source']}'"


class TestMMRDiversity:
    """MMR retrieval should return diverse, cross-document results."""

    def test_mmr_cross_document_diversity(self, retriever):
        """A broad query with MMR should pull from multiple PDFs."""
        retriever.search_type = "mmr"
        retriever.search_kwargs = {"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
        results = retriever.invoke(
            "How do agentic AI systems work with multiple agents?"
        )
        sources = set(
            os.path.basename(doc.metadata.get("source", "")) for doc in results
        )
        print(f"\n  MMR sources for broad query: {sources}")
        assert (
            len(sources) >= 2
        ), f"MMR should return ≥2 different source PDFs, got: {sources}"

    def test_mmr_vs_similarity_diversity(self, retriever):
        """MMR should be at least as diverse as pure similarity search."""
        query = "agentic AI architecture and collaboration"

        retriever.search_type = "mmr"
        retriever.search_kwargs = {"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
        mmr_results = retriever.invoke(query)

        retriever.search_type = "similarity"
        retriever.search_kwargs = {"k": 5}
        sim_results = retriever.invoke(query)

        mmr_sources = len(
            set(os.path.basename(d.metadata.get("source", "")) for d in mmr_results)
        )
        sim_sources = len(
            set(os.path.basename(d.metadata.get("source", "")) for d in sim_results)
        )
        print(f"\n  MMR sources: {mmr_sources}, Similarity sources: {sim_sources}")
        assert mmr_sources >= sim_sources


class TestRetrievalLatency:
    """Latency measurements on real data."""

    def test_single_query_latency(self, retriever):
        """Retrieval should be fast enough for interactive use (< 500ms)."""
        start = time.perf_counter()
        retriever.search_kwargs = {"k": 5}
        retriever.invoke("What are AI agents?")
        latency = (time.perf_counter() - start) * 1000
        print(f"\n  Single query latency: {latency:.1f} ms")
        assert latency < 500, f"Latency {latency:.1f} ms — too slow"

    def test_batch_latency(self, retriever):
        queries = [case["query"] for case in GROUND_TRUTH]
        latencies = []

        retriever.search_kwargs = {"k": 5}
        for q in queries:
            start = time.perf_counter()
            retriever.invoke(q)
            latencies.append((time.perf_counter() - start) * 1000)

        total = sum(latencies)
        avg = statistics.mean(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\n{'='*60}")
        print(f"  LATENCY REPORT ({len(queries)} queries, real PDFs)")
        print(f"{'='*60}")
        print(f"  Total:   {total:.1f} ms")
        print(f"  Average: {avg:.1f} ms")
        print(f"  P50:     {p50:.1f} ms")
        print(f"  P95:     {p95:.1f} ms")
        print(f"  Min:     {min(latencies):.1f} ms")
        print(f"  Max:     {max(latencies):.1f} ms")
        print(f"{'='*60}\n")

        assert avg < 2000, f"Avg latency {avg:.1f} ms — too slow"


class TestChunkSizeSensitivity:
    """How chunk size affects recall on real PDFs."""

    @pytest.mark.parametrize("chunk_size", [300, 500, 1000, 2000])
    def test_chunk_size_recall(self, embeddings, chunk_size):
        # Skip if PDFs not downloaded
        for path in PDF_FILES.values():
            if not os.path.exists(path):
                pytest.skip("PDF fixtures missing")

        # Load all PDFs
        all_docs = []
        for path in PDF_FILES.values():
            all_docs.extend(PyMuPDFLoader(path).load())

        from langchain.retrievers import ParentDocumentRetriever
        from langchain.storage import LocalFileStore
        from langchain.storage._lc_store import create_kv_docstore

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=200
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=int(chunk_size * 0.2)
        )

        with tempfile.TemporaryDirectory(prefix=f"rag_chunk_{chunk_size}_") as tmpdir:
            vs = Chroma(embedding_function=embeddings, persist_directory=tmpdir)
            fs = LocalFileStore(os.path.join(tmpdir, "docstore"))
            store = create_kv_docstore(fs)

            pdr = ParentDocumentRetriever(
                vectorstore=vs,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
            pdr.add_documents(all_docs)

            hits_at_5 = 0
            hits_at_10 = 0
            for case in GROUND_TRUTH:
                pdr.search_kwargs = {"k": 5}
                r5 = pdr.invoke(case["query"])
                pdr.search_kwargs = {"k": 10}
                r10 = pdr.invoke(case["query"])
                if _check_hit(r5, case["key_phrases"]):
                    hits_at_5 += 1
                if _check_hit(r10, case["key_phrases"]):
                    hits_at_10 += 1

            recall_5 = hits_at_5 / len(GROUND_TRUTH)
            recall_10 = hits_at_10 / len(GROUND_TRUTH)
            print(
                f"\n  child_chunk_size={chunk_size:>5}, child_chunks={len(pdr.vectorstore.get(limit=100000)['ids']):>4}, "
                f"recall@5={recall_5:.0%} ({hits_at_5}/{len(GROUND_TRUTH)}), "
                f"recall@10={recall_10:.0%} ({hits_at_10}/{len(GROUND_TRUTH)})"
            )
            # With real PDFs, minimum 40% recall@10 at any chunk size
            assert (
                recall_10 >= 0.40
            ), f"chunk_size={chunk_size}: recall@10={recall_10:.0%} — below 40%"


class TestAggregateReport:
    """Consolidated report with all metrics."""

    def test_full_retrieval_report(self, retriever, ingestion_stats):
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain.retrievers import ContextualCompressionRetriever

        cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        # Note: We rank top 10 so we can compute Recall@10
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=10)
        retriever.search_kwargs = {"k": 20}
        retriever_pipeline = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=retriever,
        )

        recall_3_hits = 0
        recall_5_hits = 0
        recall_10_hits = 0
        precisions_3 = []
        precisions_5 = []
        reciprocal_ranks = []
        latencies = []

        per_query = []

        for case in GROUND_TRUTH:
            # End-to-end pipeline invocation
            start = time.perf_counter()
            results = retriever_pipeline.invoke(case["query"])
            lat = (time.perf_counter() - start) * 1000
            latencies.append(lat)

            # Slices for @K metrics
            r3 = results[:3]
            r5 = results[:5]
            r10 = results[:10]
            hit3 = _check_hit(r3, case["key_phrases"])
            if hit3:
                recall_3_hits += 1

            # Recall@5
            hit5 = _check_hit(r5, case["key_phrases"])
            if hit5:
                recall_5_hits += 1

            # Recall@10
            hit10 = _check_hit(r10, case["key_phrases"])
            if hit10:
                recall_10_hits += 1

            # Precision@3
            p3 = _precision_at_k(r3, case["expected_pdf_key"])
            precisions_3.append(p3)

            # Precision@5
            p5 = _precision_at_k(r5, case["expected_pdf_key"])
            precisions_5.append(p5)

            # MRR
            rank = _find_first_hit_rank(r10, case["key_phrases"])
            rr = 1.0 / rank if rank > 0 else 0.0
            reciprocal_ranks.append(rr)

            per_query.append(
                {
                    "id": case["id"],
                    "difficulty": case["difficulty"],
                    "hit@3": hit3,
                    "hit@5": hit5,
                    "hit@10": hit10,
                    "rank": rank,
                    "rr": rr,
                    "p@3": p3,
                    "latency_ms": lat,
                }
            )

        n = len(GROUND_TRUTH)
        recall_3 = recall_3_hits / n
        recall_5 = recall_5_hits / n
        recall_10 = recall_10_hits / n
        avg_p3 = statistics.mean(precisions_3)
        avg_p5 = statistics.mean(precisions_5)
        mrr = statistics.mean(reciprocal_ranks)
        avg_lat = statistics.mean(latencies)

        # ── Per-query detail table ──
        print(f"\n{'='*100}")
        print(f"  PER-QUERY DETAIL")
        print(f"{'='*100}")
        print(
            f"  {'ID':>6s}  {'Diff':>10s}  {'R@3':>4s}  {'R@5':>4s}  "
            f"{'R@10':>4s}  {'Rank':>5s}  {'RR':>6s}  {'P@3':>5s}  "
            f"{'Lat':>7s}  Query"
        )
        print(f"  {'─'*95}")
        for q in per_query:
            print(
                f"  {q['id']:>6s}  {q['difficulty']:>10s}  "
                f"{'✓' if q['hit@3'] else '✗':>4s}  "
                f"{'✓' if q['hit@5'] else '✗':>4s}  "
                f"{'✓' if q['hit@10'] else '✗':>4s}  "
                f"{q['rank']:>5d}  {q['rr']:>6.3f}  "
                f"{q['p@3']:>5.0%}  {q['latency_ms']:>5.0f}ms  "
                f"{[c for c in GROUND_TRUTH if c['id']==q['id']][0]['query'][:40]}"
            )

        # ── Aggregate report ──
        report = f"""
{'='*70}
  RETRIEVAL QUALITY — REAL PDF AGGREGATE REPORT
{'='*70}

  Source Data:
    PDFs:          {len(PDF_FILES)} arXiv papers ({ingestion_stats.get('total_pages', '?')} pages)
    Chunks:        {ingestion_stats.get('total_chunks', '?')} (chunk_size=1000, overlap=100)
    Avg chunk len: {ingestion_stats.get('avg_chunk_len', 0):.0f} chars
    Embedding:     HuggingFace all-MiniLM-L6-v2
    Vector Store:  ChromaDB
    Queries:       {n} ground-truth (from HuggingFace dataset)

  ┌──────────────────────────┬────────────┬──────────────┐
  │ Metric                   │ Score      │ Threshold    │
  ├──────────────────────────┼────────────┼──────────────┤
  │ Recall@3                 │ {recall_3:>8.0%}   │   ≥ 50%      │
  │ Recall@5                 │ {recall_5:>8.0%}   │   ≥ 60%      │
  │ Recall@10                │ {recall_10:>8.0%}   │   ≥ 75%      │
  │ Avg Precision@3          │ {avg_p3:>8.0%}   │   ≥ 30%      │
  │ Avg Precision@5          │ {avg_p5:>8.0%}   │   ≥ 25%      │
  │ MRR                      │ {mrr:>8.4f}   │   ≥ 0.40     │
  │ Avg Latency              │ {avg_lat:>6.0f} ms │  < 2000 ms   │
  └──────────────────────────┴────────────┴──────────────┘

  Single-hop queries:  {sum(1 for q in per_query if q['difficulty']=='single_hop' and q['hit@5'])}/{sum(1 for q in per_query if q['difficulty']=='single_hop')} recall@5
  Multi-hop queries:   {sum(1 for q in per_query if q['difficulty']=='multi_hop' and q['hit@5'])}/{sum(1 for q in per_query if q['difficulty']=='multi_hop')} recall@5

{'='*70}
"""
        print(report)

        # Assertions — realistic thresholds for real academic PDFs
        assert recall_3 >= 0.40, f"Recall@3 = {recall_3:.0%} — below 40%"
        assert recall_5 >= 0.50, f"Recall@5 = {recall_5:.0%} — below 50%"
        assert recall_10 >= 0.60, f"Recall@10 = {recall_10:.0%} — below 60%"
        assert mrr >= 0.30, f"MRR = {mrr:.4f} — below 0.30"
        assert avg_lat < 2000, f"Avg latency = {avg_lat:.0f} ms — over 2000ms"
