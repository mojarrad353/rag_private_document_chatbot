import os
import csv
import time
import pytest
import re
from typing import List, Dict

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "docugami")
PDF_DIR = os.path.join(FIXTURES_DIR, "pdfs")
CSV_PATH = os.path.join(FIXTURES_DIR, "qna_data.csv")


def parse_ground_truth(csv_path: str, limit: int = 100) -> List[Dict]:
    cases = []
    if not os.path.exists(csv_path):
        return cases

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Question Type"] == "Multi-Doc RAG":
                answer = row["Answer"]
                # Extract sources from the end of the answer
                match = re.search(r"SOURCE\(S\):\s*(.*)", answer, re.IGNORECASE)
                sources = []
                if match:
                    # e.g. '"2022 Q3 AAPL.pdf", "2023 Q1 AAPL.pdf"' or '2022 Q3 AAPL.pdf, 2023 Q1 AAPL.pdf'
                    source_str = match.group(1).replace('"', "")
                    sources = [s.strip() for s in source_str.split(",")]

                # Filter out anything that doesn't end in .pdf (sometimes it says 'Item 2. Management...')
                clean_sources = []
                for s in sources:
                    if ".pdf" in s:
                        clean_sources.append(s.split(".pdf")[0] + ".pdf")

                if clean_sources:
                    cases.append(
                        {
                            "query": row["Question"],
                            "sources": clean_sources,
                        }
                    )

            if len(cases) >= limit:
                break
    return cases


@pytest.fixture(scope="module")
def multi_doc_retriever(tmpdir_factory):
    """Sets up the end-to-end PDR + Cross-Encoder pipeline for the 20 PDFs."""
    persist_dir = str(tmpdir_factory.mktemp("docugami_chroma"))
    docstore_dir = str(tmpdir_factory.mktemp("docugami_docstore"))

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    vs = Chroma(
        collection_name="docugami",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    fs = LocalFileStore(docstore_dir)
    store = create_kv_docstore(fs)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    pdr = ParentDocumentRetriever(
        vectorstore=vs,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Load all 20 PDFs
    all_docs = []
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(PDF_DIR, filename)
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                # Store only the filename in the metadata source to match the ground truth
                doc.metadata["source"] = filename
            all_docs.extend(docs)

    pdr.add_documents(all_docs)
    pdr.search_kwargs = {"k": 10}

    # Wrap in Cross-Encoder
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)

    retriever_pipeline = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=pdr,
    )

    return retriever_pipeline


class TestMultiDocPrecision:
    def test_precision_and_recall(self, multi_doc_retriever):
        cases = parse_ground_truth(CSV_PATH, limit=100)
        assert len(cases) > 0, "No cases found or CSV not downloaded properly."

        precisions_at_4 = []
        recalls_at_4 = []
        f1_scores = []
        reciprocal_ranks = []
        accuracies = []
        latencies = []

        print("\n" + "=" * 80)
        print("  MULTI-DOC EVALUATION REPORT (k=4)")
        print("=" * 80)

        for case in cases:
            query = case["query"]
            ground_truth_pdfs = set(case["sources"])

            start_time = time.perf_counter()
            results = multi_doc_retriever.invoke(query)
            latency = (time.perf_counter() - start_time) * 1000  # in ms
            latencies.append(latency)

            # Extract retrieved source PDFs (top 4)
            retrieved_pdfs_at_4 = []
            for doc in results[:4]:
                retrieved_pdfs_at_4.append(doc.metadata.get("source", ""))

            retrieved_set_4 = set(retrieved_pdfs_at_4)

            # True Positives
            true_positives_4 = len(retrieved_set_4.intersection(ground_truth_pdfs))

            # Precision@4
            p_at_4 = true_positives_4 / 4.0
            precisions_at_4.append(p_at_4)

            # Recall@4
            r_at_4 = (
                true_positives_4 / len(ground_truth_pdfs) if ground_truth_pdfs else 0.0
            )
            recalls_at_4.append(r_at_4)

            # F1 Score@4
            if p_at_4 + r_at_4 > 0:
                f1_4 = 2 * (p_at_4 * r_at_4) / (p_at_4 + r_at_4)
            else:
                f1_4 = 0.0
            f1_scores.append(f1_4)

            # Accuracy (Hit Rate @ 4: 1 if at least one correct doc retrieved, else 0)
            accuracy = 1.0 if true_positives_4 > 0 else 0.0
            accuracies.append(accuracy)

            # MRR (Mean Reciprocal Rank of the FIRST relevant document in top 10)
            rank = 0
            for i, doc in enumerate(results):
                if doc.metadata.get("source", "") in ground_truth_pdfs:
                    rank = i + 1
                    break
            rr = 1.0 / rank if rank > 0 else 0.0
            reciprocal_ranks.append(rr)

        avg_precision = sum(precisions_at_4) / len(precisions_at_4)
        avg_recall = sum(recalls_at_4) / len(recalls_at_4)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_latency = sum(latencies) / len(latencies)

        print(f"Tested Queries: {len(cases)}")
        print(f"Recall@4:       {avg_recall:.2%}")
        print(f"Precision@4:    {avg_precision:.2%}")
        print(f"F1 Score@4:     {avg_f1:.2%}")
        print(f"Accuracy@4:     {avg_accuracy:.2%}")
        print(f"MRR:            {mrr:.4f}")
        print(f"Avg Latency:    {avg_latency:.1f} ms")
        print("=" * 80)

        # Basic sanity check
        assert avg_precision > 0.05
