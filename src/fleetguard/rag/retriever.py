from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fleetguard.config.settings import PATHS
from fleetguard.rag.build_manuals import generate_manual_markdown_files

load_dotenv()


class OemManualRetriever:
    """
    Document-backed RAG retriever:
    1. Loads Markdown OEM manuals from `data/manuals/`.
    2. Splits structurally by Markdown headers into LangChain `Document` chunks.
    3. Embeds chunks via local Ollama (`OllamaEmbeddings`, e.g., `nomic-embed-text`)
       with an automatic TF-IDF cosine fallback if Ollama is unreachable.
    """

    def __init__(self, manuals_dir: Path | None = None):
        self.manuals_dir = manuals_dir or PATHS.manuals_dir
        if not self.manuals_dir.exists() or not list(self.manuals_dir.glob("*.md")):
            generate_manual_markdown_files(self.manuals_dir)

        self.chunks: list[Document] = self._load_and_split_manuals()
        self.embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

        self.vector_store: InMemoryVectorStore | None = None
        self.backend_name: str = "TF-IDF Lexical Fallback"
        self._init_index()

    def _load_and_split_manuals(self) -> list[Document]:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "manual_title"), ("##", "section")],
            strip_headers=False,
        )
        char_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)

        all_chunks: list[Document] = []
        for md_path in sorted(self.manuals_dir.glob("*.md")):
            raw_text = md_path.read_text(encoding="utf-8")
            doc_id = md_path.name.split("_")[0]

            header_docs = header_splitter.split_text(raw_text)
            split_docs = char_splitter.split_documents(header_docs)

            for chunk in split_docs:
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["source_file"] = md_path.name
                all_chunks.append(chunk)

        return all_chunks

    def _init_index(self) -> None:
        """Attempt dense neural indexing via local Ollama; fall back cleanly to TF-IDF."""
        try:
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(
                model=self.embed_model,
                base_url=self.ollama_base_url,
            )
            # Nomic-embed-text achieves higher retrieval accuracy with task prefixes
            prefixed_docs = [
                Document(
                    page_content=f"search_document: {d.metadata.get('manual_title', '')} | {d.metadata.get('section', '')}\n{d.page_content}",
                    metadata={**d.metadata, "raw_content": d.page_content},
                )
                for d in self.chunks
            ]
            store = InMemoryVectorStore(embedding=embeddings)
            store.add_documents(prefixed_docs)
            self.vector_store = store
            self.backend_name = f"Ollama Neural Embeddings ({self.embed_model})"
        except Exception:
            corpus_texts = [
                f"{doc.metadata.get('manual_title', '')} {doc.metadata.get('section', '')} {doc.page_content}"
                for doc in self.chunks
            ]
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
            self.doc_matrix = self.vectorizer.fit_transform(corpus_texts)
            self.backend_name = "TF-IDF Lexical Fallback"

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top_k relevant manual chunks using Ollama dense vectors (or TF-IDF fallback)."""
        if self.vector_store is not None:
            formatted_query = f"search_query: {query}" if "nomic" in self.embed_model else query
            hits_with_scores = self.vector_store.similarity_search_with_score(formatted_query, k=top_k)
            results = []
            for doc, score in hits_with_scores:
                results.append(
                    {
                        "doc_id": doc.metadata.get("doc_id", "UNKNOWN"),
                        "source_file": doc.metadata.get("source_file", ""),
                        "manual_title": doc.metadata.get("manual_title", ""),
                        "section": doc.metadata.get("section", "Overview"),
                        "relevance_score": round(float(score), 4),
                        "retrieval_backend": self.backend_name,
                        "content": doc.metadata.get("raw_content", doc.page_content).strip(),
                    }
                )
            return results

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).ravel()
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            doc = self.chunks[int(idx)]
            results.append(
                {
                    "doc_id": doc.metadata.get("doc_id", "UNKNOWN"),
                    "source_file": doc.metadata.get("source_file", ""),
                    "manual_title": doc.metadata.get("manual_title", ""),
                    "section": doc.metadata.get("section", "Overview"),
                    "relevance_score": round(float(scores[idx]), 4),
                    "retrieval_backend": self.backend_name,
                    "content": doc.page_content.strip(),
                }
            )
        return results


if __name__ == "__main__":
    retriever = OemManualRetriever()
    print(f"Retrieval Backend: {retriever.backend_name}")
    print(f"Indexed {len(retriever.chunks)} Markdown chunks from {retriever.manuals_dir}\n")

    sample_query = "truck overheating on steep climb, coolant temperature spiking"
    print(f"Semantic Query: '{sample_query}'\n")
    for hit in retriever.retrieve(sample_query, top_k=2):
        print(f"[{hit['doc_id']} | {hit['section']}] (cosine_sim={hit['relevance_score']})")
        print(hit["content"][:220] + "...\n")