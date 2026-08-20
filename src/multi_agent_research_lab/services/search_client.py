"""Search client abstraction supporting Tavily API, offline corpus, and mock fallback."""

import json
from pathlib import Path

import httpx

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client supporting Tavily, rich offline corpus, and fallback."""

    def __init__(self, corpus_dir: str | Path | None = None) -> None:
        self.corpus_dir = Path(corpus_dir or "ai_agent_offline_research_corpus_v2/topics")
        self.settings = get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Tries in order:
        1. Tavily API (if TAVILY_API_KEY is configured).
        2. Offline benchmark corpus (ai_agent_offline_research_corpus_v2).
        3. Deterministic mock documents.
        """
        # 1. Try Tavily search if API key is provided
        if self.settings.tavily_api_key:
            try:
                tavily_results = self._search_tavily(query, max_results=max_results)
                if tavily_results:
                    return tavily_results
            except Exception:
                pass

        # 2. Try Offline Corpus
        if self.corpus_dir.exists():
            corpus_results = self._search_offline_corpus(query, max_results=max_results)
            if corpus_results:
                return corpus_results

        # 3. Fallback mock documents
        return self._search_mock(query, max_results=max_results)

    def _search_tavily(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        results: list[SourceDocument] = []
        for item in data.get("results", []):
            results.append(
                SourceDocument(
                    title=item.get("title", "Tavily Search Result"),
                    url=item.get("url"),
                    snippet=item.get("content", "")[:1000],
                    metadata={"provider": "tavily", "score": item.get("score")},
                )
            )
        return results

    def _search_offline_corpus(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        query_tokens = {t.lower() for t in query.replace("-", " ").split() if len(t) > 2}
        candidate_docs: list[tuple[int, SourceDocument]] = []

        for json_file in sorted(self.corpus_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                topic_name = data.get("topic", {}).get("name", "")
                knowledge_base = data.get("knowledge_base", {})
                articles = knowledge_base.get("knowledge_articles", [])
                source_docs = data.get("source_documents", []) or data.get("sources", [])

                for art in articles:
                    art_id = art.get("article_id", "Article")
                    title = f"[{topic_name}] {art.get('title', art_id)}"
                    content = art.get("content", "")
                    text_for_match = f"{title} {content}".lower()
                    score = sum(2 for tok in query_tokens if tok in text_for_match)
                    if score > 0:
                        candidate_docs.append(
                            (
                                score,
                                SourceDocument(
                                    title=title,
                                    url=f"corpus://{json_file.stem}#{art_id}",
                                    snippet=content[:1000],
                                    metadata={
                                        "provider": "offline-corpus",
                                        "topic": topic_name,
                                        "source_type": "knowledge_article",
                                        "credibility": "peer_reviewed_summary",
                                    },
                                ),
                            )
                        )

                for sdoc in source_docs:
                    s_id = sdoc.get("source_id", "Source")
                    title = sdoc.get("title", s_id)
                    summary = sdoc.get("summary", sdoc.get("content", ""))
                    text_for_match = f"{title} {summary}".lower()
                    score = sum(2 for tok in query_tokens if tok in text_for_match)
                    if score > 0:
                        candidate_docs.append(
                            (
                                score,
                                SourceDocument(
                                    title=title,
                                    url=sdoc.get("url") or f"corpus://{json_file.stem}#{s_id}",
                                    snippet=summary[:1000],
                                    metadata={
                                        "provider": "offline-corpus",
                                        "topic": topic_name,
                                        "source_type": (
                                            "synthetic_benchmark"
                                            if sdoc.get("is_synthetic")
                                            else "public_reference"
                                        ),
                                        "credibility": (
                                            "synthetic_benchmark"
                                            if sdoc.get("is_synthetic")
                                            else "high_provenance"
                                        ),
                                    },
                                ),
                            )
                        )
            except Exception:
                continue

        candidate_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in candidate_docs[:max_results]]

    def _search_mock(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [
            SourceDocument(
                title=f"Architectural Foundations and State-of-the-Art in {query}",
                url="https://arxiv.org/abs/2402.example1",
                snippet=(
                    f"Comprehensive analysis of {query}. Discusses graph-based knowledge "
                    "extraction, hierarchical community summarization, and retrieval trade-offs."
                ),
                metadata={"provider": "mock", "credibility": "academic_preprint"},
            ),
            SourceDocument(
                title=f"Empirical Evaluation & Production Benchmarks for {query}",
                url="https://github.com/microsoft/graphrag-benchmarks",
                snippet=(
                    f"Production benchmark metrics for {query}. Evaluates latency, token overhead, "
                    "and multi-hop query answering performance compared to vector-only RAG."
                ),
                metadata={"provider": "mock", "credibility": "engineering_whitepaper"},
            ),
        ][:max_results]
