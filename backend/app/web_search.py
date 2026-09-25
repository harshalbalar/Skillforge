"""
SkillForge — Web Search Module (Phase 5).

Gives agents access to real-time web data via Tavily Search API.
The researcher agent calls this before generating answers, so outputs
are grounded in current 2026 data instead of just Gemini's training.

Usage:
  results = await web_search("latest Next.js version 2026")
  # Returns formatted search results with titles, URLs, and content
"""

import os
import asyncio
from typing import Optional

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


def get_tavily_client() -> Optional[object]:
    """Get Tavily client if API key is set."""
    if not TAVILY_AVAILABLE:
        return None
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return None
    return TavilyClient(api_key=api_key)


async def web_search(query: str, max_results: int = 5, search_depth: str = "basic") -> dict:
    """
    Search the web using Tavily and return structured results.

    Args:
        query: The search query
        max_results: Number of results to return (1-10)
        search_depth: "basic" (fast, cheaper) or "advanced" (thorough)

    Returns:
        dict with "success", "results" (list), "query", and "result_count"
    """
    client = get_tavily_client()
    if not client:
        return {
            "success": False,
            "error": "Tavily not available. Install: pip install tavily-python. Set TAVILY_API_KEY in .env.",
            "results": [],
        }

    try:
        response = await asyncio.to_thread(
            client.search,
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
            })

        return {
            "success": True,
            "query": query,
            "answer": response.get("answer", ""),
            "results": results,
            "result_count": len(results),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": [],
        }


async def multi_search(queries: list[str], max_results: int = 3) -> list[dict]:
    """
    Run multiple searches in parallel for faster research.

    Args:
        queries: List of search queries
        max_results: Results per query

    Returns:
        List of search result dicts
    """
    tasks = [web_search(q, max_results=max_results) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            output.append({
                "success": False,
                "query": queries[i],
                "error": str(r),
                "results": [],
            })
        else:
            output.append(r)

    return output


def format_search_results(search_data: dict) -> str:
    """
    Format search results into a text block that can be injected into agent prompts.
    """
    if not search_data.get("success") or not search_data.get("results"):
        return f"[No web results found for: {search_data.get('query', 'unknown')}]"

    lines = [f"Web search results for: \"{search_data['query']}\""]

    if search_data.get("answer"):
        lines.append(f"\nQuick answer: {search_data['answer']}")

    lines.append("")

    for i, r in enumerate(search_data["results"], 1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    URL: {r['url']}")
        lines.append(f"    {r['content'][:300]}")
        lines.append("")

    return "\n".join(lines)


def format_multi_search(results: list[dict]) -> str:
    """Format multiple search results into one text block."""
    sections = []
    for r in results:
        sections.append(format_search_results(r))
    return "\n---\n".join(sections)