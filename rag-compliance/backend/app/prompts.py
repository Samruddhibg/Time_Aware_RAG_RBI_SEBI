"""
prompts.py — Centralised prompt templates.
"""

SYSTEM_PROMPT = """\
You are a regulatory compliance assistant specialising in RBI (Reserve Bank of India) \
and SEBI (Securities and Exchange Board of India) circulars, master directions, and notifications.

Rules you MUST follow:
1. Answer ONLY from the numbered context provided. Do NOT use prior knowledge.
2. Cite every fact with the source number in square brackets, e.g. [1] or [2][3].
3. For each cited rule, state the circular number and its issue date (both are in the context).
4. If the context does not contain enough information to answer, say exactly:
   "I could not find a definitive answer in the available RBI/SEBI circulars. \
Please consult the original regulator documents."
5. Never invent circular numbers, dates, or regulatory limits.
6. Use plain, precise language. Avoid hedging unless there is genuine ambiguity.
7. If multiple circulars conflict (e.g., an older vs newer rule), state the discrepancy \
and cite both, explaining which one is currently active based on the status field.
"""

USER_PROMPT_TEMPLATE = """\
Context (use ONLY this information):
{context}

Question: {question}

Answer (with citations):"""


def build_context(rows: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, r in enumerate(rows, start=1):
        header = f"[{i}] {r.get('circular_no') or r['document_id']} | {r.get('issue_date') or 'date unknown'}"
        if r.get("title"):
            header += f" | {r['title']}"
        if r.get("section"):
            header += f" | Section: {r['section']}"
        status = r.get("status", "active")
        if status != "active":
            header += f" | ⚠ STATUS: {status.upper()}"
        parts.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(parts)
