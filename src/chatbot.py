"""
AI assistant using RAG (Retrieval-Augmented Generation).

1. RETRIEVE: split the knowledge base (data/knowledge/*.md) and any PDFs in data/docs/
   into small chunks, turn them into TF-IDF vectors, and find the chunks most similar
   to the question (cosine similarity).
2. AUGMENT: if the question is about funds (e.g. "best small cap fund"), also pull the
   top-scored funds from our dataset.
3. GENERATE: send question + retrieved text to Gemini, which writes a simple answer.
   With no API key, we show the best-matching knowledge directly (still useful, free).

Why RAG and not just ask the LLM? The answer is grounded in our documents and data,
so it is less likely to make things up, and it can cite fund names from our dataset.
Why TF-IDF and not a vector DB (Chroma)? For a few hundred chunks, TF-IDF is fast,
has no extra dependencies, and runs on free hosting. Chroma + embeddings would be the
next step for thousands of documents.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = """You are a friendly mutual fund guide inside an Indian investment-advisory college project.
Rules:
- Explain in very simple language for someone with no finance background. Short paragraphs or bullets.
- Use the CONTEXT (knowledge base, documents and fund data) as your main source. If the context does not
  cover the question, say so briefly and give only general, well-established information.
- Use ₹ and Indian terms (lakh, crore, SIP, ELSS). If the user writes in Hinglish, reply in Hinglish.
- When mentioning specific funds, only use funds listed in the CONTEXT, and mention past returns are not guaranteed.
- Never promise returns. For big decisions, suggest consulting a SEBI-registered advisor.
- Keep answers under 200 words unless the user asks for detail."""

# words that appear in almost every question ("mutual fund") carry no meaning for search
STOP_WORDS = list(ENGLISH_STOP_WORDS | {"mutual", "fund", "funds", "mf", "kya", "hai", "hota", "ka", "ki", "ke"})

FUND_WORDS = {
    "small cap": "Small Cap", "smallcap": "Small Cap", "mid cap": "Mid Cap", "midcap": "Mid Cap",
    "large cap": "Large Cap", "largecap": "Large Cap", "flexi": "Flexi Cap", "multi cap": "Multi Cap",
    "elss": "ELSS", "tax saving": "ELSS", "index": "Index", "liquid": "Liquid", "gilt": "Gilt",
    "debt": "Debt", "hybrid": "Hybrid", "equity": "Equity", "balanced": "Balanced",
    "value": "Value", "sectoral": "Sectoral", "thematic": "Thematic", "dividend yield": "Dividend Yield",
}


# ----------------------------------------------------------------------------
# Knowledge base
# ----------------------------------------------------------------------------
def _md_chunks(path):
    text = open(path, encoding="utf-8").read()
    for part in re.split(r"\n(?=## )", text):
        part = part.strip()
        if part.startswith("## "):
            title, _, body = part[3:].partition("\n")
            yield {"source": os.path.basename(path), "title": title.strip(), "text": body.strip()}


def _pdf_chunks(path, size=900):
    try:
        from pypdf import PdfReader
    except ImportError:
        return
    try:
        reader = PdfReader(path)
    except Exception:
        return
    name = os.path.basename(path)
    for i, page in enumerate(reader.pages):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        for j in range(0, len(text), size):
            piece = text[j:j + size]
            if len(piece) > 100:
                yield {"source": name, "title": f"{name}, page {i + 1}", "text": piece}


class KnowledgeBase:
    def __init__(self, knowledge_dir="data/knowledge", docs_dir="data/docs"):
        chunks = []
        for p in sorted(glob.glob(os.path.join(knowledge_dir, "*.md"))):
            chunks.extend(_md_chunks(p))
        for p in sorted(glob.glob(os.path.join(docs_dir, "**", "*.pdf"), recursive=True)):
            chunks.extend(_pdf_chunks(p))
        self.chunks = chunks
        # two TF-IDF indexes: one on titles, one on full text. A question like "what is SIP?"
        # should match the section TITLED "What is SIP?" even if another section mentions SIP more often.
        self.vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True)
        self.title_vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, sublinear_tf=True)
        # backup index with normal English stop-words, for questions made only of
        # common words like "what is a mutual fund?"
        self.plain_vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        if chunks:
            texts = [f"{c['title']} {c['text']}" for c in chunks]
            self.matrix = self.vectorizer.fit_transform(texts)
            self.title_matrix = self.title_vectorizer.fit_transform([c["title"] for c in chunks])
            self.plain_matrix = self.plain_vectorizer.fit_transform([f"{c['title']} {c['title']} {c['text']}" for c in chunks])
        else:
            self.matrix = None

    def search(self, query: str, k: int = 3, min_score: float = 0.08):
        if self.matrix is None:
            return []
        body = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        title = cosine_similarity(self.title_vectorizer.transform([query]), self.title_matrix).ravel()
        sims = 0.5 * title + 0.5 * body
        if sims.max() < min_score:                     # nothing found -> try the backup index
            sims = cosine_similarity(self.plain_vectorizer.transform([query]), self.plain_matrix).ravel()
        best = np.argsort(sims)[::-1][:k]
        return [{**self.chunks[i], "score": float(sims[i])} for i in best if sims[i] >= min_score]


# ----------------------------------------------------------------------------
# Fund data lookup
# ----------------------------------------------------------------------------
def fund_context(question: str, scored: pd.DataFrame, n: int = 5):
    q = question.lower()
    asks_pick = re.search(r"\b(best|top|recommend|suggest|which|good|kaunsa|konsa|kon\s?sa)\b", q)
    about_funds = "fund" in q or any(w in q for w in FUND_WORDS)
    if not (asks_pick and about_funds):
        return None, None
    df = scored
    label = "all categories"
    for word, key in FUND_WORDS.items():
        if word in q:
            mask = df["sub_category"].astype(str).str.contains(key, case=False) | \
                   df["category"].astype(str).str.contains(key, case=False)
            if mask.any():
                df, label = df[mask], key
                break
    cols = [c for c in ["scheme_name", "category", "sub_category", "risk_level", "returns_1yr",
                        "returns_3yr", "expense_ratio", "fund_score"] if c in df]
    top = df[cols].head(n).reset_index(drop=True)
    return top, label


# ----------------------------------------------------------------------------
# Answering
# ----------------------------------------------------------------------------
def _gemini(api_key, question, context, history, models=None):
    contents = []
    for turn in history[-6:]:                      # short memory of the conversation
        contents.append({"role": "user" if turn["role"] == "user" else "model",
                         "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]})
    body = {"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]}, "contents": contents,
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800}}
    last_err = None
    for model in (models or GEMINI_MODELS):
        try:
            r = requests.post(GEMINI_URL.format(model=model), json=body, timeout=40,
                              headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})
            if r.status_code == 404:               # model renamed/retired -> try the next one
                last_err = f"{model}: not found"
                continue
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip(), model
        except Exception as e:                      # quota, network, bad key...
            last_err = f"{model}: {e}"
    raise RuntimeError(last_err or "Gemini request failed")


def answer(question: str, kb: KnowledgeBase, scored: pd.DataFrame, history=None,
           api_key: str = None, profile_note: str = None, models=None) -> dict:
    """Returns {"text": markdown answer, "sources": [...], "funds": DataFrame|None, "mode": "ai"|"search"}."""
    history = history or []
    hits = kb.search(question, k=4)
    funds, label = fund_context(question, scored)

    context_parts = [f"[{h['title']}] {h['text']}" for h in hits]
    if funds is not None and not funds.empty:
        context_parts.append(f"Top funds from our dataset ({label}), ranked by our fund score:\n" + funds.to_string(index=False))
    if profile_note:
        context_parts.append(f"The user's latest recommendation in this app: {profile_note}")
    context = "\n\n".join(context_parts) or "(no matching context)"
    sources = [h["title"] for h in hits]

    note = ""
    if api_key:
        try:
            text, model = _gemini(api_key, question, context, history, models)
            return {"text": text, "sources": sources, "funds": funds, "mode": "ai", "model": model}
        except Exception as e:
            note = f"\n\n_(AI answer unavailable: {str(e)[:120]}. Showing search results instead.)_"

    # search-only answer (no key or the API failed)
    if hits:
        text = f"**{hits[0]['title']}**\n\n{hits[0]['text']}"
        if len(hits) > 1:
            text += "\n\n**Related:** " + ", ".join(h["title"] for h in hits[1:3])
    elif funds is not None:
        text = "Here are the top funds from our dataset for your question:"
    else:
        text = ("I couldn't find this in my knowledge base. Try asking about SIP, NAV, expense ratio, ELSS, "
                "taxes, index funds, risk levels, or 'best small cap fund'.")
    if funds is not None and not funds.empty and hits:
        text += f"\n\n**Top {label} funds in our dataset** (by fund score):"
    return {"text": text + note, "sources": sources, "funds": funds, "mode": "search"}
