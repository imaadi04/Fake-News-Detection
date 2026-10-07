import os
import re
import json
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

BASE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = BASE

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = "gemini-3.8-flash"

app = FastAPI(
    title="TruthCheck API",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND),
    name="static"
)


class AnalyzeRequest(BaseModel):
    text: str = Field("", max_length=50000)
    url: str = Field("", max_length=2000)


# =========================================================
# BASIC HELPERS
# =========================================================

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def extract_article(url):
    try:
        response = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "Chrome/120 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside"
        ]):
            tag.decompose()

        title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else url
        )

        paragraphs = []

        for p in soup.find_all("p"):
            text = clean(p.get_text(" ", strip=True))

            if len(text) >= 30:
                paragraphs.append(text)

        body = clean(" ".join(paragraphs))

        if len(body) < 100:
            body = clean(
                soup.get_text(" ", strip=True)
            )

        return title, body[:50000]

    except Exception as e:
        raise HTTPException(
            400,
            f"Could not read article URL: {e}"
        )


# =========================================================
# CLAIM EXTRACTION
# =========================================================

def claims_from(text):

    text = clean(text)

    parts = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    claims = []

    for part in parts:

        part = part.strip()

        if 25 <= len(part) <= 700:
            claims.append(part)

    if not claims:
        return [text[:700]]

    return claims[:10]


# =========================================================
# SOURCE QUALITY
# =========================================================

def source_quality(url):

    domain = urlparse(url).netloc.lower()

    # Extremely strong sources
    very_high = [
        ".gov",
        ".gov.in",
        ".nic.in",
        "worldbank.org",
        "imf.org",
        "who.int",
        "un.org",
        "unicef.org",
        "rbi.org.in",
        "nasa.gov",
        "isro.gov.in",
        "oecd.org",
        "ec.europa.eu",
    ]

    for source in very_high:
        if source in domain:
            return "very_high"

    # Strong news organizations
    high = [
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "thehindu.com",
        "indianexpress.com",
        "ndtv.com",
        "economictimes.indiatimes.com",
        "livemint.com",
        "hindustantimes.com",
    ]

    for source in high:
        if source in domain:
            return "high"

    # Known but not authoritative
    medium = [
        "timesofindia.indiatimes.com",
        "news18.com",
        "moneycontrol.com",
        "business-standard.com",
    ]

    for source in medium:
        if source in domain:
            return "medium"

    return "unknown"


# =========================================================
# SERPER / GOOGLE SEARCH
# =========================================================

def search_web(claim):

    if not SERPER_API_KEY:
        return []

    queries = [
        claim,
        f'"{claim}" fact check',
    ]

    results = []
    seen = set()

    for query in queries:

        try:

            response = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": SERPER_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": 8
                },
                timeout=15
            )

            response.raise_for_status()

            data = response.json()

            for item in data.get("organic", []):

                url = item.get("link", "").strip()

                if not url:
                    continue

                if url in seen:
                    continue

                seen.add(url)

                results.append({
                    "title": clean(
                        item.get("title", "Untitled")
                    ),
                    "snippet": clean(
                        item.get("snippet", "")
                    ),
                    "url": url,
                    "publisher": urlparse(url).netloc,
                    "source_quality": source_quality(url),
                    "is_demo": False
                })

        except Exception:
            continue

    return results[:12]


# =========================================================
# GEMINI
# =========================================================

def gemini_request(prompt):

    if not GEMINI_API_KEY:
        return None

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2000,
            "responseMimeType": "application/json"
        }
    }

    try:

        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        candidates = data.get(
            "candidates",
            []
        )

        if not candidates:
            return None

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        if not parts:
            return None

        output = parts[0].get(
            "text",
            ""
        ).strip()

        if not output:
            return None

        # Remove accidental markdown fences
        output = re.sub(
            r"^```json\s*",
            "",
            output,
            flags=re.IGNORECASE
        )

        output = re.sub(
            r"\s*```$",
            "",
            output
        )

        return json.loads(output)

    except Exception:
        return None


# =========================================================
# CLAIM FACT CHECK
# =========================================================

def fact_check_claim(claim, evidence):

    if not evidence:

        return {
            "score": 50,
            "verdict": "UNCERTAIN",
            "reason": (
                "No reliable web evidence was found "
                "for this claim."
            ),
            "confidence": 20,
            "key_evidence": []
        }

    evidence_text = []

    for i, item in enumerate(evidence, 1):

        evidence_text.append(
            f"""
SOURCE {i}
Title: {item['title']}
Publisher: {item['publisher']}
Source quality: {item['source_quality']}
URL: {item['url']}
Snippet: {item['snippet']}
"""
        )

    evidence_block = "\n".join(evidence_text)

    prompt = f"""
You are the fact-checking engine of a news verification website.

Your task is to determine whether the USER CLAIM is supported,
contradicted, or uncertain based ONLY on the provided web evidence.

USER CLAIM:
{claim}

WEB EVIDENCE:
{evidence_block}

IMPORTANT RULES:

1. Do NOT assume the user's claim is true.

2. Do NOT decide that a claim is true simply because Google
   returned search results.

3. Compare the actual meaning of the claim with the evidence.

4. Carefully check:
   - numbers
   - percentages
   - dates
   - years
   - names
   - locations
   - quantities
   - cause/effect statements
   - "before/after" statements

5. A source that merely discusses the same topic is NOT proof
   that the claim is true.

6. Prefer high-quality sources:
   government, official institutions, World Bank, IMF, WHO,
   RBI, research organizations, Reuters, AP, BBC and other
   established sources.

7. If multiple reliable sources agree, confidence should increase.

8. If reliable sources directly contradict the claim,
   the claim should be CONTRADICTED.

9. If evidence is incomplete or ambiguous, use UNCERTAIN.

10. Never invent facts.

11. Never invent a source.

12. Do not use your own memory when the supplied evidence
    can answer the question.

13. If sources disagree, mention the disagreement.

14. Do NOT give 80-90% merely because many search results
    exist.

15. The score represents how strongly the available evidence
    supports the claim:
       90-100 = very strong support
       75-89  = strong support
       55-74  = moderate / mixed support
       35-54  = weak / uncertain
       0-34   = strong contradiction

16. If the evidence directly contradicts an important numerical
    or factual part of the claim, score should normally be below 35.

17. If there is not enough evidence to decide, keep the score
    around 45-55 rather than guessing.

Return ONLY valid JSON:

{{
    "score": integer between 0 and 100,
    "verdict": "SUPPORTED" or "CONTRADICTED" or "UNCERTAIN",
    "reason": "short explanation of why",
    "confidence": integer between 0 and 100,
    "key_evidence": [
        {{
            "source_number": integer,
            "finding": "short explanation"
        }}
    ]
}}
"""

    result = gemini_request(prompt)

    if not result:
        return {
            "score": 50,
            "verdict": "UNCERTAIN",
            "reason": (
                "AI verification could not be completed. "
                "Please verify using the listed sources."
            ),
            "confidence": 0,
            "key_evidence": []
        }

    # Safety / validation
    try:
        score = int(result.get("score", 50))
    except Exception:
        score = 50

    score = max(0, min(100, score))

    verdict_value = str(
        result.get("verdict", "UNCERTAIN")
    ).upper()

    if verdict_value not in [
        "SUPPORTED",
        "CONTRADICTED",
        "UNCERTAIN"
    ]:
        verdict_value = "UNCERTAIN"

    return {
        "score": score,
        "verdict": verdict_value,
        "reason": clean(
            str(
                result.get(
                    "reason",
                    "Insufficient evidence."
                )
            )
        ),
        "confidence": max(
            0,
            min(
                100,
                int(
                    result.get(
                        "confidence",
                        50
                    )
                )
            )
        ),
        "key_evidence": result.get(
            "key_evidence",
            []
        )
    }


# =========================================================
# OVERALL VERDICT
# =========================================================

def overall_verdict(score):

    if score >= 85:
        return "Strongly supported"

    if score >= 70:
        return "Mostly supported"

    if score >= 55:
        return "Mixed / needs verification"

    if score >= 35:
        return "Likely false / weak evidence"

    return "Strongly contradicted"


# =========================================================
# ANALYZE
# =========================================================

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):

    text = clean(req.text)
    source_title = ""

    # ---------------------------------------------
    # URL MODE
    # ---------------------------------------------

    if req.url.strip():

        url = clean(req.url)

        if not url.startswith(
            ("http://", "https://")
        ):
            raise HTTPException(
                400,
                "Please enter a valid URL."
            )

        source_title, text = extract_article(url)

    if len(text) < 20:

        raise HTTPException(
            400,
            "Please provide a paragraph or article URL."
        )

    # ---------------------------------------------
    # CLAIMS
    # ---------------------------------------------

    claims = claims_from(text)

    results = []
    all_sources = []

    # ---------------------------------------------
    # FACT CHECK EACH CLAIM
    # ---------------------------------------------

    for claim in claims:

        evidence = search_web(claim)

        verification = fact_check_claim(
            claim,
            evidence
        )

        results.append({
            "claim": claim,
            "support": verification["score"],
            "verdict": verification["verdict"],
            "reason": verification["reason"],
            "confidence": verification["confidence"],
            "evidence": evidence
        })

        all_sources.extend(evidence)

    # ---------------------------------------------
    # OVERALL SCORE
    # ---------------------------------------------

    if results:

        weighted_scores = []

        for result in results:

            # Confidence slightly affects the final score,
            # but never dominates the AI judgement.
            score = result["support"]
            confidence = result["confidence"]

            weight = 0.75 + (
                confidence / 100
            ) * 0.25

            weighted_scores.append(
                score * weight
            )

        final_score = round(
            sum(weighted_scores)
            / sum(
                0.75 + (
                    r["confidence"] / 100
                ) * 0.25
                for r in results
            )
        )

    else:
        final_score = 50

    # ---------------------------------------------
    # RELATED SOURCES
    # ---------------------------------------------

    related = []
    seen = set()

    # First prefer reliable sources
    priority_order = {
        "very_high": 0,
        "high": 1,
        "medium": 2,
        "unknown": 3
    }

    all_sources.sort(
        key=lambda x: priority_order.get(
            x.get("source_quality", "unknown"),
            3
        )
    )

    for source in all_sources:

        url = source.get("url", "")

        if not url:
            continue

        if url in seen:
            continue

        seen.add(url)

        related.append(source)

        if len(related) >= 10:
            break

    return {
        "truth_score": final_score,
        "verdict": overall_verdict(
            final_score
        ),
        "source_title": source_title,
        "claims_analyzed": len(claims),
        "claims": results,
        "related_information": related,
        "demo_mode": not bool(
            SERPER_API_KEY
        ),
        "ai_mode": bool(
            GEMINI_API_KEY
        ),
        "disclaimer": (
            "This is an evidence-based AI estimate, "
            "not a guarantee of truth. Always inspect "
            "the cited sources for important claims."
        )
    }


# =========================================================
# BASIC ROUTES
# =========================================================

@app.get("/")
def home():

    return FileResponse(
        os.path.join(
            FRONTEND,
            "index.html"
        )
    )


@app.get("/health")
def health():

    return {
        "status": "ok",
        "serper_configured": bool(
            SERPER_API_KEY
        ),
        "gemini_configured": bool(
            GEMINI_API_KEY
        )
    }
