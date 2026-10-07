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


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

BASE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = BASE

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Current Gemini model
GEMINI_MODEL = "gemini-3.8-flash"


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="TruthCheck API",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# STATIC FILES
# ============================================================

STATIC_DIR = os.path.join(FRONTEND, "static")

# Your current project keeps index.html/style.css/app.js
# in the same root folder, so /static is only mounted
# if the folder actually exists.
if os.path.isdir(STATIC_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static"
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalyzeRequest(BaseModel):
    text: str = Field("", max_length=50000)
    url: str = Field("", max_length=2000)


# ============================================================
# GLOBAL DEBUG STATE
# ============================================================

LAST_GEMINI_ERROR = ""


# ============================================================
# HELPERS
# ============================================================

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


def safe_int(value, default=50):
    try:
        return int(float(value))
    except Exception:
        return default


# ============================================================
# ARTICLE URL EXTRACTION
# ============================================================

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
                    "(KHTML, like Gecko) "
                    "Chrome/154.0 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML
        for tag in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        # Title
        title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else url
        )

        # Paragraphs
        paragraphs = []

        for p in soup.find_all("p"):
            text = clean(
                p.get_text(" ", strip=True)
            )

            if len(text) >= 30:
                paragraphs.append(text)

        body = clean(" ".join(paragraphs))

        # Fallback if paragraphs weren't found
        if len(body) < 100:
            body = clean(
                soup.get_text(" ", strip=True)
            )

        if len(body) < 50:
            raise ValueError(
                "Could not extract enough article text."
            )

        return title, body[:50000]

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read article URL: {str(e)}"
        )


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def claims_from(text):

    text = clean(text)

    # First split using sentence punctuation
    parts = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    claims = []

    for part in parts:

        part = clean(part)

        if 25 <= len(part) <= 800:
            claims.append(part)

    # If sentence splitting didn't work
    if not claims:

        chunks = re.split(
            r"[,;]\s+",
            text
        )

        for chunk in chunks:

            chunk = clean(chunk)

            if len(chunk) >= 25:
                claims.append(chunk)

    # At least one claim
    if not claims:
        claims = [text[:800]]

    # Maximum 8 claims for performance/cost
    return claims[:8]


# ============================================================
# SOURCE QUALITY
# ============================================================

VERY_HIGH_QUALITY = [
    ".gov",
    ".nic.in",
    "worldbank.org",
    "imf.org",
    "who.int",
    "un.org",
    "nasa.gov",
    "isro.gov.in",
    "rbi.org.in",
    "oecd.org",
    "ec.europa.eu"
]

HIGH_QUALITY = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "thehindu.com",
    "indianexpress.com",
    "ndtv.com",
    "timesofindia.indiatimes.com",
    "hindustantimes.com",
    "economictimes.indiatimes.com",
    "bloomberg.com",
    "cnbc.com",
    "espn.com"
]


def source_quality(url):

    domain = urlparse(url).netloc.lower()

    for site in VERY_HIGH_QUALITY:
        if site in domain:
            return 3

    for site in HIGH_QUALITY:
        if site in domain:
            return 2

    return 1


# ============================================================
# SERPER SEARCH
# ============================================================

def search_web(query):

    if not SERPER_API_KEY:

        return [{
            "title": "Live search not configured",
            "snippet": (
                "SERPER_API_KEY is not configured."
            ),
            "url": "https://serper.dev/",
            "publisher": "TruthCheck",
            "is_demo": True,
            "quality": 0
        }]

    try:

        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "q": query,
                "num": 6
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get("organic", [])[:6]:

            link = item.get("link", "").strip()

            if not link:
                continue

            results.append({
                "title": item.get(
                    "title",
                    "Untitled"
                ),
                "snippet": clean(
                    item.get("snippet", "")
                ),
                "url": link,
                "publisher": urlparse(
                    link
                ).netloc,
                "is_demo": False,
                "quality": source_quality(link)
            })

        return results

    except Exception as e:

        print(
            "SERPER ERROR:",
            repr(e)
        )

        return [{
            "title": "Search error",
            "snippet": (
                "Live web search failed."
            ),
            "url": "https://serper.dev/",
            "publisher": "TruthCheck",
            "is_demo": True,
            "quality": 0
        }]


# ============================================================
# COLLECT EVIDENCE
# ============================================================

def collect_evidence(claim):

    queries = [
        claim,
        f'"{claim}"',
        f"{claim} fact check"
    ]

    all_results = []
    seen_urls = set()

    for query in queries:

        results = search_web(query)

        for result in results:

            url = result.get("url", "")

            if (
                result.get("is_demo")
                or not url
                or url in seen_urls
            ):
                continue

            seen_urls.add(url)
            all_results.append(result)

    # Highest quality sources first
    all_results.sort(
        key=lambda x: x.get("quality", 1),
        reverse=True
    )

    return all_results[:10]


# ============================================================
# GEMINI API
# ============================================================

def gemini_request(prompt):

    global LAST_GEMINI_ERROR

    LAST_GEMINI_ERROR = ""

    if not GEMINI_API_KEY:

        LAST_GEMINI_ERROR = (
            "GEMINI_API_KEY is missing."
        )

        print(
            "❌ GEMINI_API_KEY missing"
        )

        return None

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

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
            "responseMimeType": "application/json"
        }
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=40
        )

        print(
            "GEMINI STATUS:",
            response.status_code
        )

        # IMPORTANT:
        # This lets us see the real Gemini error
        # in Render logs instead of silently returning 50.
        if response.status_code != 200:

            print(
                "GEMINI ERROR RESPONSE:",
                response.text[:5000]
            )

            LAST_GEMINI_ERROR = (
                f"Gemini HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

            return None

        data = response.json()

        candidates = data.get(
            "candidates",
            []
        )

        if not candidates:

            LAST_GEMINI_ERROR = (
                "Gemini returned no candidates."
            )

            print(
                "❌ Gemini returned no candidates"
            )

            return None

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        if not parts:

            LAST_GEMINI_ERROR = (
                "Gemini returned no text parts."
            )

            print(
                "❌ Gemini returned no text parts"
            )

            return None

        text = parts[0].get(
            "text",
            ""
        ).strip()

        print(
            "GEMINI RESPONSE:",
            text[:3000]
        )

        if not text:

            LAST_GEMINI_ERROR = (
                "Gemini returned empty text."
            )

            return None

        return text

    except Exception as e:

        LAST_GEMINI_ERROR = repr(e)

        print(
            "❌ GEMINI EXCEPTION:",
            repr(e)
        )

        return None


# ============================================================
# JSON PARSER
# ============================================================

def parse_gemini_json(text):

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences if Gemini adds them
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:

        return json.loads(text)

    except Exception:

        # Try extracting first JSON object
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:

            try:
                return json.loads(
                    text[start:end + 1]
                )

            except Exception:
                pass

    return None


# ============================================================
# GEMINI CLAIM FACT CHECK
# ============================================================

def fact_check_claim(
    claim,
    evidence
):

    # Keep evidence compact
    evidence_text = []

    for index, item in enumerate(
        evidence[:10],
        start=1
    ):

        evidence_text.append(
            f"""
SOURCE {index}
Title: {item.get("title", "")}
Publisher: {item.get("publisher", "")}
URL: {item.get("url", "")}
Quality: {item.get("quality", 1)}
Snippet: {item.get("snippet", "")}
"""
        )

    evidence_block = "\n".join(
        evidence_text
    )

    prompt = f"""
You are an evidence-based fact checking AI.

Your job is to evaluate the user's CLAIM using
ONLY the provided web evidence.

USER CLAIM:
{claim}

WEB EVIDENCE:
{evidence_block}

IMPORTANT RULES:

1. Do NOT assume the claim is true.

2. Compare the actual meaning of the claim with
   the evidence.

3. Carefully compare:
   - numbers
   - percentages
   - currency values
   - dates
   - years
   - names
   - locations
   - quantities
   - cause/effect
   - units

4. For time-sensitive information such as:
   - currency exchange rates
   - stock prices
   - weather
   - sports scores
   - current office holders
   - current policies
   - current statistics

   ALWAYS consider the date/time of the evidence.

5. A value that is slightly outdated but approximately
   correct should NOT automatically be considered false.

   Example:
   If a claim says 1 USD = 95.98 INR and current
   sources say approximately 96.4 INR, this is close.
   If no date is provided, consider it mostly supported
   rather than completely false.

6. If the claim contains an exact number and reliable
   evidence gives a materially different number,
   reduce the score.

7. If a high-quality source directly contradicts an
   important part of the claim, the score should
   normally be below 35.

8. If evidence is insufficient, do NOT invent facts.
   Use a score around 45-55.

9. Prefer high-quality sources such as:
   government websites, RBI, World Bank, IMF, WHO,
   UN, NASA, ISRO, Reuters, AP, BBC and established
   news organizations.

10. Search snippets can be incomplete. Do not claim
    certainty when the evidence is weak.

11. The score means:
    90-100 = strongly supported
    75-89  = mostly supported
    60-74  = somewhat supported
    40-59  = uncertain / mixed
    20-39  = mostly unsupported
    0-19   = strongly contradicted

12. Do not give 100 just because a search result looks
    similar. There must be strong evidence.

RETURN ONLY VALID JSON:

{{
  "score": 0,
  "verdict": "",
  "reason": "",
  "confidence": 0,
  "key_evidence": ""
}}

The score must be an integer from 0 to 100.

The verdict should be one of:
- Strongly supported
- Mostly supported
- Somewhat supported
- Uncertain / mixed
- Mostly unsupported
- Strongly contradicted

The reason should be short and explain the comparison.

Confidence must be from 0 to 100.

Do not include markdown.
"""

    raw = gemini_request(prompt)

    result = parse_gemini_json(raw)

    if not result:

        return {
            "score": 50,
            "verdict": "Uncertain / mixed",
            "reason": (
                "AI verification could not be completed."
            ),
            "confidence": 0,
            "key_evidence": ""
        }

    score_value = max(
        0,
        min(
            100,
            safe_int(
                result.get(
                    "score",
                    50
                )
            )
        )
    )

    confidence = max(
        0,
        min(
            100,
            safe_int(
                result.get(
                    "confidence",
                    50
                )
            )
        )
    )

    return {
        "score": score_value,
        "verdict": str(
            result.get(
                "verdict",
                "Uncertain / mixed"
            )
        ),
        "reason": str(
            result.get(
                "reason",
                ""
            )
        ),
        "confidence": confidence,
        "key_evidence": str(
            result.get(
                "key_evidence",
                ""
            )
        )
    }


# ============================================================
# FALLBACK SCORE
# ============================================================

def fallback_score(
    evidence
):

    real = [
        x for x in evidence
        if not x.get("is_demo")
    ]

    if not real:
        return 50

    # Quality based fallback
    quality_sum = sum(
        x.get("quality", 1)
        for x in real
    )

    max_quality = len(real) * 3

    ratio = (
        quality_sum / max_quality
        if max_quality
        else 0
    )

    return round(
        55 + ratio * 30
    )


# ============================================================
# VERDICT
# ============================================================

def verdict(score):

    if score >= 90:
        return "Strongly supported"

    if score >= 75:
        return "Mostly supported"

    if score >= 60:
        return "Somewhat supported"

    if score >= 40:
        return "Uncertain / mixed"

    if score >= 20:
        return "Mostly unsupported"

    return "Strongly contradicted"


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        os.path.join(
            FRONTEND,
            "index.html"
        )
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "serper_configured": bool(
            SERPER_API_KEY
        ),
        "gemini_configured": bool(
            GEMINI_API_KEY
        ),
        "gemini_model": GEMINI_MODEL,
        "last_gemini_error": LAST_GEMINI_ERROR
    }


# ============================================================
# MAIN ANALYZE API
# ============================================================

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):

    text = clean(req.text)

    source_title = ""

    # --------------------------------------------------------
    # URL INPUT
    # --------------------------------------------------------

    if req.url.strip():

        url = clean(req.url)

        if not url.startswith(
            ("http://", "https://")
        ):
            raise HTTPException(
                status_code=400,
                detail="Please enter a valid URL."
            )

        source_title, text = extract_article(
            url
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(text) < 20:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please provide a paragraph "
                "or article URL."
            )
        )

    # --------------------------------------------------------
    # CLAIMS
    # --------------------------------------------------------

    claims = claims_from(text)

    results = []
    all_evidence = []

    # --------------------------------------------------------
    # FACT CHECK EACH CLAIM
    # --------------------------------------------------------

    for claim in claims:

        evidence = collect_evidence(
            claim
        )

        all_evidence.extend(
            evidence
        )

        # Gemini evaluates claim
        ai_result = fact_check_claim(
            claim,
            evidence
        )

        # If Gemini failed, use fallback
        if (
            ai_result["confidence"] == 0
            and ai_result["score"] == 50
        ):

            fallback = fallback_score(
                evidence
            )

            ai_result["score"] = fallback

            ai_result["verdict"] = verdict(
                fallback
            )

            ai_result["reason"] = (
                "AI verification was unavailable; "
                "score estimated from available "
                "web-source quality."
            )

        results.append({

            "claim": claim,

            "support": ai_result["score"],

            "score": ai_result["score"],

            "verdict": ai_result["verdict"],

            "reason": ai_result["reason"],

            "confidence": ai_result[
                "confidence"
            ],

            "key_evidence": ai_result[
                "key_evidence"
            ],

            "evidence": evidence
        })

    # --------------------------------------------------------
    # ARTICLE LEVEL SCORE
    # --------------------------------------------------------

    if results:

        weighted_scores = []
        weights = []

        for item in results:

            score_value = item["score"]

            confidence = item["confidence"]

            # Confidence should influence the
            # article score, but never completely
            # dominate it.
            weight = max(
                0.5,
                confidence / 100
            )

            weighted_scores.append(
                score_value * weight
            )

            weights.append(weight)

        if sum(weights) > 0:

            article_score = round(
                sum(weighted_scores)
                / sum(weights)
            )

        else:

            article_score = round(
                sum(
                    x["score"]
                    for x in results
                ) / len(results)
            )

    else:

        article_score = 50

    article_score = max(
        0,
        min(
            100,
            article_score
        )
    )

    # --------------------------------------------------------
    # RELATED SOURCES
    # --------------------------------------------------------

    seen = set()
    related = []

    # Sort best quality first
    all_evidence.sort(
        key=lambda x: x.get(
            "quality",
            1
        ),
        reverse=True
    )

    for item in all_evidence:

        url = item.get(
            "url",
            ""
        )

        if (
            item.get("is_demo")
            or not url
            or url in seen
        ):
            continue

        seen.add(url)

        related.append({
            "title": item.get(
                "title",
                "Source"
            ),
            "snippet": item.get(
                "snippet",
                ""
            ),
            "url": url,
            "publisher": item.get(
                "publisher",
                ""
            ),
            "quality": item.get(
                "quality",
                1
            ),
            "is_demo": False
        })

        if len(related) >= 8:
            break

    # --------------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "truth_score": article_score,

        "verdict": verdict(
            article_score
        ),

        "source_title": source_title,

        "claims_analyzed": len(
            claims
        ),

        "claims": results,

        "related_information": related,

        "demo_mode": not bool(
            SERPER_API_KEY
        ),

        "ai_enabled": bool(
            GEMINI_API_KEY
        ),

        "disclaimer": (
            "This score is an evidence-based "
            "estimate, not a guarantee of truth."
        )
    }
