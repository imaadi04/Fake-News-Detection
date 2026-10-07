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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Current Gemini model
GEMINI_MODEL = "gemini-3.8-flash"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="TruthCheck",
    description="AI powered fake news and claim verification",
    version="3.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalyzeRequest(BaseModel):
    text: str = Field(
        default="",
        max_length=50000
    )

    url: str = Field(
        default="",
        max_length=3000
    )


# ============================================================
# DEBUG
# ============================================================

LAST_GEMINI_ERROR = ""


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(text):
    return re.sub(
        r"\s+",
        " ",
        text or ""
    ).strip()


def safe_int(value, default=50):

    try:
        return int(float(value))

    except Exception:
        return default


# ============================================================
# SERVE FRONTEND FILES
# ============================================================

@app.get("/")
def serve_home():

    path = os.path.join(
        BASE_DIR,
        "index.html"
    )

    if not os.path.exists(path):
        raise HTTPException(
            status_code=500,
            detail="index.html not found"
        )

    return FileResponse(path)


@app.get("/style.css")
def serve_css():

    path = os.path.join(
        BASE_DIR,
        "style.css"
    )

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="style.css not found"
        )

    return FileResponse(
        path,
        media_type="text/css"
    )


@app.get("/app.js")
def serve_js():

    path = os.path.join(
        BASE_DIR,
        "app.js"
    )

    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="app.js not found"
        )

    return FileResponse(
        path,
        media_type="application/javascript"
    )


# ============================================================
# ARTICLE URL EXTRACTOR
# ============================================================

def extract_article(url):

    try:

        response = requests.get(
            url,
            timeout=20,
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

        # Remove unnecessary elements
        for tag in soup([
            "script",
            "style",
            "noscript",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "svg"
        ]):

            tag.decompose()

        # Title
        title = (
            soup.title.get_text(
                " ",
                strip=True
            )
            if soup.title
            else url
        )

        paragraphs = []

        for p in soup.find_all("p"):

            paragraph = clean_text(
                p.get_text(
                    " ",
                    strip=True
                )
            )

            if len(paragraph) >= 30:
                paragraphs.append(paragraph)

        article = clean_text(
            " ".join(paragraphs)
        )

        # Fallback
        if len(article) < 100:

            article = clean_text(
                soup.get_text(
                    " ",
                    strip=True
                )
            )

        if len(article) < 50:

            raise ValueError(
                "Not enough article text could be extracted."
            )

        return title, article[:50000]

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read article URL: {str(e)}"
            )
        )


# ============================================================
# CLAIM SPLITTER
# ============================================================

def claims_from(text):

    text = clean_text(text)

    # Split sentences
    parts = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    claims = []

    for part in parts:

        part = clean_text(part)

        if 20 <= len(part) <= 1000:
            claims.append(part)

    # Fallback
    if not claims:

        claims = [
            text[:1000]
        ]

    # Limit
    return claims[:8]


# ============================================================
# SOURCE QUALITY
# ============================================================

VERY_HIGH_QUALITY = [
    ".gov",
    ".nic.in",
    "rbi.org.in",
    "worldbank.org",
    "imf.org",
    "who.int",
    "un.org",
    "nasa.gov",
    "isro.gov.in",
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
    "hindustantimes.com",
    "timesofindia.indiatimes.com",
    "economictimes.indiatimes.com",
    "bloomberg.com",
    "cnbc.com"
]


def source_quality(url):

    domain = urlparse(
        url
    ).netloc.lower()

    for domain_name in VERY_HIGH_QUALITY:

        if domain_name in domain:
            return 3

    for domain_name in HIGH_QUALITY:

        if domain_name in domain:
            return 2

    return 1


# ============================================================
# SERPER GOOGLE SEARCH
# ============================================================

def search_web(query):

    if not SERPER_API_KEY:

        print(
            "SERPER_API_KEY missing"
        )

        return []

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

            timeout=20
        )

        print(
            "SERPER STATUS:",
            response.status_code
        )

        if response.status_code != 200:

            print(
                "SERPER ERROR:",
                response.text[:2000]
            )

            return []

        data = response.json()

        results = []

        for item in data.get(
            "organic",
            []
        )[:8]:

            url = item.get(
                "link",
                ""
            ).strip()

            if not url:
                continue

            results.append({

                "title": item.get(
                    "title",
                    "Source"
                ),

                "snippet": clean_text(
                    item.get(
                        "snippet",
                        ""
                    )
                ),

                "url": url,

                "publisher": urlparse(
                    url
                ).netloc,

                "quality": source_quality(
                    url
                )
            })

        return results

    except Exception as e:

        print(
            "SERPER EXCEPTION:",
            repr(e)
        )

        return []


# ============================================================
# COLLECT WEB EVIDENCE
# ============================================================

def collect_evidence(claim):

    queries = [

        claim,

        f'"{claim}"',

        f"{claim} fact check",

        f"{claim} official source"

    ]

    evidence = []

    seen_urls = set()

    for query in queries:

        results = search_web(
            query
        )

        for result in results:

            url = result.get(
                "url",
                ""
            )

            if not url:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(url)

            evidence.append(
                result
            )

    # Highest quality first
    evidence.sort(
        key=lambda x: x.get(
            "quality",
            1
        ),
        reverse=True
    )

    return evidence[:12]


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

        "Content-Type":
            "application/json",

        "x-goog-api-key":
            GEMINI_API_KEY

    }

    payload = {

        "contents": [

            {

                "role": "user",

                "parts": [

                    {
                        "text": prompt
                    }

                ]

            }

        ],

        "generationConfig": {

            "temperature": 0.1,

            "responseMimeType":
                "application/json"

        }

    }

    try:

        response = requests.post(

            url,

            headers=headers,

            json=payload,

            timeout=45

        )

        print(
            "GEMINI STATUS:",
            response.status_code
        )

        # IMPORTANT:
        # Never silently hide API errors.
        if response.status_code != 200:

            print(
                "GEMINI ERROR RESPONSE:",
                response.text[:5000]
            )

            LAST_GEMINI_ERROR = (
                f"HTTP {response.status_code}: "
                f"{response.text[:1500]}"
            )

            return None

        data = response.json()

        candidates = data.get(
            "candidates",
            []
        )

        if not candidates:

            LAST_GEMINI_ERROR = (
                "No candidates returned."
            )

            print(
                "❌ No Gemini candidates"
            )

            return None

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        if not parts:

            LAST_GEMINI_ERROR = (
                "No Gemini response parts."
            )

            print(
                "❌ No Gemini parts"
            )

            return None

        generated_text = parts[0].get(
            "text",
            ""
        ).strip()

        print(
            "GEMINI RESPONSE:",
            generated_text[:3000]
        )

        if not generated_text:

            LAST_GEMINI_ERROR = (
                "Gemini returned empty text."
            )

            return None

        return generated_text

    except Exception as e:

        LAST_GEMINI_ERROR = repr(e)

        print(
            "❌ GEMINI EXCEPTION:",
            repr(e)
        )

        return None


# ============================================================
# PARSE GEMINI JSON
# ============================================================

def parse_json(text):

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences
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

        return json.loads(
            text
        )

    except Exception:

        # Extract JSON object
        start = text.find("{")
        end = text.rfind("}")

        if (
            start != -1
            and end != -1
            and end > start
        ):

            try:

                return json.loads(
                    text[start:end + 1]
                )

            except Exception:
                pass

    return None


# ============================================================
# FACT CHECK ONE CLAIM
# ============================================================

def fact_check_claim(
    claim,
    evidence
):

    evidence_lines = []

    for index, source in enumerate(
        evidence[:12],
        start=1
    ):

        evidence_lines.append(

            f"""
SOURCE {index}
Title: {source.get("title", "")}
Publisher: {source.get("publisher", "")}
URL: {source.get("url", "")}
Source Quality: {source.get("quality", 1)}/3
Snippet: {source.get("snippet", "")}
"""

        )

    evidence_block = "\n".join(
        evidence_lines
    )

    prompt = f"""
You are TruthCheck, an evidence-based
fact-checking AI.

USER CLAIM:
{claim}

WEB EVIDENCE:
{evidence_block}

Your job is to determine how strongly
the evidence supports the user's claim.

IMPORTANT:

1. Do NOT assume the claim is true.

2. Compare the actual meaning of the claim
   against the evidence.

3. Pay special attention to:
   - numbers
   - percentages
   - currency values
   - dates
   - years
   - units
   - names
   - locations
   - quantities
   - cause and effect

4. TIME-SENSITIVE FACTS:

For currency rates, stock prices,
weather, sports, current statistics,
current politicians, current policies,
etc., ALWAYS consider when the source
information was published.

A slightly outdated value should NOT
automatically be called false.

Example:

Claim:
"1 USD is approximately 95.98 INR."

Evidence:
"1 USD is approximately 96.40 INR."

If the claim has no date and the difference
is small, this is generally MOSTLY SUPPORTED,
not completely false.

5. For exact numerical claims:

Small difference:
usually still approximately correct.

Material difference:
reduce the score significantly.

6. If a reliable source directly contradicts
an important part of the claim, score below 35
unless there is a clear date/context difference.

7. If evidence is insufficient or ambiguous,
use approximately 45-55.

8. Prefer:
- Government sources
- RBI
- World Bank
- IMF
- WHO
- UN
- NASA
- ISRO
- Reuters
- AP
- BBC
- established newspapers

9. Search snippets can be incomplete.
Do not claim certainty if evidence is weak.

10. SCORE GUIDE:

90-100:
Strongly supported

75-89:
Mostly supported

60-74:
Somewhat supported

40-59:
Uncertain / mixed

20-39:
Mostly unsupported

0-19:
Strongly contradicted

11. Do not give 100 simply because
one result looks similar.

12. Explain WHY the evidence supports
or contradicts the claim.

RETURN ONLY JSON.

Required JSON:

{{
    "score": 0,
    "verdict": "",
    "reason": "",
    "confidence": 0,
    "key_evidence": ""
}}

Allowed verdict values:

Strongly supported
Mostly supported
Somewhat supported
Uncertain / mixed
Mostly unsupported
Strongly contradicted

score must be integer 0-100.

confidence must be integer 0-100.

No markdown.
"""

    raw = gemini_request(
        prompt
    )

    parsed = parse_json(
        raw
    )

    if not parsed:

        return {

            "score": 50,

            "verdict":
                "Uncertain / mixed",

            "reason":
                "AI verification failed.",

            "confidence": 0,

            "key_evidence": ""

        }

    score = max(
        0,
        min(
            100,
            safe_int(
                parsed.get(
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
                parsed.get(
                    "confidence",
                    50
                )
            )
        )
    )

    return {

        "score": score,

        "verdict": str(
            parsed.get(
                "verdict",
                "Uncertain / mixed"
            )
        ),

        "reason": str(
            parsed.get(
                "reason",
                ""
            )
        ),

        "confidence": confidence,

        "key_evidence": str(
            parsed.get(
                "key_evidence",
                ""
            )
        )

    }


# ============================================================
# VERDICT
# ============================================================

def get_verdict(score):

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
# ANALYZE API
# ============================================================

@app.post("/api/analyze")
def analyze(
    request: AnalyzeRequest
):

    text = clean_text(
        request.text
    )

    article_title = ""

    # --------------------------------------------------------
    # URL INPUT
    # --------------------------------------------------------

    if request.url.strip():

        url = clean_text(
            request.url
        )

        if not url.startswith(
            (
                "http://",
                "https://"
            )
        ):

            raise HTTPException(
                status_code=400,
                detail="Invalid article URL."
            )

        article_title, text = (
            extract_article(url)
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if len(text) < 20:

        raise HTTPException(

            status_code=400,

            detail=(
                "Please enter a news claim, "
                "paragraph or article URL."
            )

        )

    # --------------------------------------------------------
    # CLAIMS
    # --------------------------------------------------------

    claims = claims_from(
        text
    )

    claim_results = []

    all_sources = []

    # --------------------------------------------------------
    # CHECK EACH CLAIM
    # --------------------------------------------------------

    for claim in claims:

        evidence = collect_evidence(
            claim
        )

        all_sources.extend(
            evidence
        )

        result = fact_check_claim(
            claim,
            evidence
        )

        claim_results.append({

            "claim":
                claim,

            "score":
                result["score"],

            "support":
                result["score"],

            "verdict":
                result["verdict"],

            "reason":
                result["reason"],

            "confidence":
                result["confidence"],

            "key_evidence":
                result["key_evidence"],

            "evidence":
                evidence

        })

    # --------------------------------------------------------
    # ARTICLE SCORE
    # --------------------------------------------------------

    if claim_results:

        weighted_total = 0
        total_weight = 0

        for result in claim_results:

            score = result["score"]

            confidence = result["confidence"]

            # Confidence has influence but
            # cannot completely dominate.
            weight = max(
                0.5,
                confidence / 100
            )

            weighted_total += (
                score * weight
            )

            total_weight += weight

        if total_weight:

            truth_score = round(
                weighted_total /
                total_weight
            )

        else:

            truth_score = round(
                sum(
                    x["score"]
                    for x in claim_results
                )
                / len(claim_results)
            )

    else:

        truth_score = 50

    truth_score = max(
        0,
        min(
            100,
            truth_score
        )
    )

    # --------------------------------------------------------
    # UNIQUE SOURCES
    # --------------------------------------------------------

    unique_sources = []

    seen = set()

    # Best source quality first
    all_sources.sort(
        key=lambda x: x.get(
            "quality",
            1
        ),
        reverse=True
    )

    for source in all_sources:

        url = source.get(
            "url",
            ""
        )

        if not url:
            continue

        if url in seen:
            continue

        seen.add(url)

        unique_sources.append({

            "title":
                source.get(
                    "title",
                    "Source"
                ),

            "snippet":
                source.get(
                    "snippet",
                    ""
                ),

            "url":
                url,

            "publisher":
                source.get(
                    "publisher",
                    ""
                ),

            "quality":
                source.get(
                    "quality",
                    1
                )

        })

        if len(unique_sources) >= 10:
            break

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "truth_score":
            truth_score,

        "verdict":
            get_verdict(
                truth_score
            ),

        "source_title":
            article_title,

        "claims_analyzed":
            len(claim_results),

        "claims":
            claim_results,

        "related_information":
            unique_sources,

        "ai_enabled":
            bool(GEMINI_API_KEY),

        "live_search_enabled":
            bool(SERPER_API_KEY),

        "disclaimer":
            (
                "TruthCheck provides an "
                "evidence-based estimate. "
                "It is not a guarantee of truth."
            )

    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "serper_configured":
            bool(SERPER_API_KEY),

        "gemini_configured":
            bool(GEMINI_API_KEY),

        "gemini_model":
            GEMINI_MODEL,

        "last_gemini_error":
            LAST_GEMINI_ERROR

    }
