import os
import re
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

app = FastAPI(title="TruthCheck API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class AnalyzeRequest(BaseModel):
    text: str = Field("", max_length=50000)
    url: str = Field("", max_length=2000)


def clean(s):
    return re.sub(r"\s+", " ", s or "").strip()


def extract_article(url):
    try:
        r = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": "Mozilla/5.0 TruthCheck/2.0"},
        )

        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else url

        body = clean(
            " ".join(
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
            )
        )

        if len(body) < 80:
            body = clean(soup.get_text(" ", strip=True))

        return title, body[:50000]

    except Exception as e:
        raise HTTPException(400, f"Could not read article URL: {e}")


def claims_from(text):
    parts = re.split(r"(?<=[.!?])\s+", clean(text))

    claims = [
        x.strip()
        for x in parts
        if 35 <= len(x.strip()) <= 600
    ]

    return (claims or [text[:600]])[:8]


# ---------------------------------------------------------
# SOURCE RELIABILITY
# ---------------------------------------------------------

def source_reliability(url):
    domain = urlparse(url).netloc.lower()

    trusted = [
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
        "ec.europa.eu",
        "oecd.org",
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
    ]

    for source in trusted:
        if source in domain:
            return 1.0

    news = [
        "thehindu.com",
        "indianexpress.com",
        "ndtv.com",
        "timesofindia.indiatimes.com",
        "hindustantimes.com",
        "economictimes.indiatimes.com",
        "livemint.com",
    ]

    for source in news:
        if source in domain:
            return 0.8

    return 0.55


# ---------------------------------------------------------
# NUMBER EXTRACTION
# ---------------------------------------------------------

def normalize_number(value):
    try:
        return float(value.replace(",", ""))
    except:
        return None


def extract_numbers(text):
    pattern = r"(?<!\w)(\d+(?:,\d{3})*(?:\.\d+)?)(?:\s*)?(%|percent|percentage)?"

    numbers = []

    for match in re.finditer(pattern, text.lower()):
        number = normalize_number(match.group(1))

        if number is None:
            continue

        unit = match.group(2) or ""

        numbers.append({
            "value": number,
            "unit": unit,
            "start": match.start(),
            "end": match.end()
        })

    return numbers


def extract_years(text):
    return [
        int(x)
        for x in re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    ]


# ---------------------------------------------------------
# CLAIM VS EVIDENCE
# ---------------------------------------------------------

def token_set(text):
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())

    stopwords = {
        "the", "and", "was", "were", "has", "have",
        "had", "for", "with", "that", "this", "from",
        "into", "than", "about", "which", "their",
        "there", "they", "them", "been", "also",
        "are", "its", "but", "not", "you", "your",
        "india", "according"
    }

    return set(x for x in words if x not in stopwords)


def text_overlap(claim, evidence):
    a = token_set(claim)
    b = token_set(evidence)

    if not a or not b:
        return 0

    return len(a & b) / len(a)


def numeric_check(claim, evidence):
    """
    Returns:
        +1  -> numerical evidence supports claim
        -1  -> numerical evidence contradicts claim
         0  -> inconclusive
    """

    claim_numbers = extract_numbers(claim)

    if not claim_numbers:
        return 0

    evidence_numbers = extract_numbers(evidence)

    if not evidence_numbers:
        return 0

    claim_years = extract_years(claim)
    evidence_years = extract_years(evidence)

    # If claim contains a year, evidence should ideally contain same year
    same_year = False

    if claim_years:
        same_year = any(
            year in evidence_years
            for year in claim_years
        )

    # Compare numbers
    for cn in claim_numbers:

        # Ignore years
        if 1900 <= cn["value"] <= 2100:
            continue

        for en in evidence_numbers:

            if 1900 <= en["value"] <= 2100:
                continue

            # Same number -> supporting evidence
            if abs(cn["value"] - en["value"]) < 0.01:
                return 1

    # Different number + same year -> contradiction
    if same_year:

        for cn in claim_numbers:

            if 1900 <= cn["value"] <= 2100:
                continue

            for en in evidence_numbers:

                if 1900 <= en["value"] <= 2100:
                    continue

                # Different meaningful number
                difference = abs(cn["value"] - en["value"])

                if difference > max(0.05, abs(cn["value"]) * 0.05):
                    return -1

    return 0


# ---------------------------------------------------------
# SEARCH
# ---------------------------------------------------------

def search(claim):

    if not SERPER_API_KEY:
        return [{
            "title": "Live search not configured",
            "snippet": "Add SERPER_API_KEY to retrieve current web sources.",
            "url": "https://serper.dev/",
            "publisher": "TruthCheck",
            "is_demo": True
        }]

    try:

        # Search original claim
        queries = [
            claim,
            claim + " fact check"
        ]

        results = []

        for query in queries:

            r = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": SERPER_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "q": query,
                    "num": 5
                },
                timeout=12
            )

            r.raise_for_status()

            for x in r.json().get("organic", []):

                link = x.get("link", "")

                if not link:
                    continue

                results.append({
                    "title": x.get("title", "Untitled"),
                    "snippet": x.get("snippet", ""),
                    "url": link,
                    "publisher": urlparse(link).netloc,
                    "is_demo": False
                })

        # Remove duplicates
        unique = []
        seen = set()

        for item in results:

            if item["url"] in seen:
                continue

            seen.add(item["url"])
            unique.append(item)

        return unique[:8] or search_demo()

    except Exception:
        return search_demo()


def search_demo():

    return [{
        "title": "No live evidence available",
        "snippet": "The search provider could not return live evidence.",
        "url": "https://serper.dev/",
        "publisher": "TruthCheck",
        "is_demo": True
    }]


# ---------------------------------------------------------
# EVIDENCE ANALYSIS
# ---------------------------------------------------------

def analyze_evidence(claim, evidence):

    if evidence.get("is_demo"):
        return {
            "status": "unknown",
            "score": 0
        }

    combined_text = clean(
        evidence.get("title", "") + " " +
        evidence.get("snippet", "")
    )

    overlap = text_overlap(claim, combined_text)

    number_result = numeric_check(claim, combined_text)

    reliability = source_reliability(evidence["url"])

    # Strong numerical contradiction
    if number_result == -1 and overlap >= 0.25:
        return {
            "status": "contradicts",
            "score": -45 * reliability
        }

    # Numerical support
    if number_result == 1 and overlap >= 0.20:
        return {
            "status": "supports",
            "score": 35 * reliability
        }

    # General textual support
    if overlap >= 0.55:
        return {
            "status": "supports",
            "score": 25 * reliability
        }

    if overlap >= 0.30:
        return {
            "status": "related",
            "score": 8 * reliability
        }

    return {
        "status": "unknown",
        "score": 0
    }


# ---------------------------------------------------------
# CLAIM SCORE
# ---------------------------------------------------------

def calculate_claim_score(claim, evidence):

    if not evidence:
        return 50, "Insufficient evidence"

    scores = []

    support_count = 0
    contradiction_count = 0

    for item in evidence:

        analysis = analyze_evidence(claim, item)

        scores.append(analysis["score"])

        if analysis["status"] == "supports":
            support_count += 1

        elif analysis["status"] == "contradicts":
            contradiction_count += 1

    if not scores:
        return 50, "Insufficient evidence"

    # Start from neutral
    score = 50

    positive = sum(x for x in scores if x > 0)
    negative = sum(x for x in scores if x < 0)

    # Add supporting evidence
    score += min(35, positive)

    # Subtract contradictory evidence
    score += max(-45, negative)

    score = max(5, min(95, round(score)))

    if contradiction_count > support_count:
        verdict = "Likely false / contradicted"

    elif support_count > contradiction_count:
        verdict = "Mostly supported"

    else:
        verdict = "Mixed / needs verification"

    return score, verdict


# ---------------------------------------------------------
# OVERALL SCORE
# ---------------------------------------------------------

def overall_score(claim_results):

    if not claim_results:
        return 50

    scores = [
        x["support"]
        for x in claim_results
    ]

    return round(sum(scores) / len(scores))


def verdict(score):

    if score >= 85:
        return "Strongly supported"

    if score >= 70:
        return "Mostly supported"

    if score >= 55:
        return "Mixed / needs verification"

    if score >= 35:
        return "Likely false / weak evidence"

    return "Strongly contradicted"


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

@app.get("/")
def home():
    return FileResponse(
        os.path.join(FRONTEND, "index.html")
    )


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):

    text = clean(req.text)
    source_title = ""

    # URL input
    if req.url.strip():

        url = clean(req.url)

        if not url.startswith(("http://", "https://")):
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

    claims = claims_from(text)

    all_evidence = []
    results = []

    for claim in claims:

        evidence = search(claim)

        all_evidence.extend(evidence)

        claim_score, claim_verdict = calculate_claim_score(
            claim,
            evidence
        )

        results.append({
            "claim": claim,
            "support": claim_score,
            "verdict": claim_verdict,
            "evidence": evidence
        })

    # Related information
    seen = set()
    related = []

    for evidence in all_evidence:

        if evidence.get("is_demo"):
            continue

        url = evidence.get("url", "")

        if not url or url in seen:
            continue

        seen.add(url)

        related.append(evidence)

        if len(related) >= 8:
            break

    final_score = overall_score(results)

    return {
        "truth_score": final_score,
        "verdict": verdict(final_score),
        "source_title": source_title,
        "claims_analyzed": len(claims),
        "claims": results,
        "related_information": related,
        "demo_mode": not bool(SERPER_API_KEY),
        "disclaimer": (
            "This score is an evidence-based estimate, "
            "not a guarantee of truth."
        )
    }
