import os, re
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
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(BASE, "frontend")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()

app = FastAPI(title="TruthCheck API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND, "static")), name="static")

class AnalyzeRequest(BaseModel):
    text: str = Field("", max_length=50000)
    url: str = Field("", max_length=2000)

def clean(s): return re.sub(r"\s+", " ", s or "").strip()

def extract_article(url):
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0 TruthCheck/1.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript","nav","footer","header"]): tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        body = clean(" ".join(p.get_text(" ", strip=True) for p in soup.find_all("p")))
        if len(body) < 80: body = clean(soup.get_text(" ", strip=True))
        return title, body[:50000]
    except Exception as e:
        raise HTTPException(400, f"Could not read article URL: {e}")

def claims_from(text):
    parts = re.split(r"(?<=[.!?])\s+", clean(text))
    claims = [x.strip() for x in parts if 35 <= len(x.strip()) <= 600]
    return (claims or [text[:600]])[:8]

def search(claim):
    if not SERPER_API_KEY:
        return [{
            "title":"Live search not configured",
            "snippet":"Add SERPER_API_KEY to .env to retrieve current web sources for this claim.",
            "url":"https://serper.dev/",
            "publisher":"TruthCheck",
            "is_demo":True
        }]
    try:
        r = requests.post("https://google.serper.dev/search",
            headers={"X-API-KEY":SERPER_API_KEY,"Content-Type":"application/json"},
            json={"q":claim,"num":5}, timeout=12)
        r.raise_for_status()
        out=[]
        for x in r.json().get("organic",[])[:5]:
            link=x.get("link","")
            out.append({"title":x.get("title","Untitled"),
                         "snippet":x.get("snippet",""),"url":link,
                         "publisher":urlparse(link).netloc,"is_demo":False})
        return out or search_demo()
    except Exception:
        return search_demo()

def search_demo():
    return [{"title":"No live evidence available",
             "snippet":"The search provider could not return live evidence.",
             "url":"https://serper.dev/","publisher":"TruthCheck","is_demo":True}]

def score(claims, evidence):
    real=[e for e in evidence if not e["is_demo"]]
    if not real: return 50
    coverage=min(1,len(real)/max(1,len(claims)*2))
    return round(50+coverage*35)

def verdict(s):
    if s>=85:return "Strongly supported"
    if s>=70:return "Mostly supported"
    if s>=55:return "Mixed / needs verification"
    if s>=40:return "Weakly supported"
    return "Poorly supported"

@app.get("/")
def home(): return FileResponse(os.path.join(FRONTEND,"index.html"))

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    text=clean(req.text); source_title=""
    if req.url.strip():
        url=clean(req.url)
        if not url.startswith(("http://","https://")): raise HTTPException(400,"Please enter a valid URL.")
        source_title,text=extract_article(url)
    if len(text)<20: raise HTTPException(400,"Please provide a paragraph or article URL.")

    claims=claims_from(text); all_evidence=[]; results=[]
    for claim in claims:
        ev=search(claim); all_evidence.extend(ev)
        results.append({"claim":claim,
                        "support":50 if all(e["is_demo"] for e in ev) else min(90,55+7*len(ev)),
                        "evidence":ev})
    seen=set(); related=[]
    for e in all_evidence:
        if e["is_demo"] or not e["url"] or e["url"] in seen: continue
        seen.add(e["url"]); related.append(e)
        if len(related)>=8: break

    return {"truth_score":score(claims,all_evidence),
            "verdict":verdict(score(claims,all_evidence)),
            "source_title":source_title,"claims_analyzed":len(claims),
            "claims":results,"related_information":related,
            "demo_mode":not bool(SERPER_API_KEY),
            "disclaimer":"This score is an evidence-based estimate, not a guarantee of truth."}
