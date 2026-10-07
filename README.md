# TruthCheck

Simple end-to-end fake-news / claim-verification MVP.

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn backend.main:app --reload
```
Open http://127.0.0.1:8000

Add `SERPER_API_KEY` in `.env` for live web evidence. Without it, the app runs in clearly labelled demo mode.

The truth percentage is an evidence-based estimate, not a guarantee of truth.
