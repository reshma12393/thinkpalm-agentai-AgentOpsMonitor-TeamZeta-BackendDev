# How to run — AutoML Debate System

Quick reference to start the backend API and the React UI. For architecture and API shapes, see [README.md](./README.md).

**Paths:** Run all `cd` commands from the **repository root** (the directory that contains `src/`, `requirements.txt`, and this file). Code lives in **`src/backend`** and **`src/frontend`** only.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| Python | 3.11+ recommended |
| Node.js | 18+ (Vite 6) |
| OpenRouter API key | Optional — enables LLM parts of EDA, debate, judge via [OpenRouter](https://openrouter.ai/); heuristics work without it |

---

## 1. Backend (FastAPI)

```bash
cd src/backend
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (cmd / PowerShell)
pip install -r ../../requirements.txt
```

Optional: copy **`src/backend/.env.example`** to **`src/backend/.env`** and set OpenRouter variables ([API keys](https://openrouter.ai/keys)):

```env
OPENROUTER_API_KEY=your-key-here
# OPENROUTER_MODEL=anthropic/claude-haiku-4.5
```

Start the server from **`src/backend/`** so `.env` and relative `data/` paths match `app.config`. Restart after edits.

**`PYTHONPATH=.`** is required: the importable package is `app` (`src/backend/app`). Without it, Uvicorn will fail with `ModuleNotFoundError: No module named 'app'`.

Start the server:

```bash
cd src/backend
source .venv/bin/activate
PYTHONPATH=. python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Use **`python -m uvicorn`** (with the venv active) instead of the bare **`uvicorn`** command. After moving the project or the `.venv` folder, console scripts in the venv can still point at an old Python path; `python -m uvicorn` always uses the interpreter you activated.

Verify:

```bash
curl -s http://127.0.0.1:8000/health
# Expected: {"status":"ok"}
```

API base URL for local use: `http://127.0.0.1:8000`.

---

## 2. Frontend (React + Vite)

In a **second terminal**, from the **same repo root**:

```bash
cd src/frontend
npm install
npm run dev
```

Do **not** use a top-level `frontend/` folder — the React app is only under **`src/frontend`**. Running Vite from the wrong directory may create an empty stray `frontend/` at the root (cache); always `cd src/frontend` first.

Default UI: **http://localhost:5173**

The dev server proxies API calls to the backend (`/automl-debate`, `/api`, `/health` → port 8000). Keep the backend running first.

---

## 3. Run a job from the UI

1. Open the app in the browser.
2. Choose a **CSV** file.
3. Enter the **target column** name (exact column header).
4. Click **Run AutoML debate** and wait until training finishes (CPU time depends on data size).

---

## 4. Run a job from the terminal (sync API)

```bash
curl -s -X POST "http://127.0.0.1:8000/automl-debate" \
  -F "file=@/path/to/your.csv" \
  -F "target_column=your_label_column" \
  | head -c 2000
```

Inspect `winner` and `metrics` in the JSON response.

---

## 5. Async API (optional)

Start a background run:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/v1/debate" \
  -F "file=@/path/to/your.csv" \
  -F "target_column=your_label_column"
```

Note the `run_id`, then poll:

```bash
curl -s "http://127.0.0.1:8000/api/v1/debate/<run_id>"
```

Stop when `status` is `completed` or `failed`; full payload is under `result`.

---

## 6. Production build (frontend only)

```bash
cd src/frontend
npm run build
npm run preview
```

Serve `src/frontend/dist/` behind any static host; configure that host to **reverse-proxy** `/automl-debate`, `/api`, and `/health` to your FastAPI process, or set `axios` `baseURL` to the API origin.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `ModuleNotFoundError: No module named 'app'` | Run Uvicorn from **`src/backend`** with **`PYTHONPATH=.`** (see §1). Do not start it from the repo root without adjusting `PYTHONPATH`. |
| `ModuleNotFoundError: No module named 'pandas'` (or other deps) | Activate **`src/backend/.venv`** (`source .venv/bin/activate`), then run **`pip install -r ../../requirements.txt`**. If the repo (or `.venv`) was moved, recreate the venv: `cd src/backend && rm -rf .venv && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r ../../requirements.txt`. Always start the API with the venv’s **`python`** (e.g. **`python -m uvicorn`**). |
| UI says backend unreachable | Confirm `uvicorn` is on port **8000**; check firewall; try `curl http://127.0.0.1:8000/health`. |
| Chroma / SQLite errors on import | Backend uses `app/sqlite_patch.py` and may require `pysqlite3-binary` (see `requirements.txt`). |
| CORS errors from a custom origin | Add your origin to `cors_origins` in `src/backend/app/config.py` or set via env if you extend settings. |
| Out of memory on large CSV | Use a smaller sample or increase system RAM; training is in-process. |
| Empty `frontend/` folder at repo root | Safe to delete; it is not the real app. Use **`src/frontend`**; root `frontend/` is git-ignored if recreated by a mis-placed Vite run. |

---

## Ports summary

| Service | Port |
|---------|------|
| FastAPI (default) | 8000 |
| Vite dev | 5173 |
