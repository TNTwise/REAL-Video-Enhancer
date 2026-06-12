# AGENTS.md

## Branches
- **Main branch**: `v2-main` (not `main`). All PRs target `v2-main`.
- Current work happens on feature branches like `redesign-ui`.

## Architecture — Three-Tier App
Project is migrating from a PySide6 desktop app (v1) to Tauri + React (v2).

| Layer | Location | Tech |
|---|---|---|
| Frontend | `client/src/` | React 19 + TypeScript + Vite + Chakra UI 3 + react-router-dom 7 |
| Desktop shell | `client/src-tauri/` | Tauri 2 (Rust) |
| Backend API | `backend/` | FastAPI + uvicorn, Python ≥3.12 |

Backend serves HTTP on port 8000 with CORS wide open (`allow_origins=["*"]`). Frontend communicates via HTTP, not Tauri IPC. **There is no API client layer or state management in the frontend yet.**

### Backend entry point
`backend/main.py` — FastAPI app with routers for settings, render, and video_info. Run with `python main.py` (uvicorn on 0.0.0.0:8000).

### Frontend entry point
`client/src/main.tsx` → `App.tsx` → `RootLayout.tsx` with pages in `client/src/pages/`.

### Tauri config
Dev server runs at `http://localhost:1420`. Frontend dist lands in `client/dist/`.

## Commands

### Frontend (`cd client`)
```bash
npm install          # dependencies
npm run dev          # Vite dev server (port 1420)
npm run build        # tsc + vite build → ./dist
npm run preview      # serve production build locally
```
Note: `npm run lint` does **not** exist. ESLint config is at `client/eslint.config.js` but only covers `.js/.jsx`, not TS/TSX.

### Python backend (from repo root)
```bash
./scripts/format_lint.sh        # check mode: isort + ruff format + ruff check + ty
./scripts/format_lint.sh fix    # fix mode: auto-correct imports, format, and lint issues
```
The lint script uses `find` to locate all `.py` files except those under `client/`, `.venv/`, `venv/`, `__pycache__/`, `bin/`. It runs isort (black profile), ruff format, ruff check, and `ty` type checker in that order.

### Tauri (from repo root)
```bash
cargo tauri dev               # dev mode (runs npm run dev inside client/)
cargo tauri build             # production build
```

## Python Setup
- Managed with `uv` (lockfile: `backend/uv.lock`). Install deps via `uv sync` in `backend/`.
- `backend/requirements.txt` lists all deps including heavy ones (torch, TensorRT, ncnn). **Do not blindly install everything** — backends are installed on demand.
- v2 goal: bundle FFmpeg and Python runtime inside the app; per-backend AI libraries install via API.

## TypeScript Path Alias
`@/*` maps to `client/src/*` (configured in `tsconfig.json` and Vite's `vite-tsconfig-paths`).

## Missing Pieces (per TODO.md)
- Tauri production build not yet working
- No backend process launcher from frontend
- Settings storage should move to backend API
- Model management via local JSON with downloadable flag

## Extra Context Files
- `.claude/memory/` — project state, frontend structure, backend structure notes
- `TODO.md` — v2 migration checklist
