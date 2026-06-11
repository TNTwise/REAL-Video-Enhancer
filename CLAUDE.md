# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REAL Video Enhancer applies AI super-resolution models (Real-ESRGAN, SwinIR, SRVGG, etc.) to upscale and enhance video resolution. The project is migrating from a PySide6 desktop app (v1) to a Tauri + React desktop app (v2). The current branch `redesign-ui` is the v2 migration work; PRs target `v2-main`.

## Architecture

### Three-Tier Structure

```
┌─────────────────────────────────────────────────────────┐
│                   Tauri Window (Rust)                    │
│  ┌─────────────────────────────────────────────────────┐│
│  │              React Frontend (client/)                ││
│  │  - Pages: HomePage, DownloadPage                     ││
│  │  - UI: Chakra components (DefaultButton, etc.)       ││
│  │  - Theme: dark/light mode support                    ││
│  └────────────────────┬────────────────────────────────┘│
│                       │ HTTP / Tauri IPC                 │
│  ┌────────────────────▼────────────────────────────────┐│
│  │           Python Backend (backend/)                  ││
│  │  - FastAPI server (routers started, no entrypoint)   ││
│  │  - AI backends: NCNN (default), PyTorch, ONNX        ││
│  │  - Render pipeline: extract → enhance → encode       ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### Backend (`backend/`)
- **AI backends**: `ncnn/` (fast CPU, default), `pytorch/` (GPU via Spandrel), `onnx/` (alternative)
- **Render pipeline** (`render/`): `RenderVideo.py` orchestrates FFmpeg frame extraction → AI enhancement per frame → video re-encoding. `InformationWriteout.py` logs progress. `FFmpegWrite.py` handles encoding.
- **Model repo** (`model/`): Discovers model definition JSON, validates weights, configures backend loaders
- **Settings** (`settings/`): Python dataclasses for app config; v2 migrates to FastAPI endpoints
- **Routers**: `render_router.py` and `settings_router.py` exist but no `server.py` entrypoint yet

### Frontend (`client/`)
- React + TypeScript + Vite, built with Chakra UI components
- Routing via `BrowserRouter`, layout in `RootLayout.tsx`
- **Missing**: API client layer, state management, Tauri IPC bridge, backend process launcher

### Tauri Shell (`src-tauri/`)
- Rust app wrapping React frontend; basic config exists, production build not yet working

## Common Commands

### Frontend (run from `client/`)
```bash
npm install              # Install dependencies
npm run dev              # Vite dev server
npm run build            # Production build
npm run lint             # ESLint
```

### Backend / Project Root
```bash
bash scripts/format_lint.sh   # Format and lint Python code
```

### Tauri (from project root)
```bash
cargo tauri dev              # Dev mode (requires Tauri CLI)
cargo tauri build            # Production build
```

### Legacy v1 Build (PySide — reference only)
```bash
python build.py --build cx_freeze --copy_backend
python build.py --build pyinstaller --copy_backend
```

## Key Files
- `TODO.md` — v2 migration task list
- `src-tauri/tauri.conf.json` — Tauri app configuration
- `client/package.json` — frontend scripts
- `.github/workflows/build-prerelease.yml` — cross-platform CI build (v1 reference)

## Git
- **Main branch**: `v2-main` (not `main`)
- **Current work**: `redesign-ui` branch
- PRs always target `v2-main`

## Memory
Project context and decisions are stored in `.claude/memory/`. See `MEMORY.md` for index.
