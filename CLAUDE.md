# REAL Video Enhancer (RVE)

## Project Summary

REAL Voice Enhancer is a cross-platform AI video enhancement desktop application for upscaling and interpolating videos. It supports multiple AI backends (NCNN, PyTorch, TensorRT) and runs on Windows, macOS, and Linux with CPU or GPU acceleration. The project is undergoing a major UI migration from PySide6 to a Tauri + React frontend.

---

## Development Environment

- **OS:** Linux (Arch), Windows, macOS
- **Python:** 3.12+ (main app), 3.11 for Linux portable builds
- **Frontend:** TypeScript, React 19, Vite 7 (in `client/`)
- **Package Manager:** npm (frontend), pip/uv (Python backend)

### Getting Started

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd client && npm install
```

---

## Architecture

### Dual-UI Architecture (Transition Phase)

The project maintains TWO UIs simultaneously during migration:

| Component | Location | Status |
|-----------|----------|--------|
| **Legacy UI** | `REAL-Video-Enhancer.py` + `mainwindow.py` + `src/` | PySide6, production |
| **New UI** | `client/` | Tauri + React + TypeScript + Chakra UI v3, in development |
| **Backend API** | `backend/rve-backend.py` | Python backend service |

### Legacy UI (PySide6) – `REAL-Video-Enhancer.py`

Entry point: `REAL-Video-Enhancer.py` → loads `mainwindow.py` (Qt generated UI from `testRVEInterface.ui`)

#### Key Source Files (`src/`)

| File | Purpose |
|------|---------|
| `ModelHandler.py` | AI model registry (NCNN, PyTorch, TensorRT for upscale/interpolate) |
| `GenerateFFMpegCommand.py` | FFmpeg command builder for video processing pipelines |
| `DownloadDeps.py` | Dependency downloader (Python runtime, FFmpeg, etc.) |
| `DownloadModels.py` | AI model download from GitHub releases |
| `PresetManager.py` | Preset configuration management |
| `VideoInfo.py` | Video metadata extraction |
| `Util.py` | Logging, file ops, OS detection, disk/RAM utilities |
| `constants.py` | Global paths, platform detection, config |
| `version.py` | App version string |
| `ui/ProcessTab.py` | Per-processing-tab widget class |

### New UI (Tauri + React) – `client/`

- **Stack:** Tauri 2 (Rust backend) + Vite + React 19 + TypeScript + Chakra UI v3 + react-router-dom
- **Theme:** next-themes for dark/light mode support
- **Structure:** Standard Vite+React layout in `client/src/`, Tauri config in `client/src-tauri/`

### Backend API – `backend/rve-backend.py`

Python backend service for the new UI. Handles settings storage and AI processing orchestration. Uses FastAPI (planned per TODO.md).

---

## Build System

### Build Script – `build.py`

Primary build tool. Supports two modes:
- **PyInstaller** (`--build pyinstaller`) — Windows/macOS builds
- **cx_Freeze** (`--build cx_freeze`) — Linux portable builds

Flags: `--copy_backend` includes the backend submodule. Downloads embedded Python runtime and Qt dependencies for portable bundles.

### Frontend Build

```bash
cd client
npm run build      # TypeScript + Vite production build
npm run dev        # Vite dev server
npm run tauri build  # Full Tauri desktop bundle
```

### CI/CD

Workflow: `.github/workflows/prerelease.yml` — builds all platforms on PRs to `main` or manual dispatch on `dev`:
- **Linux x86_64/arm64:** cx_Freeze in distrobox (Ubuntu 20.04)
- **Windows x86_64:** PyInstaller + NSIS installer
- **macOS x86_64/arm64:** PyInstaller on macOS runners
- Produces portable zips, Windows installer exe, and backend tarball

---

## Data Flow

1. User selects video → `VideoInfo.py` extracts metadata (resolution, FPS, codec)
2. User chooses enhancement model from `ModelHandler.py` registry
3. Processing tab (`ProcessTab.py`) manages per-tab state and progress
4. `GenerateFFMpprogCommand.py` constructs FFmpeg pipeline: extract frames → run AI model → reassemble video
5. Settings persisted to `settings.txt` (legacy) / backend API (new UI)

---

## Key Directories

| Path | Contents |
|------|----------|
| `icons/` | Application icon assets (SVG, PNG for Qt resources) |
| `models/` | Downloaded AI model weights |
| `presets/` | Enhancement preset configurations |
| `custom_models/` | User-provided custom models |
| `scripts/` | Release/changelog utilities |
| `bin/` | Build output (untracked) |

---

## Current Goals (TODO.md)

1. **Migrate UI to TypeScript/Tauri** – get usable Tauri build running
2. **Store settings via backend** – move from frontend-local storage to backend API
3. **Integrate FastAPI** – into the Python backend
4. **Connect UI with backend** – full end-to-end integration
5. **Bundle all dependencies** – package FFmpeg and Python runtime with the app

---

## Working Notes

- Main branch for PRs: `v2-main` (current dev work on `redesign-ui` branch)
- Settings file: `settings.txt` at project root
- Models download from: `https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/`
- The `backend/` submodule contains the AI processing library (pytorch/spandrel)
