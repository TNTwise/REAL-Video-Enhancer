# Migrate UI to Tauri + TypeScript

## Core Tauri Setup
- [ ] Get a working Tauri build running (dev + production)

## Settings Management
- [ ] Move all settings storage to the backend API (replace frontend-only persistence)
- [ ] Load settings through the built-in settings module via API requests instead of command-line arguments

## Backend Service
- [ ] Integrate FastAPI into the Python backend
- [ ] Add per-backend installation via API — each AI backend (NCNN, PyTorch, TensorRT) exposes its Python dependencies as a list the frontend can install on demand
- [ ] Have the client discover an available port and launch the backend API on start-up

## End-to-End Integration
- [ ] Connect the Tauri frontend to the backend API so enhancement workflows run end-to-end


# Bundle Base Dependencies

Package FFmpeg and the Python runtime inside the app so it runs without system-level prerequisites. Per-backend AI libraries (PyTorch, TensorRT, etc.) install on demand through the API instead.
