# Design UI

- [ ] Design the interface for the new Tauri + React frontend

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


## Model Management
- [ ] Implement a local JSON file for model definitions — downloadable models include a URL and a `downloadable` boolean tag, while non-downloadable models use a direct file path

# Refactor Backend

## FFmpegWrite
- [ ] Refactor FFmpegWrite

## RenderVideo
- [ ] Refactor RenderVideo (after FFmpegWrite)

## InformationWriteout
- [ ] Refactor InformationWriteout

# Bundle Base Dependencies

Package FFmpeg and the Python runtime inside the app so it runs without system-level prerequisites. Per-backend AI libraries (PyTorch, TensorRT, etc.) install on demand through the API instead.

Make the ffmpeg render error out, instead of saying render started when calling the render api endpoint.
say fps is none if there is no render occuring

# Fix all code # TODOs
