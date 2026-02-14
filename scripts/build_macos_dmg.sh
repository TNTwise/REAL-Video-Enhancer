#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/build_macos_dmg.sh [options]

Options:
  --python PATH   Python executable to use (default: python3.11)
  --no-clean      Keep existing build/venv artifacts
  --help          Show this help
USAGE
}

PYTHON_BIN="python3.11"
CLEAN=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --no-clean)
      CLEAN=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script only supports macOS." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Install Python 3.11 (recommended), or pass --python /path/to/python" >&2
  exit 1
fi

APP_NAME="REAL-Video-Enhancer"
APP_BUNDLE="dist/${APP_NAME}.app"
VERSION="$(sed -nE 's/^version = "([^"]+)"/\1/p' src/version.py | head -n 1)"
if [[ -z "$VERSION" ]]; then
  VERSION="unknown"
fi

DMG_NAME="${APP_NAME}-macOS-${VERSION}.dmg"
DMG_PATH="dist/${DMG_NAME}"
DMG_ROOT="dist/dmg-root"

if [[ "$CLEAN" -eq 1 ]]; then
  echo "==> Cleaning previous build artifacts"
  rm -rf venv dist build __pycache__ mainwindow.py resources_rc.py REAL-Video-Enhancer.spec
fi

echo "==> Building app bundle with ${PYTHON_BIN}"
"$PYTHON_BIN" build.py --build pyinstaller --copy_backend

if [[ ! -d "$APP_BUNDLE" ]]; then
  echo "App bundle not found: $APP_BUNDLE" >&2
  exit 1
fi

echo "==> Creating DMG layout"
rm -rf "$DMG_ROOT"
mkdir -p "$DMG_ROOT"
cp -R "$APP_BUNDLE" "$DMG_ROOT/"
ln -s /Applications "$DMG_ROOT/Applications"

echo "==> Packaging DMG"
hdiutil create -volname "REAL Video Enhancer" -srcfolder "$DMG_ROOT" -ov -format UDZO "$DMG_PATH"

echo "==> DMG created"
ls -lh "$DMG_PATH"
shasum -a 256 "$DMG_PATH"

echo "Done. Output: $ROOT_DIR/$DMG_PATH"
