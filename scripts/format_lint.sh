#!/usr/bin/env bash
set -euo pipefail

# ---------- config ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND="$PROJECT_ROOT/backend"

# Python files to check (backend + root, excludes venvs and cache)
FILES=$(find "$PROJECT_ROOT" -name "*.py" \
  -not -path "*/client/*" \
  -not -path "*/.venv/*" \
  -not -path "*/venv/*" \
  -not -path "*__pycache__*" \
  -not -path "*/bin/*")

MODE="${1:-check}"  # check (default) or fix

# ---------- colour helpers ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no colour

info()  { echo -e "${YELLOW}[format_lint] $*${NC}"; }
ok()    { echo -e "${GREEN}[PASS]  $*${NC}"; }
fail()  { echo -e "${RED}[FAIL]  $*${NC}"; }

EXIT_CODE=0

# ---------- 1. isort ----------
info "Running isort ($MODE mode)..."
if [ "$MODE" = "fix" ]; then
  isort --check-only=false --profile black $FILES
  ok "isort completed (fixed)"
else
  if isort --check-only --profile black $FILES; then
    ok "isort — imports are sorted"
  else
    fail "isort — unsorted imports found (run with 'fix' to auto-fix)"
    EXIT_CODE=1
  fi
fi

# ---------- 2. ruff format ----------
info "Running ruff format ($MODE mode)..."
if [ "$MODE" = "fix" ]; then
  ruff format --check=false $FILES
  ok "ruff format completed (fixed)"
else
  if ruff format --check $FILES; then
    ok "ruff format — code is formatted"
  else
    fail "ruff format — unformatted code found (run with 'fix' to auto-fix)"
    EXIT_CODE=1
  fi
fi

# ---------- 3. ruff check (lint) ----------
info "Running ruff check (lint)..."
if [ "$MODE" = "fix" ]; then
  ruff check --fix $FILES || true
  ok "ruff check completed (auto-fixed where possible)"
else
  if ruff check $FILES; then
    ok "ruff check — no lint issues"
  else
    fail "ruff check — lint issues found (run with 'fix' to auto-fix)"
    EXIT_CODE=1
  fi
fi

# ---------- 4. ty (type checking) ----------
info "Running ty (type check)..."
if [ "$MODE" = "fix" ]; then
  ty check $FILES || true
  ok "ty completed"
else
  if ty check $FILES; then
    ok "ty — no type issues"
  else
    fail "ty — type issues found"
    EXIT_CODE=1
  fi
fi

# ---------- summary ----------
echo ""
if [ $EXIT_CODE -eq 0 ]; then
  ok "All checks passed!"
else
  fail "Some checks failed. Run './scripts/format_lint.sh fix' to auto-fix where possible."
fi
exit $EXIT_CODE
