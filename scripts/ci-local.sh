#!/usr/bin/env bash
# ci-local.sh — local mirror of the GitHub Actions Lint+Unit job
# in .github/workflows/ci-cd.yaml (job: pre-build-checks).
#
# Goal: run the EXACT same commands locally so that pre-push gate and CI
# never diverge again. Anything that passes here must pass in CI.
#
# Run from the repo root:
#   ./scripts/ci-local.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# === Install dependencies ===
uv sync --all-extras

# === Run ruff linter ===
uv run ruff check src/ tests/

# === Run ruff formatter (check only) ===
uv run ruff format --check src/ tests/

# === Run ty type checking ===
uv run ty check src/

# === Run unit tests ===
mkdir -p test-results
uv run pytest tests/ -v --tb=short -x \
    --cov=src --cov-report=xml --cov-report=term-missing \
    --junitxml=test-results/unit-tests.xml

# === Security scan - detect secrets ===
uv run detect-secrets scan --list --baseline .secrets.baseline

# === Security scan - pip-audit ===
uv run pip-audit

# === Security scan - bandit ===
uv run bandit -r src/ -c pyproject.toml

echo ""
echo "ci-local.sh: all checks passed."
