set shell := ["bash", "-euo", "pipefail", "-c"]

# Build the Rust extension and sync the Python environment.
build:
    cd python && uv sync --reinstall-package mixtime-rs

# Run the Rust unit tests.
test-rust:
    cd rust && cargo test

# Run the Python fixture tests (rebuilds the extension first).
test-python: build
    cd python && uv run pytest

# Run all tests (Rust + Python).
test: test-rust test-python

# Regenerate the R golden fixtures (requires mixtime installed in R; set R_LIBS if non-default).
fixtures:
    Rscript fixtures/generate_fixtures.R
    Rscript fixtures/generate_time_model_fixtures.R

# Remove build and test artifacts.
clean:
    cd rust && cargo clean
    rm -rf python/.venv python/.pytest_cache
    find python -type d -name __pycache__ -prune -exec rm -rf {} +
