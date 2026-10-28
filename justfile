set shell := ["bash", "-euo", "pipefail", "-c"]

# Build the Rust extension and sync the Python environment.
build:
    cd python && uv sync --reinstall-package mixtime

# Run the Rust unit tests.
test-rust:
    cd rust && cargo test

# Run the Python fixture tests (rebuilds the extension first).
test-python: build
    cd python && uv run pytest

# Run all tests (Rust + Python).
test: test-rust test-python

# Build the wheel (Rust extension plus Python sources).
wheel:
    cd python && uv build --wheel

# Build the wheel, install it into a clean venv, and run a smoke test.
wheel-smoke: wheel
    uv venv --clear /tmp/mixtime-wheel-smoke
    uv pip install --python /tmp/mixtime-wheel-smoke/bin/python python/dist/*.whl
    /tmp/mixtime-wheel-smoke/bin/python -c "import mixtime as mt; assert mt.yearmonth(['2024-03', '2024-01']).sort().format() == ['2024 Jan', '2024 Mar']; print('wheel smoke test passed')"

# Regenerate the R golden fixtures (requires mixtime installed in R; set R_LIBS if non-default).
fixtures:
    Rscript fixtures/generate_fixtures.R
    Rscript fixtures/generate_time_model_fixtures.R

# Remove build and test artifacts.
clean:
    cd rust && cargo clean
    rm -rf python/.venv python/.pytest_cache
    find python -type d -name __pycache__ -prune -exec rm -rf {} +
