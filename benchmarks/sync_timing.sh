#!/usr/bin/env bash
# Time the full upstream-sync loop for the mixtime port.
#
# Usage (from a clean shell, with R_LIBS and just on PATH):
#   R_LIBS=/path/to/Rlib ./benchmarks/sync_timing.sh \
#       /path/to/new/mixtime/source
#
# Steps: install new R source -> replace upstream/mixtime -> regenerate
# fixtures -> run Rust + Python suites -> compare fixtures to a snapshot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?usage: sync_timing.sh /path/to/mixtime-source}"
BEFORE="${BEFORE_FIXTURES:-/tmp/mixtime-fixtures-before}"
: "${R_LIBS:?R_LIBS must be set}"

cd "$ROOT"

now() { date +%s.%N; }
elapsed() { python3 -c "print(f'{$2 - $1:.2f}')"; }

t0=$(now)

echo "== [1/5] install upstream source: $SRC =="
R CMD INSTALL --library="$R_LIBS" "$SRC"

echo "== [2/5] replace upstream/mixtime =="
chmod -R u+w upstream/mixtime
rm -rf upstream/mixtime
cp -r "$SRC" upstream/mixtime
chmod -R a-w upstream/mixtime
t1=$(now)
echo "install+replace: $(elapsed "$t0" "$t1") s"

echo "== [3/5] regenerate fixtures =="
just fixtures
t2=$(now)
echo "fixtures: $(elapsed "$t1" "$t2") s"

echo "== [4/5] rerun Rust + Python suites =="
just test
t3=$(now)
echo "tests: $(elapsed "$t2" "$t3") s"

echo "== [5/5] reconcile fixture diffs =="
python3 - "$BEFORE" <<'PY'
import json
import sys
from pathlib import Path

before = Path(sys.argv[1])
root = Path.cwd() / "fixtures"

for name in ("fixtures.json", "crosscheck.json"):
    b = json.loads((before / name).read_text())
    a = json.loads((root / name).read_text())
    bv, av = b.get("package_version"), a.get("package_version")
    same_cases = b.get("cases") == a.get("cases")
    same_inputs = b.get("inputs") == a.get("inputs")
    print(
        f"{name}: cases={'identical' if same_cases else 'DIFF'} "
        f"inputs={'identical' if same_inputs else 'DIFF'} "
        f"version={bv}->{av}"
    )
    if not same_cases:
        for i, (bc, ac) in enumerate(zip(b["cases"], a["cases"])):
            if bc != ac:
                print(f"  first differing case: {bc.get('name')} (index {i})")
                break
PY
t4=$(now)
echo "reconcile: $(elapsed "$t3" "$t4") s"
echo "TOTAL: $(elapsed "$t0" "$t4") s"
