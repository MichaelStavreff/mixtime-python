# mixtime port — shared notes

- Rust core: [`rust/`](rust/README.md)
- Python wrapper: [`python/`](python/README.md)
- Coverage map: [`COVERAGE.md`](COVERAGE.md)
- Decision record: [`DECISION.md`](DECISION.md)

Scope is the four Gregorian `chronon_divmod` splits from
`R/cal_gregorian.R`: day→month (including 3-month/quarter blocks), day→year,
month→day (including multi-month granules), and year→day. Parsing, formatting,
timezones, astronomical calendars, and other calendar systems are not ported.

## Upstream sync

The procedure below was exercised on 2026-10-23 from a **clean shell**
(`env -i`, no profile) against GitHub `main` (`mixtime 0.4.0.9000`) starting
from the pinned CRAN `0.4.0` baseline.

1. **Snapshot the current fixtures.** The repo is not a git repository, so keep
   a copy for reconciliation:
   ```sh
   mkdir -p /tmp/opencode/fixtures_before
   cp fixtures/{fixtures.json,crosscheck.json} /tmp/opencode/fixtures_before/
   ```
2. **Install the new R source into the isolated library.** Do not touch the
   system R library:
   ```sh
   R_LIBS=/tmp/opencode/Rlib R CMD INSTALL --library=/tmp/opencode/Rlib /path/to/new/mixtime
   ```
3. **Replace the pinned upstream source.** `upstream/mixtime` is read-only, so
   unlock it first:
   ```sh
   chmod -R u+w upstream/mixtime
   rm -rf upstream/mixtime
   cp -r /path/to/new/mixtime upstream/mixtime
   chmod -R a-w upstream/mixtime
   ```
4. **Regenerate the fixtures from the R original:**
   ```sh
   R_LIBS=/tmp/opencode/Rlib just fixtures
   ```
5. **Rerun both suites.** This rebuilds the Rust extension before pytest:
   ```sh
   just test
   ```
6. **Reconcile the diffs.** Compare `cases` and `inputs`, not just the
   `package_version` metadata. The timed script does this automatically:
   ```sh
   python3 - <<'PY'
   import json
   from pathlib import Path
   before = Path("/tmp/opencode/fixtures_before")
   after = Path("fixtures")
   for name in ("fixtures.json", "crosscheck.json"):
       b = json.loads((before / name).read_text())
       a = json.loads((after / name).read_text())
       print(name, b["cases"] == a["cases"], b["inputs"] == a["inputs"],
             b["package_version"], "->", a["package_version"])
   PY
   ```
   If numeric cases differ, fix the port or document an intentional divergence;
   never loosen a tolerance to hide a diff. If only `package_version` changed,
   the port is numerically compatible.
7. **Record the outcome.** Update version pins and this README, or revert to the
   pinned CRAN release if the bump was a rehearsal. Do not leave `upstream/`
   and the fixtures on different versions.

The whole loop is scripted in
[`benchmarks/sync_timing.sh`](benchmarks/sync_timing.sh).
The clean-shell invocation used for the timing was:

```sh
env -i HOME="$HOME" PATH="/path/to/just:/usr/bin:/bin" \
  R_LIBS=/path/to/Rlib BEFORE_FIXTURES=/tmp/mixtime-fixtures-before \
  bash --noprofile --norc -c \
  'cd /path/to/mixtime && ./benchmarks/sync_timing.sh /path/to/mixtime-main'
```

Measured wall-clock on the final 14-test suite: **24.41 s** total (install +
replace 9.57 s, regenerate fixtures 1.15 s, Rust + Python suites 13.61 s,
reconciliation 0.08 s). Both fixture files had identical `cases` and `inputs`;
only `package_version` changed (`0.4.0` → `0.4.0.9000`). The rehearsal was then
reverted to the pinned CRAN `0.4.0` baseline.

### Environment-specific dependencies

| Dependency | Value / requirement | Why it matters |
|---|---|---|
| `R_LIBS` | `/path/to/Rlib` | mixtime and its R dependencies are not in the system library |
| `just` | on `PATH` | task entrypoint |
| `HOME` | your home directory | cargo/uv caches and R's user library (cpp11 headers) |
| R | 4.6.1, plus `jsonlite` | fixture generation |
| Rust | 1.98.1 (`cargo`, `rustc`) | rebuilds the PyO3 extension |
| Python / uv | 3.14.7 / 0.13.4 | pytest environment |
| C++ compiler | `g++` (C++20) | mixtime's compiled timezone code |
| Network | first run only | downloading the new R source and crates |
| `BEFORE_FIXTURES` | `/tmp/mixtime-fixtures-before` | reconciliation snapshot |

A fresh machine needs R + `jsonlite`, the mixtime dependency chain, a Rust
toolchain, uv, just, and a C++20 compiler. CI can skip R unless it regenerates
fixtures.
