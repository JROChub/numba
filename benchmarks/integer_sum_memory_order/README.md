# Integer reductions in memory order

A proposed Numba optimization for no-axis integer sums, with exact-oracle checks
and controlled before/after measurements. Floating-point reductions are unchanged.

| Measured workload | Unpatched / patched time |
| --- | ---: |
| int32, Fortran layout | 28.362× |
| int64, transposed layout | 13.622× |
| Public-image dark-correction, clamp and reduction | 1.494× |
| int64, sliced control | 0.928× (7.8% slower) |

Measurements are from one host, relative to the same unmodified Numba revision,
not NumPy. All 22 cases and all controlled samples are published, including
regressions. These are Numba results, not QQfenx performance measurements.

The final source passed 536 regression tests with three skips and 640 additional
separately constructed oracle cases. These are differential validation results,
not an external audit. Another-machine reproduction remains pending. The
contributor reported completing an outside review before authorizing upstream
submission.

- [Complete results, methodology and retained failures](RESULTS.md)
- [Reproduction instructions](REPRODUCE.md)
- [Proposed two-file patch](numba-integer-memory-order.patch)
- [Upstream PR #10820](https://github.com/numba/numba/pull/10820)
- [Submission text](UPSTREAM_PR.md)
- [All controlled results](controlled-summary.json)
- [Complete final test log](tests-final-full.log)
- [Code-only branch](https://github.com/JROChub/numba/tree/perf/integer-sum-memory-order)

The code-only branch contains implementation commit
`8e4401c86d4c57484bc7efb7e82df4574bd1dcdb`. This evidence branch adds supporting
artifacts separately. The upstream proposal contains two implementation/test
files plus a PR-numbered release-note fragment; the measured code is unchanged.

Publication copies standardize descriptive metadata and replace local username
and temporary-directory prefixes with `/workspace/contributions` and
`/workspace/qualification`.
Numerical measurements, result identities and source digests are unchanged.
Original local records are retained. `summarize_results.py` accepts `--source`
for portable source-hash verification. Historical diagnostics are explicitly
distinguished from the final controlled trials in the results report.

Published and submitted with the repository owner's approval. Follow
[PR #10820](https://github.com/numba/numba/pull/10820) for current review and CI
status; submission is not upstream acceptance. Numba's existing BSD-2-Clause
license remains unchanged; no proprietary Lights Out, QQfenx or Power House
code is included.
