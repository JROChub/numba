# Numba integer reduction: controlled engineering results

September 7, 2026. Published evidence for a proposed integer-reduction optimization.
The tested code is on the public fork's `perf/integer-sum-memory-order` branch;
supporting evidence is isolated on `evidence/integer-sum-memory-order`.
Upstream submission: [Numba PR #10820](https://github.com/numba/numba/pull/10820).

Publication copies standardize descriptive metadata and replace local username
and temporary-directory prefixes with `/workspace/contributions` and
`/workspace/qualification`. Timings, counters,
results, source digests and recorded failures are unchanged. Original unredacted
records remain local. The summary script's source-path selection and reproduction
instructions were made portable; the measured kernels and harnesses are unchanged.

Implementation commit: `8e4401c86d4c57484bc7efb7e82df4574bd1dcdb`, branch
`perf/integer-sum-memory-order`. The two-file patch applies cleanly to the pinned
baseline. The submitted source is byte-identical to the measured source;
commit-message changes do not alter implementation, tests or measurements.

The patch substantially improves several integer memory-layout cases: the
Fortran/transpose measurements below improve by 13.59–28.36×. The three public
image workloads improve by 1.49–1.67×. Results are not uniformly better:
**int64/sliced is 7.8% slower**, and the synthetic image pipeline is essentially
unchanged (1.01×). All 22 cases and every controlled trial are retained.

## Change and issue scope

The implementation changes no-axis summation traversal for integer or boolean
inputs with an integer accumulator. C/F-contiguous arrays use a flat physical
scan at every supported rank. Arbitrary-layout arrays of rank at most four
sort dimensions by stride magnitude, normalize negative strides, and coalesce
adjacent/singleton dimensions. Higher-rank arbitrary layouts retain the original
iterator. Per-element casts and accumulator selection are preserved. Optimized
accumulation uses explicit wrapping LLVM addition without no-overflow flags;
zero/overlapping strides preserve logical multiplicity.

Floating/complex arithmetic, boolean accumulators and explicit-axis reductions
retain their original path. This is a related integer optimization, **not a fix
for the original floating-point reproducer in [issue 9819][issue]**. The recorded
[unmodified 0.67.0 baseline](release-0.67.0-baseline.json) confirms a roughly 14×
Numba/NumPy floating-point transpose gap; that path remains unchanged by design.
No proprietary QQfenx source was used, and these are not QQfenx performance results.

## Controlled method

Both builds use upstream revision `b6168b053d8d7f22a8c7986f073cceb3c124e838`,
Python 3.13.12, NumPy 2.3.5 and llvmlite 0.50.0dev2; image decoding uses Pillow
12.2.0. The patched checkout differs only by its recorded local changes.
For each harness, two independent fresh processes per build run sequentially:
`before-1`, `after-2`, `after-3`, `before-4`. Each process records 31 warm paired
NumPy/Numba samples per case, alternating which engine runs first.

All processes are pinned to CPU 0 on one host; `NUMBA_NUM_THREADS=2`, but the
compiled functions do not use parallel JIT. This is not a dedicated isolated
machine or a multi-machine reproduction. Compilation, image decoding, hashing
and oracle checks are outside timing. Image pipeline timing includes its crop,
integer dark subtraction, clamping and reduction, but not file I/O.

Table times are the median of the two process medians for **Numba**, in ms.
The ratio is unpatched/patched time: above 1 means faster, below 1 slower.
NumPy raw timings remain in every trial file; these ratios are not comparisons
against NumPy. Small percentage differences should not be overinterpreted.

Synthetic integer inputs use seed 123 and a 128³ base, with six layouts per
dtype. Sliced views have fewer elements; zero-stride controls retain repeated
logical elements. Each before/after case has identical input hashes and results.
Large results match NumPy using Numba's actual accumulator dtype, preserving
existing platform-dependent promotion behavior. Small cases additionally match
a Python arbitrary-precision sum reduced modulo the output width.

## Every controlled case

| Case | Before ms | After ms | Before / after |
| --- | ---: | ---: | ---: |
| int64 / C | 1.437687 | 1.131698 | 1.270× |
| int64 / Fortran | 22.438741 | 1.123885 | 19.965× |
| int64 / transpose (2,0,1) | 16.892552 | 1.240119 | 13.622× |
| int64 / sliced | 0.989121 | 1.065803 | 0.928× |
| int64 / negative stride | 2.168854 | 1.231327 | 1.761× |
| int64 / zero stride | 0.860045 | 0.745715 | 1.153× |
| uint64 / C | 1.359956 | 1.164611 | 1.168× |
| uint64 / Fortran | 21.931898 | 1.208129 | 18.154× |
| uint64 / transpose (2,0,1) | 16.931994 | 1.246001 | 13.589× |
| uint64 / sliced | 1.033655 | 1.012067 | 1.021× |
| uint64 / negative stride | 2.234406 | 1.192402 | 1.874× |
| uint64 / zero stride | 0.829951 | 0.743552 | 1.116× |
| int32 / C | 0.927732 | 0.708636 | 1.309× |
| int32 / Fortran | 20.786542 | 0.732912 | 28.362× |
| int32 / transpose (2,0,1) | 14.792643 | 0.825988 | 17.909× |
| int32 / sliced | 0.616238 | 0.593935 | 1.038× |
| int32 / negative stride | 1.133910 | 0.761856 | 1.488× |
| int32 / zero stride | 0.810573 | 0.675092 | 1.201× |
| Synthetic image / dark-corrected ROI | 17.263153 | 17.077188 | 1.011× |
| Public image / channel-first total | 0.296367 | 0.177615 | 1.669× |
| Public image / cropped/reversed total | 0.215331 | 0.128856 | 1.671× |
| Public image / dark-corrected ROI | 1.825430 | 1.221599 | 1.494× |

## Public input and exact acceptance

The input is the original 512×512×3 uint8 [scikit-image astronaut image][image],
identified as public domain in [scikit-image's documentation][image-doc]. Its
file SHA-256 matches the [v0.25.2 registry][registry]:
`88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5`.
It was not resized, tiled, repeated or converted to synthetic data.

Every pixel contribution is independently checked using Python integers, not
the optimized iterator. The accepted totals are `90124324` for channel-first
aggregation, `70570997` for the 448×448 cropped/reversed view, and `52006844`
after the full dark-corrected ROI pipeline. All NumPy and Numba results match
these oracles in all four trials. Input immutability is also checked.
These are **ad-hoc workloads on a real public dataset**, not an upstream
scikit-image benchmark, a NASA application, or demonstrated application-wide speedup.

## Review, regressions and retained history

The [independent coalescing review](independent-review-coalesced.json) passes
640 separately constructed cases against a Python modular-integer oracle,
including RGB, negative/gapped/overlapping strides, empty arrays, unaligned
inputs and noncanonical boolean bytes. These are differential validation
results, not external human review or a security audit.

An earlier patch made the public RGB aggregation cases about 3.6–4.0× slower.
[Before](image-diagnostic-before.json), [initial patch](image-diagnostic-after.json)
and [coalesced diagnostics](image-diagnostic-coalesced.json) remain available.
Negative-stride normalization and dimension coalescing resolved that design
finding; the final controlled image results are reported above. Those initial
diagnostics are not mixed into the controlled estimates.

The initial unrestricted sorting network also exceeded a 30-second rank-32
compilation timeout. A compile-time rank guard now selects the original path
for arbitrary layouts above rank four. The [retained review](independent-review-final.json)
records the formerly timed-out case completing; it does not establish a compile
speedup. Earlier review and baseline artifacts have not been deleted.

The final source completed **539 upstream tests in 594.472 seconds**: 536
passed and three were skipped. The complete [test log](tests-final-full.log)
includes the runner's resource measurements. Command, from the patched checkout:

```sh
python -m numba.runtests -m 2 numba.tests.test_array_reductions numba.tests.test_array_methods -v
```

The six added test methods cover 372 input combinations, including explicit
unaligned signatures and high-rank fallbacks. The independent 640-case probe
is additional. Repository-config lint passes; explicitly checking the normally
excluded legacy test file found zero new diagnostics versus the baseline.
See [lint review](lint-review.json).

The earlier broad run was deliberately interrupted after the implementation
changed; its [partial log](tests-initial-interrupted.log) is retained, not counted
as a completed run. The 539-test result above comes from the final coalesced,
rank-guarded source, SHA-256
`35b24b313abd518331c13912a27f17bec19815a86c5f436fcefde8fe51c101db`.

## Reproduction artifacts and submission status

[Summary](controlled-summary.json) and [comparison script](summarize_results.py)
retain exact values; [integer](integer_benchmark.py) and [public-image](external_image_benchmark.py)
harnesses emit raw samples and source provenance. For each selected build, run
the harness with CPU 0 affinity and `--repeats 31`; select the checkout explicitly
with `PYTHONPATH`. The image harness additionally requires the pinned image path.
See [full reproduction instructions](REPRODUCE.md) and the
[two-file patch](numba-integer-memory-order.patch).

Raw integer trials: [before 1](controlled-integer-before-1.json),
[after 2](controlled-integer-after-2.json), [after 3](controlled-integer-after-3.json),
[before 4](controlled-integer-before-4.json). Raw image trials:
[before 1](controlled-image-before-1.json), [after 2](controlled-image-after-2.json),
[after 3](controlled-image-after-3.json), [before 4](controlled-image-before-4.json).

The contributor reported completing an outside review and authorized submission.
The differential checks above are not that outside review.
[PR #10820](https://github.com/numba/numba/pull/10820) tracks
upstream review and CI; submission does not represent acceptance. No bounty or
payment has been secured.

[issue]: https://github.com/numba/numba/issues/9819
[image]: https://github.com/scikit-image/scikit-image/raw/refs/tags/v0.25.2/skimage/data/astronaut.png
[image-doc]: https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut
[registry]: https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/_registry.py
