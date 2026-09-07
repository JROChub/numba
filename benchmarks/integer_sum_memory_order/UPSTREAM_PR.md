# Upstream submission

Submitted as [Numba PR #10820](https://github.com/numba/numba/pull/10820) after
the contributor reported completing an outside review and authorized submission.
The description below is the submitted text; the PR tracks subsequent changes,
review and CI status.

## Title

Optimize no-axis integer sums in memory order

## Description

This changes traversal for no-axis integer and boolean sums with integer
accumulators. Contiguous arrays use a flat physical scan. Arbitrary layouts
up to rank four normalize negative strides, order dimensions by stride magnitude
and coalesce contiguous dimensions. Higher-rank arbitrary layouts retain the
existing iterator. Per-element casts and accumulator selection are preserved;
the optimized path uses wrapping LLVM addition without no-overflow flags.

Floating-point/complex inputs, floating or boolean accumulators and explicit-axis
reductions keep their existing path. This is related to the layout performance
problem in https://github.com/numba/numba/issues/9819, but does not fix its
floating-point reproducer or request closing that issue.

Validation on the final source: 536 tests passed and three skipped across
`numba.tests.test_array_reductions` and `numba.tests.test_array_methods`.
Six added test methods cover 372 combinations. A separately constructed
Python-integer oracle passed 640 additional cases. These are differential
validation results, not an external audit. Repository-config lint reports no
new issues.

On one host, controlled before/after trials measured 13.6× for int64 transposed
and 28.4× for int32 Fortran-layout sums; an ad-hoc public-image processing
pipeline improved 1.49×. One int64 sliced control was 7.8% slower. These compare
patched against unpatched Numba, not NumPy; no multi-machine result is claimed.
All 22 cases, raw trials, exactness checks, earlier failed diagnostics and
reproduction instructions are in the
[evidence branch](https://github.com/JROChub/numba/tree/evidence/integer-sum-memory-order/benchmarks/integer_sum_memory_order).
