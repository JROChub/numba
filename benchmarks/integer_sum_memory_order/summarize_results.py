"""Validate every controlled trial and summarize without excluding any case."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path,
        default=directory.parents[1] / "numba" / "np" / "arraymath.py",
        help="Patched numba/np/arraymath.py whose digest must match the trials",
    )
    args = parser.parse_args()
    tags = ("before-1", "after-2", "after-3", "before-4")
    groups = {}
    source_hashes = {"before": set(), "after": set()}
    revisions = set()
    versions = set()
    for group in ("integer", "image"):
        trials = {
            tag: json.loads((directory / f"controlled-{group}-{tag}.json").read_text())
            for tag in tags
        }
        cases = [row["case"] for row in trials[tags[0]]["measurements"]]
        for tag, trial in trials.items():
            assert trial["label"] == tag
            assert trial["cpu_affinity"] == [0]
            assert trial["repeats"] == 31
            assert trial["jit_compilation_excluded"]
            assert not trial["parallel_jit"]
            assert [row["case"] for row in trial["measurements"]] == cases
            phase = tag.split("-")[0]
            source_hashes[phase].add(trial["provenance"]["arraymath_sha256"])
            revisions.add(trial["provenance"]["git_revision"])
            versions.add(json.dumps(trial["versions"], sort_keys=True))
        rows = []
        for index, case in enumerate(cases):
            measured = {tag: trial["measurements"][index]
                        for tag, trial in trials.items()}
            for field in ("input_sha256", "numba_result", "compiled_signature",
                          "shape", "strides_bytes", "input_dtype"):
                assert len({json.dumps(row[field], sort_keys=True)
                            for row in measured.values()}) == 1, (case, field)
            for row in measured.values():
                assert row["input_unchanged"]
                assert row["small_oracle"]["accepted"]
                assert row["numba_result"] == row["numpy_same_accumulator_reference"]
                if group == "image":
                    assert row["full_python_integer_oracle"]["accepted"]
                for engine in ("numpy", "numba"):
                    samples = row["samples_ns"][engine]
                    assert len(samples) == 31 and all(t > 0 for t in samples)
                    assert statistics.median(samples) == row["median_ns"][engine]
            medians = {tag: row["median_ns"]["numba"]
                       for tag, row in measured.items()}
            before = statistics.median([medians["before-1"], medians["before-4"]])
            after = statistics.median([medians["after-2"], medians["after-3"]])
            rows.append({
                "case": case,
                "before_median_ms": before / 1e6,
                "after_median_ms": after / 1e6,
                "before_over_after_ratio": before / after,
                "after_time_change_percent": (after / before - 1) * 100,
                "trial_median_ns": medians,
                "input_sha256": measured[tags[0]]["input_sha256"],
                "accepted_result": measured[tags[0]]["numba_result"],
                "all_trial_results_identical": True,
            })
        groups[group] = rows
    assert len(revisions) == 1
    assert all(len(hashes) == 1 for hashes in source_hashes.values())
    expected = {"python", "numpy", "numba", "llvmlite"}
    assert len({json.dumps({k: v.removesuffix(".dirty") if k == "numba" else v
                           for k, v in json.loads(item).items() if k in expected},
                          sort_keys=True)
                for item in versions}) == 1
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() in source_hashes["after"]
    print(json.dumps({
        "schema": "numba-integer-reduction-comparison.v1",
        "method": "median of two process medians per build; ratio before/after",
        "trial_order": list(tags), "samples_per_case_per_process_per_engine": 31,
        "host_scope": "one host, CPU affinity 0; not a dedicated isolated machine",
        "comparison": "same pinned upstream revision and dependencies, with/without patch",
        "upstream_revision": next(iter(revisions)),
        "arraymath_sha256": {phase: next(iter(hashes))
                             for phase, hashes in source_hashes.items()},
        "all_cases_included": True,
        "all_input_and_output_identities_match": True,
        "groups": groups,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
