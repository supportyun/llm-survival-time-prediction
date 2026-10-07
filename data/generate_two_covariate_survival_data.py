"""Generate the two-covariate synthetic survival cohorts used by SurvLIFT.

This is a portable version of the exploratory data-generation script used on
the research server. The command-line interface and portable path handling
were added for this public repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from patsy import dmatrix
from scipy.stats import weibull_min


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate seeded two-covariate synthetic survival cohorts."
    )
    parser.add_argument(
        "--source-pbc",
        type=Path,
        required=True,
        help="Source PBC CSV used to match the censoring rate.",
    )
    parser.add_argument(
        "--base-synthetic",
        type=Path,
        default=Path(__file__).with_name("synthetic_pbc_data_N300.csv"),
        help="Intermediate cohort created by gen_synthetic.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "generated",
        help="Directory for the seeded output CSV files.",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--shape", type=float, default=1.5)
    parser.add_argument("--baseline-scale", type=float, default=13.0)
    return parser.parse_args()


def validate_input(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def estimate_ph_coefficients(base: pd.DataFrame, shape: float) -> pd.Series:
    aft = WeibullAFTFitter().fit(
        base[["time", "status", "trt", "age"]],
        duration_col="time",
        event_col="status",
        formula="C(trt) + age",
    )
    aft_coefficients = aft.params_.loc[
        "lambda_", ["C(trt)[T.2]", "age"]
    ]
    ph_coefficients = -shape * aft_coefficients
    ph_coefficients.index = ["trt", "age"]
    return ph_coefficients


def choose_censoring_bounds(
    event_times: np.ndarray,
    target_rate: float,
    maximum_year: int,
) -> tuple[int, int, float]:
    best: tuple[float, int, int, float] | None = None
    n = len(event_times)

    for lower in range(0, max(maximum_year - 1, 1)):
        for upper in range(lower + 1, maximum_year + 1):
            censoring = np.random.uniform(lower, upper, size=n)
            rate = 1.0 - np.mean(event_times <= censoring)
            candidate = (abs(rate - target_rate), lower, upper, rate)
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        raise ValueError("Could not determine censoring bounds.")
    _, lower, upper, rate = best
    return lower, upper, rate


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.source_pbc)
    base = pd.read_csv(args.base_synthetic)
    validate_input(source, {"time", "status"}, "Source PBC file")
    validate_input(
        base,
        {"id", "trt", "age", "time", "status"},
        "Intermediate synthetic file",
    )

    coefficients = estimate_ph_coefficients(base, args.shape)
    design = dmatrix("trt + age - 1", base, return_type="dataframe")
    design = design[coefficients.index]
    linear_predictor = design.to_numpy() @ coefficients.to_numpy()
    individual_scale = args.baseline_scale * np.exp(
        -linear_predictor / args.shape
    )

    target_censoring_rate = 1.0 - source["status"].mean()
    source_event_times = source.loc[source["status"] == 1, "time"]
    maximum_year = max(2, int(np.ceil(source_event_times.max() / 365.0)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(1, args.n_seeds + 1):
        np.random.seed(seed)
        event_times = weibull_min.rvs(
            c=args.shape, scale=individual_scale, size=len(base)
        )
        lower, upper, search_rate = choose_censoring_bounds(
            event_times, target_censoring_rate, maximum_year
        )
        censoring_times = np.random.uniform(lower, upper, size=len(base))

        cohort = base[["id", "trt", "age"]].copy()
        cohort["true_T"] = event_times
        cohort["C"] = censoring_times
        cohort["time"] = np.minimum(event_times, censoring_times)
        cohort["status"] = (event_times <= censoring_times).astype(int)
        cohort["new_lambda"] = individual_scale

        output = args.output_dir / (
            f"synthetic_pbc_data_N{len(base)}_given_2_covariate_seed{seed}.csv"
        )
        cohort.to_csv(output, index=False)
        print(
            f"Seed {seed}: saved {output}; "
            f"target censoring={target_censoring_rate:.4f}, "
            f"search={search_rate:.4f}, "
            f"final={1.0 - cohort['status'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
