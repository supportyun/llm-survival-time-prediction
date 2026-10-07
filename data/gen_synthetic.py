"""Generate an intermediate PBC-like synthetic cohort.

This is a portable version of the exploratory script used on the research
server. The source PBC file is supplied explicitly because the original
clinical-data file is not distributed in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import WeibullFitter
from scipy.stats import weibull_min


BINARY_COLUMNS = ["trt", "sex", "ascites", "hepato", "spiders"]
CATEGORICAL_COLUMNS = ["edema", "stage"]
CONTINUOUS_COLUMNS = ["age", "bili", "albumin", "alk.phos", "ast", "protime"]
REQUIRED_COLUMNS = [
    "time",
    "status",
    *BINARY_COLUMNS,
    *CATEGORICAL_COLUMNS,
    *CONTINUOUS_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an intermediate PBC-like synthetic cohort."
    )
    parser.add_argument(
        "--source-pbc",
        type=Path,
        required=True,
        help="CSV containing the source PBC variables used to estimate distributions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("synthetic_pbc_data_N300.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--n", type=int, default=300, help="Synthetic sample size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def validate_source(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Source PBC file is missing columns: {missing}")


def generate_categorical(column: str, n: int, source: pd.DataFrame) -> np.ndarray:
    distribution = source[column].dropna().value_counts(normalize=True, sort=False)
    return np.random.choice(
        distribution.index.to_numpy(), size=n, p=distribution.to_numpy()
    )


def generate_covariates(source: pd.DataFrame, n: int) -> pd.DataFrame:
    synthetic: dict[str, np.ndarray | pd.Series] = {}

    for column in BINARY_COLUMNS:
        if column == "sex":
            encoded = source[column].map({"f": 1, "m": 0})
            probability = encoded.value_counts(normalize=True).get(1, 0.5)
            synthetic[column] = pd.Series(
                np.random.binomial(n=1, p=probability, size=n)
            ).map({0: "m", 1: "f"})
        else:
            probability = source[column].value_counts(normalize=True).get(1, 0.5)
            synthetic[column] = np.random.binomial(n=1, p=probability, size=n)
            if column == "trt":
                synthetic[column] = 2 - synthetic[column]

    for column in CATEGORICAL_COLUMNS:
        synthetic[column] = generate_categorical(column, n, source)

    for column in CONTINUOUS_COLUMNS:
        observed = source[column].dropna()
        if column in {"bili", "alk.phos", "protime"}:
            log_observed = np.log(observed)
            synthetic[column] = np.exp(
                np.random.normal(log_observed.mean(), log_observed.std(), size=n)
            )
        else:
            synthetic[column] = np.random.normal(
                observed.mean(), observed.std(), size=n
            )

    return pd.DataFrame(synthetic)


def choose_censoring_bounds(
    event_times: np.ndarray,
    n: int,
    target_censoring_rate: float,
) -> tuple[int, int, float]:
    maximum = int(np.max(event_times))
    best: tuple[float, int, int, float] | None = None

    for lower in range(0, max(maximum - 100, 1), 100):
        for upper in range(lower + 100, maximum + 1, 100):
            censoring = np.random.uniform(lower, upper, size=n)
            rate = 1.0 - np.mean(event_times <= censoring)
            candidate = (abs(rate - target_censoring_rate), lower, upper, rate)
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        raise ValueError("Could not determine censoring bounds from source times.")
    _, lower, upper, rate = best
    return lower, upper, rate


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    source = pd.read_csv(args.source_pbc)
    validate_source(source)
    synthetic = generate_covariates(source, args.n)

    weibull = WeibullFitter().fit(
        source["time"], event_observed=source["status"]
    )
    event_times = weibull_min.rvs(
        c=weibull.rho_, scale=weibull.lambda_, size=args.n
    )
    target_censoring_rate = 1.0 - source["status"].mean()
    lower, upper, achieved_rate = choose_censoring_bounds(
        event_times, args.n, target_censoring_rate
    )
    censoring_times = np.random.uniform(lower, upper, size=args.n)

    synthetic["true_T"] = event_times
    synthetic["C"] = censoring_times
    synthetic["time"] = np.minimum(event_times, censoring_times)
    synthetic["status"] = (event_times <= censoring_times).astype(int)
    synthetic["id"] = np.arange(1, args.n + 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(args.output, index=False)
    print(f"Saved {len(synthetic)} rows to {args.output}")
    print(
        "Censoring rate: "
        f"target={target_censoring_rate:.4f}, search={achieved_rate:.4f}, "
        f"final={1.0 - synthetic['status'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
