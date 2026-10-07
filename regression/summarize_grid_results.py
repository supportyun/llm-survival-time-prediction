import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_FILENAME = "final_summary_results.csv"
DEFAULT_STATS_FILENAME = "final_summary_results_stats.csv"
# NOTE: 공부용 메모 - 하이퍼파라미터 조합 기준.
CONFIG_COLS = ["Rank", "Alpha", "LR", "Target_Scale"]
# NOTE: 공부용 메모 - 집계 대상 성능 지표 컬럼(있으면 mean/std 계산).
KNOWN_METRICS = [
    "RMSE",
    "Invalid_Ratio",
    "Final_Invalid_Ratio",
    "Best_Train_Loss",
    "Best_Val_Loss",
    "Best_Val_RMSE",
    "Best_Invalid_Ratio",
    "Best_Final_Invalid_Ratio",
    "Best_Greedy_Success_Rate",
    "Best_Retry_Success_Rate",
    "Best_Fallback_Rate",
    "Test_Greedy_Success_Rate",
    "Test_Retry_Success_Rate",
    "Test_Fallback_Rate",
    "Test_Loss",
]
ALPHA_CANDIDATES = ["Alpha", "Lora_Alpha", "LoRA_Alpha", "lora_alpha"]
CONFIG_DIR_RE = re.compile(
    r"^epochs_(?P<epochs>\d+)_rank(?P<rank>\d+)_alpha(?P<alpha>\d+)_lr(?P<lr>[^_]+)_drop(?P<drop>[^_]+)$"
)
SCALE_DIR_RE = re.compile(r"^scale_(?P<scale>log|linear)$")
DATA_SEED_RE = re.compile(r"^data(?P<data_seed>\d+)$")
MODEL_SEED_RE = re.compile(r"^seed_(?P<model_seed>\d+)$")


def _get_metric_cols(df: pd.DataFrame) -> list[str]:
    # NOTE: 공부용 메모 - 데이터에 존재하는 metric만 선택.
    return [col for col in KNOWN_METRICS if col in df.columns]


def _make_key(row: pd.Series) -> tuple:
    # NOTE: 공부용 메모 - NaN은 '<NA>'로 바꿔 그룹 키 안정화.
    return tuple("<NA>" if pd.isna(value) else value for value in row)


def _order_map(df: pd.DataFrame, cols: list[str]) -> dict[tuple, int]:
    # NOTE: 공부용 메모 - 등장 순서대로 정렬 기준 생성.
    order = {}
    for _, row in df[cols].iterrows():
        key = _make_key(row)
        if key not in order:
            order[key] = len(order)
    return order


def _apply_config_order(
    df: pd.DataFrame, config_order: dict[tuple, int], cols: list[str]
) -> pd.DataFrame:
    # NOTE: 공부용 메모 - config 순서를 유지해 보기 좋게 정렬.
    ordered = df.copy()
    ordered["_config_order"] = [
        config_order[_make_key(row)] for _, row in ordered[cols].iterrows()
    ]
    ordered = ordered.sort_values(["_config_order"]).drop(columns=["_config_order"])
    return ordered


def summarize_results(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # NOTE: 공부용 메모 - final_summary_results.csv를 읽어 mean/std 통계를 계산.
    df = pd.read_csv(csv_path)
    if "Alpha" not in df.columns:
        for candidate in ALPHA_CANDIDATES:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "Alpha"})
                break
    required_cols = {"Data_Seed", *CONFIG_COLS}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    metric_cols = _get_metric_cols(df)
    if not metric_cols:
        raise ValueError("No metric columns found to summarize.")

    config_order = _order_map(df, CONFIG_COLS)
    data_seed_order = _order_map(df, ["Data_Seed"])

    # NOTE: 공부용 메모 - (Data_Seed, config) 기준 mean/std.
    per_seed_grouped = (
        df.groupby(["Data_Seed", *CONFIG_COLS], dropna=False, sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_seed_grouped.columns = [
        "_".join(col).strip("_") for col in per_seed_grouped.columns.to_flat_index()
    ]
    per_seed_grouped["n_runs"] = (
        df.groupby(["Data_Seed", *CONFIG_COLS], dropna=False, sort=False)
        .size()
        .values
    )

    per_seed_grouped["_seed_order"] = [
        data_seed_order[_make_key(pd.Series([seed]))]
        for seed in per_seed_grouped["Data_Seed"]
    ]
    per_seed_grouped["_config_order"] = [
        config_order[_make_key(row)] for _, row in per_seed_grouped[CONFIG_COLS].iterrows()
    ]
    per_seed_grouped = per_seed_grouped.sort_values(
        ["_seed_order", "_config_order"]
    ).drop(columns=["_seed_order", "_config_order"])

    # NOTE: 공부용 메모 - config 기준 mean/std (seed 전체).
    overall_grouped = (
        df.groupby(CONFIG_COLS, dropna=False, sort=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    overall_grouped.columns = [
        "_".join(col).strip("_") for col in overall_grouped.columns.to_flat_index()
    ]
    overall_grouped["n_seeds"] = (
        df.groupby(CONFIG_COLS, dropna=False, sort=False)["Data_Seed"]
        .nunique()
        .values
    )
    overall_grouped = _apply_config_order(overall_grouped, config_order, CONFIG_COLS)

    return per_seed_grouped, overall_grouped


def _write_fixed_width_table(df: pd.DataFrame, out_path: Path) -> None:
    # NOTE: 공부용 메모 - 헤더 포함 고정 폭 텍스트 출력.
    rows = [list(df.columns)] + df.astype(str).values.tolist()
    if not rows:
        return
    num_cols = len(rows[0])
    widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        for idx, row in enumerate(rows):
            padded = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
            f.write(" | ".join(padded).rstrip() + "\n")
            if idx == 0:
                f.write("-+-".join("-" * w for w in widths).rstrip() + "\n")


def _parse_config_from_dir(dir_name: str) -> dict:
    # NOTE: 공부용 메모 - 폴더명에서 하이퍼파라미터 추출.
    match = CONFIG_DIR_RE.match(dir_name)
    if not match:
        return {}
    return {
        "Epochs": int(match.group("epochs")),
        "Rank": int(match.group("rank")),
        "Alpha": int(match.group("alpha")),
        "LR": match.group("lr"),
        "Dropout": float(match.group("drop")),
    }


def _find_target_scale(parts: tuple[str, ...]) -> str | None:
    # NOTE: 공부용 메모 - 경로에서 scale_log/scale_linear 찾기.
    for part in parts:
        match = SCALE_DIR_RE.match(part)
        if match:
            return match.group("scale")
    return None


def _find_seed_from_parts(parts: tuple[str, ...], regex: re.Pattern) -> int | None:
    # NOTE: 공부용 메모 - data{seed}, seed_{model} 같은 경로 조각 찾기.
    for part in parts:
        match = regex.match(part)
        if match:
            return int(next(iter(match.groupdict().values())))
    return None


def _metric_at_epoch(metrics: dict, key: str, epoch: int | None) -> float | None:
    # NOTE: 공부용 메모 - best epoch에 해당하는 metric 값 꺼내기.
    if not epoch:
        return None
    values = metrics.get(key)
    if not values:
        return None
    # NOTE: 공부용 메모 - validation_metrics는 epoch 0 포함 리스트인 경우가 많음.
    # epoch 0 포함이면 index=epoch, 아니면 index=epoch-1로 fallback.
    if 0 <= epoch < len(values):
        return values[epoch]
    if 1 <= epoch <= len(values):
        return values[epoch - 1]
    return None


def backfill_best_model_info(results_root: Path) -> int:
    # NOTE: 공부용 메모 - 기존 실험의 best_model_info.json에 비율 값을 채움.
    updated = 0
    for best_info_path in results_root.rglob("best_model_info.json"):
        metrics_path = best_info_path.with_name("validation_metrics.json")
        if not metrics_path.exists():
            continue

        try:
            with open(best_info_path, "r") as f:
                best_info = json.load(f)
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            continue

        best_epoch = best_info.get("epoch")
        if not best_epoch:
            continue

        greedy = _metric_at_epoch(metrics, "greedy_success_rate", best_epoch)
        retry = _metric_at_epoch(metrics, "retry_success_rate", best_epoch)
        fallback = _metric_at_epoch(metrics, "fallback_rate", best_epoch)

        changed = False
        for key, value in [
            ("greedy_success_rate", greedy),
            ("retry_success_rate", retry),
            ("fallback_rate", fallback),
        ]:
            if key not in best_info or best_info.get(key) is None:
                best_info[key] = value
                changed = True

        if changed:
            with open(best_info_path, "w") as f:
                json.dump(best_info, f, indent=2)
            updated += 1

    return updated


def collect_final_summary(results_root: Path) -> pd.DataFrame:
    # NOTE: 공부용 메모 - results 폴더를 스캔해 final_summary_results.csv를 재구성.
    rows = []
    for test_metrics_path in results_root.rglob("test_metrics.json"):
        config_dir = test_metrics_path.parent
        config = _parse_config_from_dir(config_dir.name)
        if not config:
            continue

        parts = tuple(test_metrics_path.parts)
        data_seed = _find_seed_from_parts(parts, DATA_SEED_RE)
        model_seed = _find_seed_from_parts(parts, MODEL_SEED_RE)
        target_scale = _find_target_scale(parts)
        # NOTE: 공부용 메모 - Experiment 컬럼은 날짜 폴더 + data 폴더 조합.
        experiment = results_root.name
        if results_root in test_metrics_path.parents:
            relative = test_metrics_path.relative_to(results_root)
            if relative.parts and relative.parts[0].startswith("data"):
                experiment = f"{results_root.name}_{relative.parts[0]}"
            elif len(relative.parts) > 1 and relative.parts[1].startswith("data"):
                experiment = f"{relative.parts[0]}_{relative.parts[1]}"
            else:
                experiment = relative.parts[0] if relative.parts else results_root.name

        with open(test_metrics_path, "r") as fp:
            test_metrics = json.load(fp)

        best_model_info = {}
        best_model_path = config_dir / "best_model_info.json"
        if best_model_path.exists():
            with open(best_model_path, "r") as fp:
                best_model_info = json.load(fp)

        validation_metrics = {}
        validation_metrics_path = config_dir / "validation_metrics.json"
        if validation_metrics_path.exists():
            with open(validation_metrics_path, "r") as fp:
                validation_metrics = json.load(fp)

        best_epoch = best_model_info.get("epoch")
        best_train_loss = _metric_at_epoch(validation_metrics, "train_loss", best_epoch)
        best_val_loss = best_model_info.get("val_loss") or _metric_at_epoch(
            validation_metrics, "val_loss", best_epoch
        )
        best_val_rmse = best_model_info.get("val_rmse") or _metric_at_epoch(
            validation_metrics, "val_rmse", best_epoch
        )
        best_invalid_ratio = best_model_info.get("invalid_ratio") or _metric_at_epoch(
            validation_metrics, "invalid_answer_ratio", best_epoch
        )
        best_final_invalid_ratio = best_model_info.get(
            "final_invalid_ratio"
        ) or _metric_at_epoch(validation_metrics, "final_invalid_answer_ratio", best_epoch)
        best_greedy_success_rate = _metric_at_epoch(
            validation_metrics, "greedy_success_rate", best_epoch
        )
        best_retry_success_rate = _metric_at_epoch(
            validation_metrics, "retry_success_rate", best_epoch
        )
        best_fallback_rate = _metric_at_epoch(
            validation_metrics, "fallback_rate", best_epoch
        )

        rows.append(
            {
                "Experiment": experiment,
                "Data_Seed": data_seed,
                "Model_Seed": model_seed,
                "Target_Scale": target_scale,
                **config,
                "RMSE": test_metrics.get("test_rmse"),
                "Invalid_Ratio": test_metrics.get("invalid_answer_ratio"),
                "Final_Invalid_Ratio": test_metrics.get("final_invalid_answer_ratio"),
                "Test_Greedy_Success_Rate": test_metrics.get("greedy_success_rate"),
                "Test_Retry_Success_Rate": test_metrics.get("retry_success_rate"),
                "Test_Fallback_Rate": test_metrics.get("fallback_rate"),
                "Best_Train_Loss": best_train_loss,
                "Best_Val_Loss": best_val_loss,
                "Best_Val_RMSE": best_val_rmse,
                "Best_Invalid_Ratio": best_invalid_ratio,
                "Best_Final_Invalid_Ratio": best_final_invalid_ratio,
                "Best_Greedy_Success_Rate": best_greedy_success_rate,
                "Best_Retry_Success_Rate": best_retry_success_rate,
                "Best_Fallback_Rate": best_fallback_rate,
                "Best_Epoch": best_epoch,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No test_metrics.json files found under {results_root}")

    # NOTE: 공부용 메모 - 같은 config끼리 묶고 seed 순서로 정렬.
    # Order: config group first, then data seed within each config.
    config_order = _order_map(df, CONFIG_COLS)
    df["_config_order"] = [
        config_order[_make_key(row)] for _, row in df[CONFIG_COLS].iterrows()
    ]
    df = df.sort_values(["_config_order", "Data_Seed", "Model_Seed"]).drop(
        columns=["_config_order"]
    )

    # NOTE: 공부용 메모 - 성능 지표는 소수 4자리로 맞춤.
    # Format non-hyperparam numeric metrics to 4 decimals.
    metric_cols = [col for col in KNOWN_METRICS if col in df.columns]
    for col in metric_cols:
        df[col] = df[col].apply(
            lambda val: round(val, 4) if pd.notna(val) else val
        )

    return df


def main() -> None:
    # NOTE: 공부용 메모 - collect 모드: 결과 스캔/CSV 생성, 요약 모드: mean/std 생성.
    parser = argparse.ArgumentParser(
        description="Summarize grid search results by seed and overall config stats."
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect per-run metrics into final_summary_results.csv",
    )
    parser.add_argument(
        "--backfill-best-info",
        action="store_true",
        help="Fill greedy/retry/fallback rates into existing best_model_info.json files",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to final_summary_results.csv",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).with_name("results"),
        help="Root results folder containing experiment folders (dates).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment folder name under results root (e.g., 240101).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path for aggregated stats",
    )
    parser.add_argument(
        "--out-by-seed",
        type=Path,
        default=None,
        help="Optional output CSV path for per-seed aggregated stats",
    )
    args = parser.parse_args()

    if args.backfill_best_info:
        backfill_root = args.results_root
        if args.experiment:
            backfill_root = backfill_root / args.experiment
        updated = backfill_best_model_info(backfill_root)
        print("Updated best_model_info.json files:", updated)
        return

    if args.collect:
        collect_root = args.results_root
        if args.experiment:
            collect_root = collect_root / args.experiment
        out_csv = args.out or (collect_root / DEFAULT_RESULTS_FILENAME)
        collected = collect_final_summary(collect_root)
        collected.to_csv(out_csv, index=False)
        print("Saved collected results to:", out_csv)
        return

    if args.csv is not None:
        csv_path = args.csv
    elif args.experiment is not None:
        csv_path = (
            args.results_root / args.experiment / DEFAULT_RESULTS_FILENAME
        )
    else:
        csv_path = Path(__file__).with_name(DEFAULT_RESULTS_FILENAME)

    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    out_dir = csv_path.parent
    out_stats = args.out or (out_dir / DEFAULT_STATS_FILENAME)
    out_by_seed = args.out_by_seed

    per_seed_summary, overall_summary = summarize_results(csv_path)
    if out_by_seed is not None:
        per_seed_summary.to_csv(out_by_seed, index=False)
        print("Saved per-seed summary to:", out_by_seed)
        _write_fixed_width_table(
            per_seed_summary, out_by_seed.with_suffix(".txt")
        )
        print("Saved per-seed summary (fixed width) to:", out_by_seed.with_suffix(".txt"))
    overall_summary.to_csv(out_stats, index=False)
    print("Saved overall summary to:", out_stats)
    _write_fixed_width_table(overall_summary, out_stats.with_suffix(".txt"))
    print("Saved overall summary (fixed width) to:", out_stats.with_suffix(".txt"))


if __name__ == "__main__":
    main()
