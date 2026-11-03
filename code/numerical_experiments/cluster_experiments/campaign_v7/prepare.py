"""Prepare corrected QP tuning and independently loaded real-data pilots."""

import json
from pathlib import Path

from releases.accounting_v7.experiments.io import identity, read_csv, write_csv
from releases.accounting_v7.experiments.plan import split

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign_v7"


def emit(name, configs):
    path = CAMPAIGN / "manifests" / f"{name}.csv"
    rows = [
        dict(task_id=i, run_id=identity(c), config=json.dumps(c, sort_keys=True))
        for i, c in enumerate(configs)
    ]
    if path.exists():
        if read_csv(path) != [{k: str(v) for k, v in row.items()} for row in rows]:
            raise ValueError(f"refusing to replace campaign manifest: {path}")
    else:
        write_csv(path, rows)
        path.chmod(0o444)
    return rows


def main():
    original = [
        json.loads(r["config"])
        for r in read_csv(ROOT / "manifests/qp_full_tune_20261030.csv")
    ]
    fast, screen = split(original)
    for config in [*fast, *screen]:
        config.update(
            setup_policy="independent", campaign_id="v7_20261031", cpu_feature="icelake"
        )
    emit("kaczmarz_candidates", [c for c in original if "kaczmarz" in c["inner"]])
    fast_rows = emit("qp_tune", fast)
    screen_rows = emit("kaczmarz_screen", screen)
    fast_pilot = next(
        i
        for i, c in enumerate(fast)
        if c["inner"] == "nystrom_pcg"
        and c["side"] == 100
        and c["rank"] == 128
        and c["seed"] == 0
        and c.get("sigma") == 0.2
        and c.get("stride") == 1
        and c.get("inner_initialization") == "warm"
    )
    screen_pilot = next(
        i
        for i, c in enumerate(screen)
        if c["inner"] == "block_kaczmarz"
        and c["block_size"] == 512
        and c["stride"] == 1
        and c.get("inner_initialization") == "warm"
    )
    sources = {}
    for dataset in ["realsim", "smallnorb", "rcv1", "cifar10"]:
        directory = ROOT / "results" / f"shared_dataset_pilots_v3_{dataset}"
        row = read_csv(next(directory.glob("*/result.csv")))[0]
        config = json.loads(row["config"])
        sources[dataset] = dict(dataset=config["dataset"], spec=config["matrix_source"])
    # Tall realsim and wide smallNORB retain their native labels and dimensions.
    # These pilots screen penalty/step scales; every run rebuilds its own setup.
    regression = []
    for dataset in ["smallnorb", "realsim"]:
        for family, inner in [
            ("lasso", "nystrom_pcg"),
            ("logistic", "gd"),
            ("lasso", "pdhg"),
            ("logistic", "pdhg"),
        ]:
            for ratio in [100.0, 10000.0] if inner != "pdhg" else [1.0, 10.0]:
                config = dict(
                    family=family,
                    inner=inner,
                    dataset=sources[dataset]["dataset"],
                    matrix_source=sources[dataset]["spec"],
                    orientation="native",
                    seed=0,
                    phase="pilot",
                    sweep="independent_budget_pilot",
                    implementation="accounting_v7",
                    campaign_id="v7_20261031",
                    cpu_feature="icelake",
                    blas_threads=1,
                    setup_policy="independent",
                    ratio=ratio,
                    sigma=0.2,
                    stride=5,
                    inner_initialization="warm",
                    check_source="recurrence",
                    budget_unit="work",
                    time_limit=1800.0,
                    inner_time_limit=300.0,
                    max_inner=100000,
                    max_inner_per_outer=10000,
                    max_outer=10000,
                    tolerance=1e-6,
                )
                if inner == "nystrom_pcg":
                    config["rank"] = 128
                if inner == "pdhg":
                    config["baseline_step"] = ratio
                regression.append(config)
    reg_rows = emit("regression_pilots", regression)
    plan = dict(
        fast_tasks=len(fast_rows),
        fast_batch_size=8,
        fast_batches=(len(fast_rows) + 7) // 8,
        fast_pilot=fast_pilot,
        screening_tasks=len(screen_rows),
        screening_pilot=screen_pilot,
        regression_pilots=len(reg_rows),
        regression_pilot=0,
        priority=[
            "qp_tune_and_kaczmarz_screen",
            "independent_real_data_pilots",
            "steps_then_held_out_evaluation",
        ],
        source_scope=(
            "Current prepared NysADMM-family artifacts; native labels; "
            "independent setup"
        ),
    )
    (CAMPAIGN / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
