"""Generate small test fixtures from the local 5-seed baseline."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "results" / "baseline"


def make_agg_fixture():
    rows = []
    if not SRC.exists():
        return
    for cfg_dir in sorted(SRC.iterdir()):
        if not cfg_dir.is_dir(): continue
        for seed_dir in sorted(cfg_dir.iterdir())[:2]:
            if not seed_dir.is_dir(): continue
            log = seed_dir / "log.json"
            if not log.exists(): continue
            d = json.load(open(log))
            last = d["rounds"][-1]
            parts = cfg_dir.name.split("_")
            rows.append({
                "label": f"{parts[0]}+{parts[1]}",
                "attack": parts[2],
                "seed": int(seed_dir.name.split("=")[1]),
                "acc": last["accuracy"],
                "carb": last["cum_carbon_g"],
            })
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "fixture_aggregation_n2.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out} ({len(df)} rows)")


def make_trust_fixture():
    if not SRC.exists():
        return
    for cfg_dir in SRC.iterdir():
        if cfg_dir.name.startswith("fedcamo_cr_pr+trust_weighted") and "mr" in cfg_dir.name:
            for seed_dir in cfg_dir.iterdir():
                log = seed_dir / "log.json"
                if not log.exists(): continue
                d = json.load(open(log))
                trust = [r["trust_mean"] for r in d["rounds"][:30]]
                lambda_ = [r["lambda_t"] for r in d["rounds"][:30]]
                mu = [r["mu_t"] for r in d["rounds"][:30]]
                df = pd.DataFrame({
                    "round": list(range(30)),
                    "trust_mean": trust,
                    "lambda": lambda_,
                    "mu": mu,
                })
                out = Path(__file__).resolve().parent / "fixture_trust_signal.csv"
                df.to_csv(out, index=False)
                print(f"  wrote {out} ({len(df)} rounds)")
                return


def make_carbon_fixture():
    rows = []
    if not SRC.exists():
        return
    for cfg_dir in sorted(SRC.iterdir()):
        if not cfg_dir.is_dir(): continue
        for seed_dir in sorted(cfg_dir.iterdir())[:3]:
            if not seed_dir.is_dir(): continue
            log = seed_dir / "log.json"
            if not log.exists(): continue
            d = json.load(open(log))
            last = d["rounds"][-1]
            parts = cfg_dir.name.split("_")
            rows.append({
                "label": f"{parts[0]}+{parts[1]}",
                "attack": parts[2],
                "seed": int(seed_dir.name.split("=")[1]),
                "acc": last["accuracy"],
                "carb": last["cum_carbon_g"],
            })
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "fixture_carbon_n3.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    print("Generating fixtures...")
    make_agg_fixture()
    make_trust_fixture()
    make_carbon_fixture()
    print("done.")
