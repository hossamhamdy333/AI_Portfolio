"""
Generates a small synthetic marketing_AB-shaped CSV for one of three
scenarios, used by test_ab_analysis.py to check the significance test
logic before it's trusted on the real data.

Usage: python3 make_fake_data.py <scenario>
where scenario is one of: real_effect, no_effect, borderline

Writes to ../data/raw/marketing_AB_fake_<scenario>.csv directly (not to
the shared marketing_AB_fake.csv name used by sql/01_campaign_summary.sql
and test_sql.py -- using the same path here would let this script
silently overwrite that fixture).
"""
import sys
import numpy as np
import pandas as pd

SCENARIOS = {
    # (true ad conversion rate, true psa conversion rate, n_ad, n_psa)
    "real_effect": (0.05, 0.02, 4000, 1000),
    "no_effect":   (0.025, 0.025, 4000, 1000),
    "borderline":  (0.026, 0.024, 300, 300),
}


def make_group(n, rate, group_name, rng):
    return pd.DataFrame({
        "user id": rng.integers(1_000_000, 9_999_999, size=n),
        "test group": group_name,
        "converted": rng.random(n) < rate,
        "total ads": rng.integers(1, 200, size=n),
        "most ads day": rng.choice(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            size=n,
        ),
        "most ads hour": rng.integers(0, 24, size=n),
    })


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SCENARIOS:
        print(f"usage: python3 make_fake_data.py <{'|'.join(SCENARIOS)}>")
        sys.exit(1)

    scenario = sys.argv[1]
    rate_ad, rate_psa, n_ad, n_psa = SCENARIOS[scenario]

    rng = np.random.default_rng(seed=42)
    df = pd.concat([
        make_group(n_ad, rate_ad, "ad", rng),
        make_group(n_psa, rate_psa, "psa", rng),
    ], ignore_index=True)

    out_path = f"../data/raw/marketing_AB_fake_{scenario}.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(df)} rows "
          f"(ad rate={df[df['test group']=='ad']['converted'].mean():.4f}, "
          f"psa rate={df[df['test group']=='psa']['converted'].mean():.4f})")


if __name__ == "__main__":
    main()
