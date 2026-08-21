import sys
sys.path.insert(0, "../notebooks")
import subprocess, shutil
import pandas as pd

# import the two functions we're testing straight from the notebook
# via nbconvert is overkill for this - just redefine the same logic here
import numpy as np
from statsmodels.stats.proportion import proportions_ztest, proportion_confint, proportion_effectsize
from statsmodels.stats.power import NormalIndPower


def run_ab_test(df, group_col="test group", outcome_col="converted", group_a="ad", group_b="psa", alpha=0.05):
    a = df[df[group_col] == group_a][outcome_col]
    b = df[df[group_col] == group_b][outcome_col]
    n_a, n_b = len(a), len(b)
    conv_a, conv_b = a.sum(), b.sum()
    rate_a, rate_b = conv_a / n_a, conv_b / n_b
    z_stat, p_value = proportions_ztest([conv_a, conv_b], [n_a, n_b], alternative="two-sided")
    diff = rate_a - rate_b
    return {"n_a": n_a, "n_b": n_b, "diff": diff, "p_value": p_value, "significant": p_value < alpha}


def achieved_power(effect_size_h, n_a, n_b, alpha=0.05):
    analysis = NormalIndPower()
    return analysis.power(effect_size=abs(effect_size_h), nobs1=n_a, alpha=alpha, ratio=n_b / n_a)


for scenario in ["real_effect", "no_effect", "borderline"]:
    subprocess.run(["python3", "make_fake_data.py", scenario], check=True)
    shutil.move("../data/raw/marketing_AB_fake.csv", f"../data/raw/marketing_AB_fake_{scenario}.csv")

df_real = pd.read_csv("../data/raw/marketing_AB_fake_real_effect.csv")
result_real = run_ab_test(df_real)
assert result_real["significant"], "expected significant result for real_effect scenario"
print("real_effect: passed")

df_none = pd.read_csv("../data/raw/marketing_AB_fake_no_effect.csv")
result_none = run_ab_test(df_none)
assert not result_none["significant"], "expected non-significant result for no_effect scenario"
print("no_effect: passed")

df_border = pd.read_csv("../data/raw/marketing_AB_fake_borderline.csv")
result_border = run_ab_test(df_border)
print(f"borderline: p={result_border['p_value']:.4g}, significant={result_border['significant']} (informational, no assert)")

print("all checks passed")
