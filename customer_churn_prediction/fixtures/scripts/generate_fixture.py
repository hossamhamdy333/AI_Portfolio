"""
Generates a fake customer churn CSV with the exact same columns and dtypes
as the real Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv).

This is NOT real data. It exists so the SQL and dashboard can be built and
tested before the real CSV is available. Row count (7043) and column names
match the real dataset, and churn is wired up to depend on contract type,
tenure, and monthly charges so the segmentation logic has something
realistic to work with.

Run:
    python generate_fixture.py
Writes:
    ../data/telco_churn_fixture.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N_ROWS = 7043


def random_customer_ids(n):
    # matches the real format, e.g. "7590-VHVEG"
    digits = np.random.randint(1000, 9999, n)
    letters = [
        "".join(np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 5))
        for _ in range(n)
    ]
    return [f"{d}-{l}" for d, l in zip(digits, letters)]


def yes_no(n, p_yes=0.5):
    return np.random.choice(["Yes", "No"], size=n, p=[p_yes, 1 - p_yes])


customer_id = random_customer_ids(N_ROWS)
gender = np.random.choice(["Male", "Female"], N_ROWS)
senior_citizen = np.random.choice([0, 1], N_ROWS, p=[0.84, 0.16])
partner = yes_no(N_ROWS, 0.48)
dependents = yes_no(N_ROWS, 0.30)

tenure = np.random.randint(0, 73, N_ROWS)

phone_service = yes_no(N_ROWS, 0.90)
multiple_lines = np.where(
    phone_service == "No",
    "No phone service",
    np.random.choice(["Yes", "No"], N_ROWS, p=[0.42, 0.58]),
)

internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], N_ROWS, p=[0.34, 0.44, 0.22]
)


def internet_addon(internet_service, p_yes=0.4):
    # add-ons only make sense for customers who have internet
    out = np.empty(len(internet_service), dtype=object)
    has_internet = internet_service != "No"
    out[~has_internet] = "No internet service"
    n_with = has_internet.sum()
    out[has_internet] = np.random.choice(
        ["Yes", "No"], n_with, p=[p_yes, 1 - p_yes]
    )
    return out


online_security = internet_addon(internet_service, 0.29)
online_backup = internet_addon(internet_service, 0.34)
device_protection = internet_addon(internet_service, 0.34)
tech_support = internet_addon(internet_service, 0.29)
streaming_tv = internet_addon(internet_service, 0.38)
streaming_movies = internet_addon(internet_service, 0.39)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N_ROWS, p=[0.55, 0.21, 0.24]
)
paperless_billing = yes_no(N_ROWS, 0.59)
payment_method = np.random.choice(
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    N_ROWS,
    p=[0.34, 0.23, 0.22, 0.21],
)

# Base monthly charge, pushed up by internet type and add-on count
base_charge = np.random.uniform(18, 45, N_ROWS)
internet_bump = np.select(
    [internet_service == "DSL", internet_service == "Fiber optic"],
    [15, 35],
    default=0,
)
addon_cols = [
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
]
addon_bump = sum((col == "Yes").astype(int) for col in addon_cols) * 3.5
monthly_charges = np.round(base_charge + internet_bump + addon_bump, 2)

total_charges = np.round(monthly_charges * tenure, 2)
# real dataset has a handful of blank TotalCharges for tenure == 0 customers
total_charges_str = total_charges.astype(str)
total_charges_str[tenure == 0] = " "

# --- Churn, driven by the same signals the EDA notebook calls out:
# contract type, tenure, and lack of support add-ons.
churn_score = (
    (contract == "Month-to-month") * 1.6
    + (tenure < 12) * 1.3
    + (online_security == "No") * 0.5
    + (tech_support == "No") * 0.5
    + (internet_service == "Fiber optic") * 0.4
    - (contract == "Two year") * 1.4
    - (tenure > 48) * 0.8
    + np.random.normal(0, 0.9, N_ROWS)
)
churn_prob = 1 / (1 + np.exp(-(churn_score - 2.75)))
churn = np.where(np.random.uniform(size=N_ROWS) < churn_prob, "Yes", "No")

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges_str,
    "Churn": churn,
})

out_path = "../data/telco_churn_fixture.csv"
df.to_csv(out_path, index=False)
print(f"wrote {len(df)} rows to {out_path}")
print(f"fake churn rate: {(df['Churn'] == 'Yes').mean() * 100:.2f}%")
