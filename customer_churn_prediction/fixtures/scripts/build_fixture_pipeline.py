"""
Runs the same feature engineering, split, and model-calibration steps as
notebooks/EDA_and_Preprocessing.ipynb and notebooks/Modeling_and_Evaluation.ipynb,
but on the fake fixture CSV instead of the real one.

This is scaffolding, not a replacement for the real notebooks. Its only job
is to produce a scored_test.csv and a fixture model.pkl that the SQL scripts
and Streamlit dashboard can be built and tested against before the real
data is available.

Run:
    python build_fixture_pipeline.py
Writes:
    ../data/scored_test.csv
    ../models/fixture_calibrated_model.pkl
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

RAW_PATH = "../data/telco_churn_fixture.csv"

# ---- same steps as EDA_and_Preprocessing.ipynb ----
df = pd.read_csv(RAW_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

df["tenure_group"] = pd.cut(
    df["tenure"], bins=[0, 12, 24, 48, 60, 72],
    labels=["0-1y", "1-2yr", "2-4yr", "4-5yr", "5-6yr"],
)
df["charges_per_tenure"] = np.where(
    df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
)
service_cols = ["PhoneService", "OnlineSecurity", "OnlineBackup",
                 "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["num_services"] = df[service_cols].apply(lambda x: (x == "Yes").sum(), axis=1)
df["high_value"] = (
    (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75)) & (df["tenure"] > 24)
).astype(int)
df["high_risk"] = (
    (df["Contract"] == "Month-to-month") & (df["MonthlyCharges"] > df["MonthlyCharges"].median())
).astype(int)

# keep customerID and the raw (unscaled, unencoded) columns aside so we can
# rejoin them to the scored test set later - the real notebooks don't need
# this since they never look the raw values back up after encoding, but the
# dashboard and SQL layer need customer-readable columns to display.
raw_lookup = df[[
    "customerID", "gender", "Contract", "tenure", "MonthlyCharges",
    "TotalCharges", "InternetService", "PaymentMethod",
]].copy()

df_model = df.drop("customerID", axis=1)
cat_features = df_model.select_dtypes(include=["object", "category"]).columns
le = LabelEncoder()
for col in cat_features:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = X.select_dtypes(include=np.number).columns.tolist()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# ---- same steps as Modeling_and_Evaluation.ipynb (best model + calibration) ----
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf.fit(X_train_scaled, y_train)

calibrated_model = CalibratedClassifierCV(
    RandomForestClassifier(**rf.get_params()), cv=5, method="isotonic"
)
calibrated_model.fit(X_train_scaled, y_train)

y_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]

joblib.dump(calibrated_model, "../models/fixture_calibrated_model.pkl")

# ---- same steps as Customer_Segmentation.ipynb ----
scored = raw_lookup.loc[idx_test].reset_index(drop=True).copy()
scored["actual_churn"] = y_test.values
scored["churn_probability"] = y_prob
scored["risk_segment"] = pd.cut(
    y_prob, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["Very Low", "Low", "Medium", "High", "Very High"],
)
scored["value_segment"] = pd.qcut(
    scored["MonthlyCharges"], q=3, labels=["Low Value", "Mid Value", "High Value"]
)

scored.to_csv("../data/scored_test.csv", index=False)

print(f"train: {X_train.shape}, test: {X_test.shape}")
print(f"test churn rate: {y_test.mean() * 100:.2f}%")
print(f"scored_test.csv written, {len(scored)} rows")
print("\npriority matrix (customer count):")
print(scored.groupby(["value_segment", "risk_segment"], observed=True)["actual_churn"]
      .count().unstack())
