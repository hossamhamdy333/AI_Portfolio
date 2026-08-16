"""Hosted MLflow tracking via DagsHub for rag_router.

Same pattern and setup steps as ../rag_chatbot/shared/tracking.py -- see
that file's docstring for the one-time DagsHub setup, not repeated here.
Not imported from there directly since this is a standalone project; see
metrics.py's docstring for why.
"""

import getpass
import os


def init_tracking(config: dict, experiment_name: str = None):
    import mlflow

    owner = config["dagshub"]["repo_owner"]
    repo = config["dagshub"]["repo_name"]

    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        token = getpass.getpass("DagsHub token: ")
        os.environ["DAGSHUB_TOKEN"] = token

    os.environ["MLFLOW_TRACKING_USERNAME"] = hossam3759180
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = experiment_name or config["mlflow"]["experiment_name"]
    mlflow.set_experiment(exp_name)

    print(f"MLflow tracking: {tracking_uri}  (experiment: {exp_name})")
    print(f"View runs at: https://dagshub.com/{owner}/{repo}/experiments")
    return exp_name
