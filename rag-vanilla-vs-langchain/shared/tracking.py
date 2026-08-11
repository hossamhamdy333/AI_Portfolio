"""Hosted MLflow tracking via DagsHub, instead of a local sqlite file.

The old setup (`mlflow.set_tracking_uri("sqlite:///mlflow.db")`) writes a
file that only means anything if you run `mlflow ui` on the same machine --
useless for anyone else looking at this repo, including future-you on a
different laptop. DagsHub hosts a real MLflow server per repo for free and
gives every run a URL, so results are actually link-shareable the same way
the GitHub repo itself is.

Setup (one-time, per DagsHub account):
  1. Create a free account at https://dagshub.com and connect this GitHub repo
     (DagsHub -> "Create" -> "Connect a repo" -> pick AI_Portfolio).
  2. Get a token: DagsHub -> your avatar -> Settings -> Tokens -> "Generate new token".
  3. Set configs/config.yaml's dagshub.repo_owner / repo_name to match.

Every notebook that logs to MLflow calls init_tracking(config) from this
module instead of calling mlflow.set_tracking_uri(...) directly -- one
place to change if the hosting choice ever changes again.
"""

import getpass
import os


def init_tracking(config: dict, experiment_name: str = None):
    """Point MLflow at the DagsHub-hosted tracking server for this repo.

    Prompts for a DagsHub token if DAGSHUB_TOKEN isn't already set in the
    environment. Returns the experiment name actually set, so callers can
    log it (mlflow.log_param("experiment", ...)) if useful.
    """
    import mlflow

    owner = config["dagshub"]["repo_owner"]
    repo = config["dagshub"]["repo_name"]

    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        token = getpass.getpass("DagsHub token: ")
        os.environ["DAGSHUB_TOKEN"] = token

    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    tracking_uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = experiment_name or config["mlflow"]["experiment_name"]
    mlflow.set_experiment(exp_name)

    print(f"MLflow tracking: {tracking_uri}  (experiment: {exp_name})")
    print(f"View runs at: https://dagshub.com/{owner}/{repo}/experiments")
    return exp_name
