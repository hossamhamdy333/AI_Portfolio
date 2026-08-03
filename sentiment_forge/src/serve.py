from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import yaml

app = FastAPI(title="Sentiment Forge API", version="1.0.0")

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

MODEL_REPO = config["serving"]["model_repo"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
onnx_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.onnx")
ort_session = ort.InferenceSession(onnx_path)

LABEL_NAMES = config["data"]["label_names"]


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text: str
    label: str
    label_id: int
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="np",
        truncation=True,
        max_length=config["data"]["max_length"],
        padding="max_length"
    )

    logits = ort_session.run(
        None,
        {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
    )[0]

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    label_id = int(np.argmax(probs[0]))
    confidence = float(probs[0][label_id])

    return PredictResponse(
        text=request.text,
        label=LABEL_NAMES[label_id],
        label_id=label_id,
        confidence=confidence
    )
