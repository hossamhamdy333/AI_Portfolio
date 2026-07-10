
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import yaml
import os

app = FastAPI(title="Sentiment Forge API", version="1.0.0")

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

tokenizer   = AutoTokenizer.from_pretrained("outputs/models/bert_onnx")
ort_session = ort.InferenceSession("outputs/models/bert_onnx/model.onnx")

LABEL_NAMES = config["data"]["label_names"]


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    text       : str
    label      : str
    label_id   : int
    confidence : float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text,
        return_tensors = "np",
        truncation     = True,
        max_length     = config["data"]["max_length"],
        padding        = "max_length"
    )

    logits = ort_session.run(
        None,
        {
            "input_ids"      : inputs["input_ids"].astype(np.int64),
            "attention_mask" : inputs["attention_mask"].astype(np.int64),
        }
    )[0]

    probs      = np.exp(logits) / np.exp(logits).sum()
    label_id   = int(np.argmax(probs))
    confidence = float(probs[0][label_id])

    return PredictResponse(
        text       = request.text,
        label      = LABEL_NAMES[label_id],
        label_id   = label_id,
        confidence = confidence
    )
