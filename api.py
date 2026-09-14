import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import onnxruntime as ort


MODEL_ID = os.getenv(
    "INTENT_MODEL_ID",
    "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
)
MODEL_DIR = Path(
    os.getenv(
        "INTENT_MODEL_DIR",
        str(Path("models") / "multilingual-MiniLMv2-L6-mnli-xnli"),
    )
)
HYPOTHESIS_TEMPLATE = "Maksud pertanyaan pengguna adalah: {}"
ENTAILMENT_INDEX = 0
MAX_LENGTH = 512


def ensure_minilm(model_id: str, dest: Path) -> Path:
    onnx_path = dest / "onnx" / "model.onnx"
    tokenizer_ok = (dest / "tokenizer.json").exists() or (
        dest / "sentencepiece.bpe.model"
    ).exists()
    if onnx_path.exists() and tokenizer_ok:
        return onnx_path

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Model NLI tidak lengkap di {dest}. Mengunduh {model_id}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
        allow_patterns=[
            "onnx/**",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ],
    )
    if not onnx_path.exists():
        raise RuntimeError(f"File ONNX tidak ditemukan: {onnx_path}")
    print(f"Model NLI tersimpan di {dest}")
    return onnx_path


onnx_model_path = ensure_minilm(MODEL_ID, MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
session = ort.InferenceSession(
    str(onnx_model_path),
    providers=["CPUExecutionProvider"],
)
ONNX_INPUT_NAMES = [item.name for item in session.get_inputs()]

app = FastAPI(
    title="Finance Intent Classifier",
    version="2.0.0",
    description="Local MiniLM NLI intent router untuk menentukan flow setelah user bertanya.",
)

INTENTS: Dict[str, str] = {
    "DIRECT_DATA": (
        "Pengguna hanya meminta data, angka, daftar, tabel, grafik, perbandingan, "
        "atau hasil perhitungan yang dapat langsung ditampilkan tanpa analisis naratif."
    ),
    "ANALYSIS": (
        "Pengguna meminta analisis, penjelasan, alasan, penyebab, evaluasi, insight, "
        "atau interpretasi terhadap suatu data."
    ),
    "PREDICTION": (
        "Pengguna meminta prediksi, proyeksi, perkiraan, estimasi masa depan, "
        "atau kemungkinan nilai di periode berikutnya."
    ),
}

POLICY = {
    "DIRECT_DATA": {
        "need_ai_after_query": False,
        "recommended_render": "auto",
    },
    "ANALYSIS": {
        "need_ai_after_query": True,
        "recommended_render": "chart_and_narrative",
    },
    "PREDICTION": {
        "need_ai_after_query": True,
        "recommended_render": "chart_and_narrative",
    },
}

MIN_SCORE = 0.50
MIN_MARGIN = 0.10


class ClassifyRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class ScoreItem(BaseModel):
    intent: str
    score: float


class ClassifyResponse(BaseModel):
    question: str
    intent: str
    confidence: float
    margin: float
    is_ambiguous: bool
    need_ai_after_query: bool | None
    recommended_render: str | None
    scores: List[ScoreItem]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def nli_logits(premise: str, hypothesis: str) -> np.ndarray:
    encoded = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )
    inputs = {}
    for name in ONNX_INPUT_NAMES:
        if name in encoded:
            inputs[name] = encoded[name]
        elif name == "token_type_ids":
            inputs[name] = np.zeros_like(encoded["input_ids"])
        else:
            raise RuntimeError(f"Input ONNX tidak dikenali: {name}")
    outputs = session.run(None, inputs)[0]
    return outputs[0]


def classify_question(question: str) -> dict:
    labels = list(INTENTS.keys())
    entailment_scores = []

    for intent in labels:
        hypothesis = HYPOTHESIS_TEMPLATE.format(INTENTS[intent])
        logits = nli_logits(question, hypothesis)
        entailment_scores.append(float(logits[ENTAILMENT_INDEX]))

    probs = softmax(np.array(entailment_scores, dtype=np.float64))
    ranked_scores = sorted(
        [
            {"intent": intent, "score": float(score)}
            for intent, score in zip(labels, probs)
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    top = ranked_scores[0]
    second = ranked_scores[1] if len(ranked_scores) > 1 else {"score": 0.0}
    confidence = top["score"]
    margin = confidence - second["score"]
    is_ambiguous = confidence < MIN_SCORE or margin < MIN_MARGIN

    if is_ambiguous:
        intent = "UNKNOWN"
        need_ai_after_query = None
        recommended_render = None
    else:
        intent = top["intent"]
        policy = POLICY[intent]
        need_ai_after_query = policy["need_ai_after_query"]
        recommended_render = policy["recommended_render"]

    return {
        "question": question,
        "intent": intent,
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "is_ambiguous": is_ambiguous,
        "need_ai_after_query": need_ai_after_query,
        "recommended_render": recommended_render,
        "scores": [
            {
                "intent": item["intent"],
                "score": round(item["score"], 4),
            }
            for item in ranked_scores
        ],
    }


@app.get("/")
def root():
    return {
        "service": "finance-intent-classifier",
        "status": "ok",
        "model": MODEL_ID,
        "intents": list(INTENTS.keys()),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": MODEL_ID,
        "model_dir": str(MODEL_DIR),
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    try:
        return classify_question(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Klasifikasi gagal: {exc}",
        ) from exc
