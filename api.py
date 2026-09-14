from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

MODEL_NAME = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

# Load sekali saat API start, bukan setiap request.
classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=-1,  # CPU
)

app = FastAPI(
    title="Finance Intent Classifier",
    version="1.0.0",
    description="Local zero-shot intent router untuk menentukan flow setelah user bertanya.",
)

# Intent sengaja dibuat sedikit dan berdasarkan flow aplikasi.
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

# Untuk mapping hasil intent ke flow aplikasi.
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

# Threshold awal. Jangan anggap angka ini final;
# nanti sebaiknya dikalibrasi menggunakan data pertanyaan nyata.
MIN_SCORE = 0.50

# Kalau dua label teratas terlalu dekat, anggap ambigu.
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


@app.get("/")
def root():
    return {
        "service": "finance-intent-classifier",
        "status": "ok",
        "model": MODEL_NAME,
        "intents": list(INTENTS.keys()),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    labels = list(INTENTS.values())

    result = classifier(
        request.question,
        candidate_labels=labels,
        hypothesis_template="Maksud pertanyaan pengguna adalah: {}",
        multi_label=False,
    )

    # Mapping label deskriptif kembali ke kode intent.
    label_to_intent = {description: intent for intent, description in INTENTS.items()}

    ranked_scores = []
    for label, score in zip(result["labels"], result["scores"]):
        ranked_scores.append(
            {
                "intent": label_to_intent[label],
                "score": float(score),
            }
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
        "question": request.question,
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
