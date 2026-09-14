from transformers import pipeline

MODEL_NAME = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"

classifier = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=-1  # CPU
)

INTENTS = [
    "menampilkan atau mengambil data",
    "menghitung jumlah total rata-rata atau persentase",
    "membandingkan beberapa data",
    "menjelaskan alasan penyebab atau melakukan analisis",
    "memprediksi atau memperkirakan kondisi di masa depan"
]


def classify_intent(question):
    result = classifier(
        question,
        candidate_labels=INTENTS,
        hypothesis_template="Pertanyaan pengguna meminta {}.",
        multi_label=False
    )

    return {
        "intent": result["labels"][0],
        "score": result["scores"][0]
    }


questions = [
    "Tampilkan NPL seluruh cabang bulan Agustus",
]

for question in questions:
    result = classify_intent(question)

    print("\nQUESTION :", question)
    print("INTENT   :", result["intent"])
    print("SCORE    :", round(result["score"], 4))