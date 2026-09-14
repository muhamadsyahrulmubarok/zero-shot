import time

import requests

API_URL = "http://127.0.0.1:8000/classify"

QUESTIONS = [
    "Tampilkan NPL seluruh cabang bulan Agustus",
]


def test_question(question: str):
    started = time.perf_counter()
    response = requests.post(
        API_URL,
        json={"question": question},
        timeout=120,
    )
    elapsed = time.perf_counter() - started

    response.raise_for_status()
    data = response.json()

    print("=" * 80)
    print("QUESTION :", data["question"])
    print("INTENT   :", data["intent"])
    print("CONF     :", data["confidence"])
    print("MARGIN   :", data["margin"])
    print("AMBIGUOUS:", data["is_ambiguous"])
    print("NEED AI  :", data["need_ai_after_query"])
    print("RENDER   :", data["recommended_render"])
    print("TIME     :", f"{elapsed:.3f}s")

    print("\nALL SCORES:")
    for item in data["scores"]:
        print(f"  {item['intent']:<12} {item['score']}")


if __name__ == "__main__":
    for question in QUESTIONS:
        try:
            test_question(question)
        except requests.RequestException as exc:
            print(f"ERROR: {exc}")
            print(
                "\nPastikan API sudah berjalan dengan command:\n"
                "uvicorn api:app --reload"
            )
            break
