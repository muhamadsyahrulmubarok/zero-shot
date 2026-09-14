from api import classify_question

questions = [
    "Tampilkan NPL seluruh cabang bulan Agustus",
]

if __name__ == "__main__":
    for question in questions:
        result = classify_question(question)
        print("\nQUESTION :", question)
        print("INTENT   :", result["intent"])
        print("SCORE    :", result["confidence"])
        print("AMBIGUOUS:", result["is_ambiguous"])
        print("SCORES   :")
        for item in result["scores"]:
            print(f"  {item['intent']:<12} {item['score']}")
