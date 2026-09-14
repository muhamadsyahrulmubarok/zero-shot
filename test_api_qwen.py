import json
import time

import requests

API_URL = "http://127.0.0.1:8000/narrate"

TEST_CASES = [
    {
       "name": "NPL vs CR",
        "question": "Kenapa NPL naik padahal CR membaik?",
        "data": {
            "cabang": "Cisalak",
            "periode": {
                "juli_2026": {
                    "npl": 2.75,
                    "cr": 91.20
                },
                "agustus_2026": {
                    "npl": 3.42,
                    "cr": 94.80
                }
            }
        }
    },
     {
        "name": "Analisis NPL dan PAR",
        "question": "Bagaimana kualitas kredit cabang Cisalak jika dilihat dari NPL dan PAR?",
        "data": {
            "tanggal_data": "2026-08-13",
            "cabang": "Cisalak",
            "npl": 3.42,
            "par": 5.81
        }
    },
    {
        "name": "Baki Debet",
        "question": "Berapa baki debet cabang Cisalak?",
        "data": {
            "tanggal_data": "2026-08-13",
            "cabang": "Cisalak",
            "baki_debet": 22080471551
        }
    },
    {
        "name": "FPD",
        "question": "Apakah FPD cabang Tegal membaik dari Juli ke Agustus?",
        "data": {
            "cabang": "Tegal",
            "juli_2026": {
                "fpd": 4.20
            },
            "agustus_2026": {
                "fpd": 3.10
            }
        }
    },
    {
        "name": "Metric tidak dikenal",
        "question": "Bagaimana performa XYZ cabang Cisalak?",
        "data": {
            "cabang": "Cisalak",
            "xyz": 82.5
        }
    }
]


def run_case(case):
    started = time.perf_counter()

    response = requests.post(
        API_URL,
        json={
            "question": case["question"],
            "data": case["data"],
            "max_tokens": 220,
            "temperature": 0.15,
        },
        timeout=180,
    )

    client_elapsed = time.perf_counter() - started
    response.raise_for_status()

    result = response.json()

    print("=" * 90)
    print("QUESTION")
    print(case["question"])
    print("\nDATA")
    print(json.dumps(case["data"], ensure_ascii=False, indent=2))
    print("\nANSWER")
    print(result["answer"])
    print("\nMETRICS")
    print(f"Server inference : {result['elapsed_seconds']} s")
    print(f"Client total     : {client_elapsed:.4f} s")
    print(f"Prompt tokens    : {result.get('prompt_tokens')}")
    print(f"Completion tokens: {result.get('completion_tokens')}")
    print(f"Total tokens     : {result.get('total_tokens')}")
    print()


if __name__ == "__main__":
    print("Testing local Qwen narrator API...\n")

    for case in TEST_CASES:
        try:
            run_case(case)
        except requests.ConnectionError:
            print(
                "Tidak bisa connect ke API.\n"
                "Jalankan terlebih dahulu:\n\n"
                "  uvicorn api:app --host 127.0.0.1 --port 8000\n"
            )
            break
        except requests.RequestException as exc:
            print(f"Request gagal: {exc}")
            break
