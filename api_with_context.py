import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from llama_cpp import Llama


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.getenv(
    "QWEN_MODEL_PATH",
    str(Path("models") / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
)
MODEL_URL = os.getenv(
    "QWEN_MODEL_URL",
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/"
    "main/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true",
)

N_CTX = int(os.getenv("QWEN_N_CTX", "2048"))
N_THREADS = int(
    os.getenv("QWEN_N_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
)
N_BATCH = int(os.getenv("QWEN_N_BATCH", "256"))
# -1 = offload semua layer ke GPU jika llama-cpp punya CUDA/Vulkan.
# Build CPU-only akan mengabaikan nilai ini.
N_GPU_LAYERS = int(os.getenv("QWEN_N_GPU_LAYERS", "-1"))


def ensure_model(path: str, url: str) -> None:
    dest = Path(path)
    if dest.exists() and dest.stat().st_size > 0:
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Model tidak ditemukan di {dest}. Mengunduh dari Hugging Face...")

    req = urllib.request.Request(url, headers={"User-Agent": "zero-shot-api"})
    try:
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as handle:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total else None
            downloaded = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = downloaded * 100 / total_bytes
                    print(
                        f"\rMengunduh model: {pct:.1f}% "
                        f"({downloaded}/{total_bytes} bytes)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rMengunduh model: {downloaded} bytes",
                        end="",
                        flush=True,
                    )
        print()
        tmp.replace(dest)
        print(f"Model tersimpan di {dest}")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Gagal mengunduh model dari {url}: {exc}") from exc


ensure_model(MODEL_PATH, MODEL_URL)

# Model hanya di-load sekali saat API start.
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    n_gpu_layers=N_GPU_LAYERS,
    chat_format="chatml",
    verbose=False,
)


app = FastAPI(
    title="Local Finance Narrator - Qwen2.5",
    version="1.2.0",
)


# ============================================================
# BANKING KNOWLEDGE / CONTEXT
# ============================================================
#
# PENTING:
# Ini hanya CONTOH struktur.
# Ganti definition / formula / interpretation sesuai definisi
# resmi di perusahaanmu.
#
# Context ini bersifat lokal dan tidak perlu dikirim ke
# OpenAI / Claude.
# ============================================================

BANKING_CONTEXT = {
    "NPL": {
        "name": "Non Performing Loan",
        "definition": (
            "Rasio kredit bermasalah terhadap total kredit. "
            "Semakin tinggi NPL, secara umum kualitas portofolio kredit semakin buruk."
        ),
        "formula": "(kredit bermasalah / total kredit) x 100%",
        "good_direction": "lower",
        "notes": (
            "Gunakan definisi kolektibilitas dan cut-off sesuai aturan internal perusahaan."
        ),
    },

    "PAR": {
        "name": "Portfolio at Risk",
        "definition": (
            "Persentase portofolio kredit yang memiliki tunggakan sesuai bucket "
            "hari keterlambatan yang ditentukan."
        ),
        "formula": "(outstanding kredit berisiko / total outstanding) x 100%",
        "good_direction": "lower",
        "notes": (
            "Bucket hari keterlambatan harus mengikuti definisi internal perusahaan."
        ),
    },

    "FPD": {
        "name": "First Payment Default",
        "definition": (
            "Indikator kegagalan atau keterlambatan pembayaran pada cicilan awal "
            "sesuai definisi perusahaan."
        ),
        "good_direction": "lower",
        "notes": (
            "Definisi periode cicilan dan toleransi keterlambatan harus mengikuti "
            "business rule perusahaan."
        ),
    },

    "CR": {
        "name": "Collection Rate",
        "definition": (
            "Rasio realisasi pembayaran atau penagihan dibandingkan kewajiban "
            "yang seharusnya ditagih pada periode tertentu."
        ),
        "formula": "(realisasi collection / target atau kewajiban collection) x 100%",
        "good_direction": "higher",
        "notes": (
            "Formula CR dapat berbeda antar perusahaan, jadi gunakan formula internal."
        ),
    },

    "CAR": {
        "name": "Capital Adequacy Ratio",
        "definition": (
            "Rasio kecukupan modal untuk menanggung risiko atas aset yang dimiliki."
        ),
        "good_direction": "higher",
        "notes": (
            "Interpretasi harus mempertimbangkan ketentuan regulator dan kebijakan internal."
        ),
    },

    "BD": {
        "name": "Baki Debet",
        "definition": (
            "Outstanding pokok kredit yang masih menjadi kewajiban debitur pada "
            "tanggal data tertentu."
        ),
        "good_direction": "context_dependent",
    },

    "OTP": {
        "name": "OTP",
        "definition": (
            "CONTOH SAJA. Ganti dengan kepanjangan dan definisi OTP yang benar "
            "sesuai metric internal perusahaan."
        ),
        "good_direction": "context_dependent",
        "notes": (
            "Jangan mengasumsikan OTP berarti One-Time Password dalam konteks finance."
        ),
    },
}


# Alias membantu jika user menyebut nama panjang / variasi umum.
METRIC_ALIASES = {
    "NON PERFORMING LOAN": "NPL",
    "NON-PERFORMING LOAN": "NPL",
    "PORTFOLIO AT RISK": "PAR",
    "FIRST PAYMENT DEFAULT": "FPD",
    "COLLECTION RATE": "CR",
    "CAPITAL ADEQUACY RATIO": "CAR",
    "BAKI DEBET": "BD",
}


def get_relevant_banking_context(question: str, data: Any) -> Dict[str, Any]:
    """
    Ambil hanya metric yang relevan dari pertanyaan atau key data.

    Tujuannya supaya context prompt tidak terlalu panjang.
    """
    text_parts = [question]

    try:
        text_parts.append(json.dumps(data, ensure_ascii=False))
    except Exception:
        text_parts.append(str(data))

    searchable = " ".join(text_parts).upper()

    found = set()

    # Exact metric code, mis. NPL, PAR, FPD.
    for metric in BANKING_CONTEXT.keys():
        pattern = rf"\b{re.escape(metric.upper())}\b"
        if re.search(pattern, searchable):
            found.add(metric)

    # Alias, mis. "baki debet" -> BD.
    for alias, metric in METRIC_ALIASES.items():
        if alias in searchable:
            found.add(metric)

    return {
        metric: BANKING_CONTEXT[metric]
        for metric in sorted(found)
    }


# ============================================================
# REQUEST / RESPONSE
# ============================================================

class NarrateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    data: Any
    max_tokens: int = Field(default=120, ge=32, le=400)
    temperature: float = Field(default=0.10, ge=0.0, le=1.0)


class NarrateResponse(BaseModel):
    answer: str
    used_context: Dict[str, Any]
    elapsed_seconds: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
Anda adalah financial data narrator internal.

ATURAN WAJIB:
1. Jawab HANYA berdasarkan QUESTION, BANKING CONTEXT, dan DATA.
2. BANKING CONTEXT adalah definisi metric resmi yang harus diprioritaskan.
3. Jangan membuat, menebak, atau menambahkan angka/fakta yang tidak ada.
4. Jangan mengubah angka yang ada di DATA.
5. Jangan menyimpulkan penyebab jika DATA tidak cukup untuk membuktikannya.
6. Jangan membuat prediksi kecuali pertanyaan dan data memang mendukung prediksi.
7. Jangan menganggap singkatan memiliki arti umum jika BANKING CONTEXT memberinya arti khusus.
8. Jika pertanyaan hanya meminta angka, jawab angka utama terlebih dahulu.
9. Gunakan Bahasa Indonesia yang singkat, jelas, dan profesional.
10. Maksimal 2 paragraf pendek.
""".strip()


def compact_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def remove_thinking(text: str) -> str:
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "service": "local-finance-narrator",
        "status": "ok",
        "available_metrics": list(BANKING_CONTEXT.keys()),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
    }


@app.post("/narrate", response_model=NarrateResponse)
def narrate(request: NarrateRequest):
    banking_context = get_relevant_banking_context(
        request.question,
        request.data,
    )

    context_text = compact_json(banking_context)
    data_text = compact_json(request.data)

    user_prompt = f"""
QUESTION:
{request.question}

BANKING CONTEXT:
{context_text}

DATA:
{data_text}

Buat jawaban untuk user berdasarkan context dan data di atas saja.
""".strip()

    started = time.perf_counter()

    try:
        result = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.85,
            repeat_penalty=1.05,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference gagal: {exc}",
        ) from exc

    elapsed = time.perf_counter() - started

    answer = result["choices"][0]["message"]["content"] or ""
    answer = remove_thinking(answer)

    usage = result.get("usage", {})

    return NarrateResponse(
        answer=answer,
        used_context=banking_context,
        elapsed_seconds=round(elapsed, 4),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )
