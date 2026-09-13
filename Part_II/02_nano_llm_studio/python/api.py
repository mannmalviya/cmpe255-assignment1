import json
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import ModelConfig, PROFILES
from .model import PocketTransformer
from .tokenizer import ByteTokenizer
from .train import ARTIFACTS, select_device, train


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=800)
    system: str = Field(default="You are PocketLM, a concise and friendly assistant.", max_length=400)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=262)
    max_new_tokens: int = Field(default=96, ge=1, le=192)


class TokenizeRequest(BaseModel):
    text: str = Field(max_length=1000)


class TrainRequest(BaseModel):
    profile: str = Field(default="nano", pattern="^(micro|nano)$")
    epochs: int = Field(default=20, ge=1, le=100)
    batch_size: int = Field(default=16, ge=1, le=128)
    learning_rate: float = Field(default=3e-4, gt=0, le=0.1)


class Runtime:
    def __init__(self):
        self.tokenizer = ByteTokenizer()
        self.device = select_device()
        self.model: PocketTransformer | None = None
        self.training_state: dict = {"status": "idle"}
        self.lock = threading.Lock()

    def load(self) -> bool:
        path = ARTIFACTS / "pocketlm.pt"
        if not path.exists():
            return False
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        config = ModelConfig(**checkpoint["config"])
        model = PocketTransformer(config)
        model.load_state_dict(checkpoint["model"])
        self.model = model.to(self.device).eval()
        return True

    def metrics(self) -> dict | None:
        path = ARTIFACTS / "metrics.json"
        return json.loads(path.read_text()) if path.exists() else None


runtime = Runtime()
runtime.load()
app = FastAPI(title="PocketLM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    model = runtime.model
    return {
        "status": "ready" if model else "untrained",
        "device": str(runtime.device),
        "model_loaded": model is not None,
        "parameters": model.parameter_count if model else 0,
        "context_length": model.config.max_seq_len if model else PROFILES["nano"].max_seq_len,
        "vocab_size": runtime.tokenizer.vocab_size,
        "architecture": ["RoPE", "RMSNorm", "SwiGLU", "GQA", "KV cache", "SDPA"],
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    if runtime.model is None and not runtime.load():
        raise HTTPException(status_code=409, detail="No checkpoint yet. Start training from the Admin dashboard.")
    assert runtime.model is not None
    prompt = runtime.tokenizer.format_chat(request.system, request.message)
    started = time.perf_counter()
    generated = runtime.model.generate(
        prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )
    elapsed = max(time.perf_counter() - started, 1e-6)
    return {
        "reply": runtime.tokenizer.decode(generated).strip(),
        "tokens_generated": len(generated),
        "tokens_per_second": round(len(generated) / elapsed, 1),
        "latency_ms": round(elapsed * 1000, 1),
    }


@app.post("/tokenize")
def tokenize(request: TokenizeRequest) -> dict:
    tokens = runtime.tokenizer.inspect(request.text)
    return {"tokens": tokens, "count": len(tokens), "characters": len(request.text)}


@app.get("/metrics")
def metrics() -> dict:
    return runtime.metrics() or {"status": "untrained", "history": []}


@app.get("/training")
def training_status() -> dict:
    return runtime.training_state


@app.post("/training", status_code=202)
def start_training(request: TrainRequest) -> dict:
    with runtime.lock:
        if runtime.training_state.get("status") == "training":
            raise HTTPException(status_code=409, detail="Training is already running.")
        runtime.training_state = {"status": "queued", "epoch": 0, "epochs": request.epochs}

    def run() -> None:
        try:
            runtime.training_state = {"status": "training", "epoch": 0, "epochs": request.epochs}
            result = train(
                request.profile,
                request.epochs,
                request.batch_size,
                request.learning_rate,
                progress=lambda state: setattr(runtime, "training_state", state),
            )
            runtime.load()
            runtime.training_state = {"status": "ready", "metrics": result}
        except Exception as error:
            runtime.training_state = {"status": "failed", "error": str(error)}

    threading.Thread(target=run, daemon=True).start()
    return runtime.training_state


@app.get("/crisp-dm")
def crisp_dm() -> dict:
    trained = runtime.model is not None
    return {
        "phases": [
            {"name": "Business understanding", "status": "complete", "detail": "Local, inspectable chatbot for learning."},
            {"name": "Data understanding", "status": "complete", "detail": "Curated conversational JSONL with a recorded hash."},
            {"name": "Data preparation", "status": "complete", "detail": "UTF-8 bytes, chat roles, assistant-only loss masks."},
            {"name": "Modeling", "status": "complete" if trained else "ready", "detail": "Laptop-scale decoder Transformer."},
            {"name": "Evaluation", "status": "complete" if runtime.metrics() else "waiting", "detail": "Held-out loss and perplexity."},
            {"name": "Deployment", "status": "complete" if trained else "waiting", "detail": "FastAPI inference plus Next.js operations UI."},
        ]
    }
