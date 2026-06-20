#!/usr/bin/env python
"""Serve a trained Stage 2 model with a small FastAPI app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.answer_extraction import extract_final_answer
from src.models.stage2_load import load_model_and_tokenizer
from src.training.trl_dpo_utils import load_stage2_config
from src.utils.io import load_yaml

MODEL = None
TOKENIZER = None
DEVICE = None
APP_CONFIG = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_serve.yaml")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    return parser.parse_args()


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem briefly.\n\n"
        "End with the final answer exactly in this format:\n\n"
        "Answer: <number>\n\n"
        f"Question:\n{question}"
    )


def create_app(config_path: str, model_dir: str | None, max_new_tokens: int):
    from fastapi import FastAPI
    from pydantic import BaseModel

    class GenerateRequest(BaseModel):
        question: str
        max_new_tokens: int | None = None
        temperature: float = 0.0

    app = FastAPI(title="DPO CoT Compression Inference API")

    @app.on_event("startup")
    def load_model() -> None:
        global MODEL, TOKENIZER, DEVICE, APP_CONFIG
        import torch

        raw = load_yaml(config_path)
        if "experiment" in raw:
            cfg = load_stage2_config(config_path)
            resolved_model_dir = model_dir or cfg["paths"]["output_dir"]
            model_cfg = cfg["model"]
        else:
            cfg = raw
            model_cfg = raw["model"]
            resolved_model_dir = model_dir or model_cfg["model_dir"]
        APP_CONFIG = cfg
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL, TOKENIZER = load_model_and_tokenizer(resolved_model_dir, model_cfg)
        MODEL.to(DEVICE)
        MODEL.eval()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": MODEL is not None, "device": DEVICE}

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict:
        import torch

        prompt = build_prompt(req.question)
        inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
        gen_kwargs = {
            "max_new_tokens": req.max_new_tokens or max_new_tokens,
            "do_sample": req.temperature > 0,
            "temperature": req.temperature if req.temperature > 0 else None,
            "pad_token_id": TOKENIZER.eos_token_id,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        with torch.no_grad():
            generated = MODEL.generate(**inputs, **gen_kwargs)
        prediction = TOKENIZER.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return {
            "prediction": prediction,
            "pred_answer": extract_final_answer(prediction),
            "output_tokens": len(prediction.split()),
        }

    return app


def main() -> None:
    args = parse_args()
    import uvicorn

    raw = load_yaml(args.config)
    server_cfg = raw.get("server", {})
    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or int(server_cfg.get("port", 8000))
    max_new_tokens = args.max_new_tokens or int(server_cfg.get("max_new_tokens", 512))
    app = create_app(args.config, args.model_dir, max_new_tokens)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
