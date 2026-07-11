"""FastAPI application for Phase 7 inference serving."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from scripts.utils import get_logger

LOGGER = get_logger("app.main")
SUPPORTED_INFERENCE_MODES = {"real", "mock"}


def _resolve_inference_mode() -> str:
	"""Resolve and validate inference mode from environment.

	Returns:
		Normalized inference mode (`real` or `mock`).

	Raises:
		ValueError: If an unsupported mode value is provided.
	"""
	mode = os.getenv("INFERENCE_MODE", "real").strip().lower()
	if mode not in SUPPORTED_INFERENCE_MODES:
		raise ValueError(
			"Unsupported INFERENCE_MODE value "
			f"{mode!r}. Supported values: {', '.join(sorted(SUPPORTED_INFERENCE_MODES))}."
		)
	return mode


def _mock_generate_response(instruction: str, context: str | None = None, max_new_tokens: int = 256) -> str:
	"""Provide deterministic mock output for Docker integration tests."""
	_ = (context, max_new_tokens)
	return f"Mock response: {instruction}"


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Load the shared inference engine once and prepare concurrency guards."""
	mode = _resolve_inference_mode()
	LOGGER.info("Starting inference service with INFERENCE_MODE=%s", mode)
	app.state.inference_mode = mode
	app.state.inference_engine = None
	app.state.generate_callable = None
	app.state.generation_lock = asyncio.Lock()
	try:
		if mode == "real":
			from app.inference import InferenceEngine

			app.state.inference_engine = InferenceEngine()
			app.state.generate_callable = app.state.inference_engine.generate_response
			LOGGER.info("Inference engine loaded successfully")
		else:
			app.state.generate_callable = _mock_generate_response
			LOGGER.info("Mock inference mode enabled for integration testing")
		yield
	except Exception:
		LOGGER.exception("Failed to start inference service")
		raise
	finally:
		app.state.inference_engine = None
		app.state.generate_callable = None
		app.state.inference_mode = None
		app.state.generation_lock = None
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		LOGGER.info("Inference service shutdown complete")


app = FastAPI(
	title="LLM LoRA Inference API",
	description="FastAPI service for serving a fine-tuned LoRA language model.",
	version="1.0.0",
	lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
	"""Report whether the shared inference engine is available."""
	mode = getattr(request.app.state, "inference_mode", "real")
	if mode == "mock":
		return HealthResponse(status="ok-mock", model_loaded=False)

	engine = getattr(request.app.state, "inference_engine", None)
	if engine is None:
		payload = HealthResponse(status="unavailable", model_loaded=False)
		return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump())

	return HealthResponse(status="ok", model_loaded=True)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, payload: GenerateRequest) -> GenerateResponse:
	"""Generate model output for a validated instruction request."""
	generation_callable = getattr(request.app.state, "generate_callable", None)
	generation_lock = getattr(request.app.state, "generation_lock", None)

	if generation_callable is None or generation_lock is None:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Inference engine is unavailable",
		)

	try:
		async with generation_lock:
			response_text = await run_in_threadpool(
				generation_callable,
				payload.instruction,
				payload.context,
				payload.max_new_tokens,
			)
		return GenerateResponse(response=response_text)
	except Exception:
		LOGGER.exception("Generation failed")
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Generation failed",
		)
