"""FastAPI application for Phase 7 inference serving."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.inference import InferenceEngine
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from scripts.utils import get_logger

LOGGER = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Load the shared inference engine once and prepare concurrency guards."""
	LOGGER.info("Starting inference engine loading")
	app.state.inference_engine = None
	app.state.generation_lock = asyncio.Lock()
	try:
		app.state.inference_engine = InferenceEngine()
		LOGGER.info("Inference engine loaded successfully")
		yield
	except Exception:
		LOGGER.exception("Failed to start inference engine")
		raise
	finally:
		app.state.inference_engine = None
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
	engine = getattr(request.app.state, "inference_engine", None)
	if engine is None:
		payload = HealthResponse(status="unavailable", model_loaded=False)
		return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump())

	return HealthResponse(status="ok", model_loaded=True)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, payload: GenerateRequest) -> GenerateResponse:
	"""Generate model output for a validated instruction request."""
	engine = getattr(request.app.state, "inference_engine", None)
	generation_lock = getattr(request.app.state, "generation_lock", None)

	if engine is None or generation_lock is None:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail="Inference engine is unavailable",
		)

	try:
		async with generation_lock:
			response_text = await run_in_threadpool(
				engine.generate_response,
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
