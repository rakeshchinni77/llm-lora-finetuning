"""Pydantic schemas for the Phase 7 FastAPI inference API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
	"""Request body for text generation."""

	instruction: str
	context: str | None = None
	max_new_tokens: int = Field(default=256, ge=1, le=1024)

	@field_validator("instruction")
	@classmethod
	def validate_instruction(cls, value: str) -> str:
		"""Reject empty or whitespace-only instructions."""
		if not value.strip():
			raise ValueError("instruction must not be empty")
		return value


class GenerateResponse(BaseModel):
	"""Response body for text generation."""

	response: str


class HealthResponse(BaseModel):
	"""Response body for API health checks."""

	model_config = ConfigDict(protected_namespaces=())

	status: str
	model_loaded: bool
