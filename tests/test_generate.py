"""Unit tests for FastAPI generate endpoint behavior."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_generate_valid_request_returns_200_and_mock_response(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Valid request should return deterministic mock response in mock mode."""
	monkeypatch.setenv("INFERENCE_MODE", "mock")

	payload = {
		"instruction": "Explain LoRA briefly.",
		"context": None,
		"max_new_tokens": 8,
	}

	with TestClient(app) as client:
		response = client.post("/generate", json=payload)

	assert response.status_code == 200
	assert response.headers["content-type"].startswith("application/json")
	assert response.json() == {"response": "Mock response: Explain LoRA briefly."}


def test_generate_with_context_keeps_deterministic_mock_contract(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Context should not change deterministic mock behavior beyond instruction echo."""
	monkeypatch.setenv("INFERENCE_MODE", "mock")

	payload = {
		"instruction": "Explain adapters.",
		"context": "Extra context goes here.",
		"max_new_tokens": 16,
	}

	with TestClient(app) as client:
		response = client.post("/generate", json=payload)

	assert response.status_code == 200
	assert response.headers["content-type"].startswith("application/json")
	assert response.json() == {"response": "Mock response: Explain adapters."}


def test_generate_whitespace_instruction_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Whitespace-only instructions should fail request validation."""
	monkeypatch.setenv("INFERENCE_MODE", "mock")

	payload = {"instruction": "   ", "context": None, "max_new_tokens": 16}

	with TestClient(app) as client:
		response = client.post("/generate", json=payload)

	assert response.status_code == 422


def test_generate_missing_instruction_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Instruction is required by the request schema."""
	monkeypatch.setenv("INFERENCE_MODE", "mock")

	payload = {"context": None, "max_new_tokens": 16}

	with TestClient(app) as client:
		response = client.post("/generate", json=payload)

	assert response.status_code == 422


def test_generate_max_new_tokens_below_min_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
	"""max_new_tokens < 1 should fail schema validation."""
	monkeypatch.setenv("INFERENCE_MODE", "mock")

	payload = {"instruction": "Hello", "context": None, "max_new_tokens": 0}

	with TestClient(app) as client:
		response = client.post("/generate", json=payload)

	assert response.status_code == 422


def test_generate_max_new_tokens_above_max_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
	"""max_new_tokens > 1024 should fail schema validation."""
	monkeypatch.setenv("INFERENCE_MODE", "mock")

	payload = {"instruction": "Hello", "context": None, "max_new_tokens": 1025}

	with TestClient(app) as client:
		response = client.post("/generate", json=payload)

	assert response.status_code == 422


@pytest.mark.parametrize("max_new_tokens", [1, 1024])
def test_generate_max_new_tokens_boundaries_accepted(
	monkeypatch: pytest.MonkeyPatch,
	max_new_tokens: int,
) -> None:
	"""Boundary values for max_new_tokens should be accepted."""
	monkeypatch.setenv("INFERENCE_MODE", "mock")

	payload = {
		"instruction": "Boundary test",
		"context": None,
		"max_new_tokens": max_new_tokens,
	}

	with TestClient(app) as client:
		response = client.post("/generate", json=payload)

	assert response.status_code == 200
	assert response.headers["content-type"].startswith("application/json")
	assert response.json() == {"response": "Mock response: Boundary test"}
