"""Unit tests for FastAPI health endpoint behavior."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_health_mock_mode_returns_200_with_expected_body(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Health endpoint should report mock mode readiness without real model load."""
	monkeypatch.setenv("INFERENCE_MODE", "mock")

	with TestClient(app) as client:
		response = client.get("/health")

	assert response.status_code == 200
	assert response.headers["content-type"].startswith("application/json")
	assert response.json() == {"status": "ok-mock", "model_loaded": False}


def test_health_reports_unavailable_when_real_mode_engine_missing(
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	"""Health should return 503 when real mode is active but engine reference is missing."""
	monkeypatch.setenv("INFERENCE_MODE", "mock")

	with TestClient(app) as client:
		client.app.state.inference_mode = "real"
		client.app.state.inference_engine = None
		response = client.get("/health")

	assert response.status_code == 503
	assert response.headers["content-type"].startswith("application/json")
	assert response.json() == {"status": "unavailable", "model_loaded": False}


def test_invalid_inference_mode_fails_lifespan_startup(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Unsupported INFERENCE_MODE values should fail startup with ValueError."""
	monkeypatch.setenv("INFERENCE_MODE", "invalid")

	with pytest.raises(ValueError, match="Unsupported INFERENCE_MODE value"):
		with TestClient(app):
			pass
