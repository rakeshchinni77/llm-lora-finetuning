# LLM LoRA Fine-Tuning and Inference API

An end-to-end, production-oriented Large Language Model fine-tuning and serving pipeline built with Mistral-7B, LoRA/PEFT, Hugging Face Transformers, FastAPI, Docker, and pytest.

The project covers the complete lifecycle of an instruction-tuned language model:

**Data Preparation → LoRA Fine-Tuning → Evaluation → Inference Engine → FastAPI Serving → Docker Integration → Automated Testing**

The implementation is modular, configuration-driven, and designed so that training, evaluation, inference, API serving, containerization, and testing remain separate concerns.

---

## Features

- Instruction dataset preparation and prompt formatting
- Dataset validation, filtering, statistics, and train/validation output generation
- LoRA parameter-efficient fine-tuning
- 4-bit model loading for supported GPU environments
- Hugging Face Transformers and PEFT integration
- Experiment tracking support with Weights & Biases
- BLEU, ROUGE-L, and perplexity evaluation
- Comparison report generation
- Modular inference engine
- LoRA adapter loading and merging
- Configuration-driven text generation
- FastAPI inference service
- `GET /health` endpoint
- `POST /generate` endpoint
- Pydantic request validation
- FastAPI lifespan-based model initialization
- Explicit real and mock inference modes
- Docker Compose API integration
- Docker health checks
- Unit and API tests with pytest
- Deterministic mock inference for lightweight integration testing

---

## Architecture

```text
                         ┌───────────────────────┐
                         │   Instruction Dataset │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Data Preparation   │
                         │ scripts/prepare_data.py
                         └───────────┬───────────┘
                                     │
                         Train / Validation Data
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │     LoRA Training     │
                         │    Base Model + PEFT  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  Fine-Tuned Adapter   │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
         ┌───────────────────────┐       ┌───────────────────────┐
         │      Evaluation       │       │    Inference Engine   │
         │ BLEU / ROUGE-L / PPL  │       │ Model + Adapter Merge │
         └───────────────────────┘       └───────────┬───────────┘
                                                     │
                                                     ▼
                                         ┌───────────────────────┐
                                         │      FastAPI API      │
                                         │ /health   /generate   │
                                         └───────────┬───────────┘
                                                     │
                                                     ▼
                                         ┌───────────────────────┐
                                         │   Docker Integration  │
                                         │ Health + API Testing  │
                                         └───────────────────────┘
```

### Runtime Flow

```text
Client
  │
  ▼
POST /generate
  │
  ▼
Pydantic Validation
  │
  ▼
FastAPI Endpoint
  │
  ▼
Generation Lock
  │
  ▼
Inference Engine
  │
  ├── Build Training-Compatible Prompt
  ├── Tokenize Input
  ├── Generate Tokens
  ├── Decode Output
  └── Return Generated Response
  │
  ▼
JSON Response
```

---

## Project Structure

```text
llm-lora-finetuning/
│
├── app/
│   ├── inference.py
│   ├── loader.py
│   ├── main.py
│   ├── prompt_template.py
│   └── schemas.py
│
├── config/
│   └── generation_config.json
│
├── data/
│   └── processed/
│
├── models/
│   └── fine_tuned_adapter/
│
├── results/
│   ├── comparison.md
│   └── evaluation_metrics.json
│
├── scripts/
│   ├── evaluate_model.py
│   ├── prepare_data.py
│   ├── train.py
│   └── utils.py
│
├── tests/
│   ├── test_generate.py
│   ├── test_health.py
│   └── test_prepare_data.py
│
├── .dockerignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

Generated datasets, model artifacts, evaluation results, secrets, caches, and local runtime files may be excluded from Git or the Docker build context.

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Base Model | Mistral-7B |
| Model Framework | Hugging Face Transformers |
| Fine-Tuning | LoRA / PEFT |
| Quantization | BitsAndBytes |
| Dataset Processing | Hugging Face Datasets |
| Evaluation | BLEU, ROUGE-L, Perplexity |
| Experiment Tracking | Weights & Biases |
| API | FastAPI |
| Validation | Pydantic |
| ASGI Server | Uvicorn |
| Containerization | Docker and Docker Compose |
| Testing | pytest and FastAPI TestClient |

---

## Requirements

The project was developed and validated with Python 3.10 locally.

A CUDA-capable GPU environment is strongly recommended for training and full-model evaluation.

Real Mistral-7B inference requires substantial system memory. Hardware requirements depend on the loading configuration, quantization support, operating system, and container memory limits.

### Recommended Environment

- Python 3.10+
- Git
- pip
- Docker Desktop for container testing
- CUDA-capable GPU for training
- Sufficient disk space for the base model and adapter artifacts
- Sufficient RAM or VRAM for real inference

---

## Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd llm-lora-finetuning
```

### 2. Create a Virtual Environment

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file.

Windows:

```bat
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Configure `.env` with your own values.

```env
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
BASE_MODEL_ID=mistralai/Mistral-7B-v0.1
INFERENCE_MODE=real

HOST_BASE_MODEL_PATH=C:/absolute/path/to/mistral7b
HOST_ADAPTER_PATH=C:/absolute/path/to/llm-lora-finetuning/models/fine_tuned_adapter
```

Never commit `.env`, Hugging Face tokens, Weights & Biases API keys, or other secrets.

---

## Data Preparation

The data preparation pipeline is implemented in:

```text
scripts/prepare_data.py
```

Responsibilities include:

- Loading the source dataset
- Validating examples
- Filtering malformed or oversized records
- Formatting instruction examples
- Handling optional context
- Producing processed records
- Writing processed data files
- Reporting dataset statistics

### Prompt Format

Without context:

```text
### Instruction:
Explain LoRA.

### Response:
LoRA is...
```

With context:

```text
### Instruction:
Summarize the following text.

### Context:
Context text goes here.

### Response:
Generated response goes here.
```

Run data preparation using the command supported by the current script:

```bash
python scripts/prepare_data.py
```

Processed artifacts are written under the configured data output location.

---

## Training

Training is implemented separately from API serving and inference.

The training pipeline uses:

- Mistral-7B base model
- LoRA adapters
- PEFT
- Hugging Face Transformers
- Quantized model loading where supported
- Configuration-driven training
- Experiment tracking support

Run training in a GPU environment:

```bash
python scripts/train.py
```

Training produces the fine-tuned LoRA adapter under the configured model output directory.

Example:

```text
models/fine_tuned_adapter/
```

### Training Notes

- GPU execution is strongly recommended.
- Training a 7B parameter model directly on a typical CPU laptop is not practical.
- LoRA reduces the number of trainable parameters but does not eliminate base-model memory requirements.
- Keep the prompt format consistent across data preparation, training, evaluation, and inference.

---

## Evaluation

Evaluation is implemented in:

```text
scripts/evaluate_model.py
```

The evaluation pipeline:

1. Loads the validation dataset.
2. Loads the tokenizer.
3. Loads the base model.
4. Loads and merges the LoRA adapter.
5. Generates model predictions.
6. Computes evaluation metrics.
7. Writes machine-readable metrics.
8. Generates a human-readable comparison report.

Run:

```bash
python scripts/evaluate_model.py
```

### Evaluation Outputs

```text
results/evaluation_metrics.json
results/comparison.md
```

Metrics include:

- BLEU
- ROUGE-L
- Perplexity

Evaluation metrics should be interpreted together with qualitative inspection of generated responses. Lexical-overlap metrics alone do not fully measure instruction-following quality or factual correctness.

---

## Inference Engine

The inference engine is separated into three modules.

### `app/loader.py`

Responsible for:

- Loading environment variables
- Loading the tokenizer
- Loading the base model
- Loading the LoRA adapter
- Merging the adapter
- Returning the model and tokenizer

### `app/prompt_template.py`

Responsible only for constructing prompts that match the training format.

### `app/inference.py`

Responsible for:

- Loading the model through the loader
- Loading generation configuration
- Building prompts
- Tokenization
- Text generation
- Decoding
- Removing the prompt from generated output
- Returning the generated response

### Console Inference

Run:

```bash
python app/inference.py
```

For command-line options:

```bash
python app/inference.py --help
```

The console demo supports interactive instruction entry and exits with `exit` or `quit`.

Real inference was validated locally with the downloaded Mistral-7B base model and fine-tuned LoRA adapter.

---

## Generation Configuration

Generation settings are stored in:

```text
config/generation_config.json
```

The inference engine loads generation parameters from configuration rather than hardcoding them in endpoint code.

This makes generation behavior easier to reproduce and adjust independently of API implementation.

---

## FastAPI Backend

The FastAPI application is implemented in:

```text
app/main.py
```

Request and response schemas are implemented in:

```text
app/schemas.py
```

### Start the API

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The application loads the inference service during FastAPI lifespan startup.

### Swagger UI

Open:

```text
http://127.0.0.1:8000/docs
```

### ReDoc

Open:

```text
http://127.0.0.1:8000/redoc
```

---

## API Endpoints

### `GET /health`

Reports inference service availability.

Real mode example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

Mock mode example:

```json
{
  "status": "ok-mock",
  "model_loaded": false
}
```

Unavailable real-engine example:

```json
{
  "status": "unavailable",
  "model_loaded": false
}
```

### `POST /generate`

Example request:

```json
{
  "instruction": "Explain LoRA briefly.",
  "context": null,
  "max_new_tokens": 32
}
```

Example response shape:

```json
{
  "response": "Generated model response"
}
```

### Validation

The API rejects invalid requests, including:

- Missing instruction
- Empty or whitespace-only instruction
- `max_new_tokens` below `1`
- `max_new_tokens` above `1024`

Invalid request data returns HTTP `422`.

---

## API Testing with curl

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Generate Text

Windows Command Prompt:

```bat
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"instruction\":\"Explain LoRA briefly.\",\"context\":null,\"max_new_tokens\":32}"
```

Linux/macOS:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Explain LoRA briefly.","context":null,"max_new_tokens":32}'
```

### Validation Test

Windows Command Prompt:

```bat
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"instruction\":\"   \",\"context\":null,\"max_new_tokens\":32}"
```

Expected result: HTTP `422`.

---

## Inference Modes

The API supports two explicit inference modes.

### Real Mode

```env
INFERENCE_MODE=real
```

Real mode:

- Loads the base model
- Loads the LoRA adapter
- Merges the adapter
- Performs actual text generation
- Requires sufficient memory

### Mock Mode

```env
INFERENCE_MODE=mock
```

Mock mode:

- Does not load the base model
- Does not load the LoRA adapter
- Does not require model downloads
- Returns deterministic responses
- Supports API integration testing
- Supports Docker integration testing on constrained hardware

Example:

```text
Instruction:
Explain LoRA briefly.

Response:
Mock response: Explain LoRA briefly.
```

Mock mode is a testing mechanism and must not be represented as real model inference.

---

## Docker

The project provides:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

The Docker API service mounts the base model and LoRA adapter from the host instead of copying large model artifacts into the image.

### Configure Docker Paths

Set absolute host paths in `.env`.

Example:

```env
HOST_BASE_MODEL_PATH=C:/AI/Models/mistral7b/content/mistral7b
HOST_ADAPTER_PATH=C:/Users/username/llm-lora-finetuning/models/fine_tuned_adapter
```

### Validate Compose Configuration

```bash
docker compose config
```

### Build the API Image

```bash
docker compose build api
```

### Start the API

```bash
docker compose up api
```

### Check Container Status

```bash
docker compose ps
```

### Test Health

```bash
curl http://127.0.0.1:8000/health
```

### Test Generation

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"instruction\":\"Explain LoRA briefly.\",\"context\":null,\"max_new_tokens\":8}"
```

### Stop Containers

```bash
docker compose down
```

### Docker Validation Performed

The Docker API integration path was validated with explicit mock inference mode:

```text
Docker Image
    ↓
Container Startup
    ↓
FastAPI Lifespan
    ↓
Mock Inference Service
    ↓
GET /health → 200 OK
    ↓
POST /generate → 200 OK
    ↓
Invalid Request → 422
    ↓
Docker Health Check → healthy
    ↓
Graceful Shutdown
```

An attempted real Mistral-7B Docker startup on the development machine was terminated with exit code `137` and `OOMKilled=true` because Docker had approximately `7.6 GiB` of memory available.

This is a hardware/runtime memory limitation, not a successful real-model Docker inference validation.

---

## Testing

The project contains unit and API tests for:

```text
tests/test_prepare_data.py
tests/test_health.py
tests/test_generate.py
```

The tests cover:

- Prompt formatting with context
- Prompt formatting without context
- Invalid sample filtering
- Whitespace-only instruction handling
- Oversized sample filtering
- Non-string field handling
- JSON output schema
- Deterministic percentile calculation
- Mock-mode health behavior
- Unavailable real-engine health behavior
- Invalid inference mode startup
- Valid generation requests
- Context requests
- Request validation
- Token-limit validation
- Boundary values

### Run Test Collection

```bash
python -m pytest --collect-only -q
```

### Run Individual Test Files

```bash
python -m pytest tests/test_prepare_data.py -v
python -m pytest tests/test_health.py -v
python -m pytest tests/test_generate.py -v
```

### Run the Full Suite

```bash
python -m pytest -v
```

or:

```bash
python -m pytest -q
```

### Verified Test Result

```text
19 passed
```

The complete test suite runs in seconds and does not load Mistral-7B, download models, require Docker, or require a GPU.

---

## Screenshots

Add project screenshots under:

```text
docs/screenshots/
```

Recommended screenshots:

1. Data preparation completion
2. Training completion or training metrics
3. Evaluation completion
4. Evaluation metrics output
5. Local inference engine startup
6. FastAPI `/health` response
7. Swagger UI
8. Docker image build completion
9. Docker container healthy status
10. Docker `/generate` response
11. Full pytest result showing `19 passed`

Example README syntax:

```markdown
![Evaluation Results](docs/screenshots/evaluation-results.png)

![Swagger UI](docs/screenshots/swagger-ui.png)

![Docker Health](docs/screenshots/docker-health.png)

![Test Results](docs/screenshots/test-results.png)
```

Do not commit screenshots containing API keys, access tokens, `.env` contents, private account information, or other secrets.

---

## Common Issues

### 1. `bitsandbytes` / `Accelerate` Error

Example:

```text
ImportError: Using bitsandbytes quantization requires Accelerate
```

Ensure required dependencies are installed:

```bash
pip install accelerate bitsandbytes
```

On Windows CPU environments, BitsAndBytes GPU quantization may be unavailable. The loader should follow the project's supported CPU fallback behavior.

### 2. Base Model Download Is Very Slow

Mistral-7B model files are large.

Interrupted Hugging Face downloads may resume from cache. For unreliable connections, download the model once to a stable local directory and configure the project to load it locally.

### 3. Windows Hugging Face Symlink Warning

Windows may report that Hugging Face cache symlinks are unavailable.

The cache can still work but may consume additional disk space.

Enabling Windows Developer Mode or running with appropriate permissions can enable symlink support.

### 4. Docker Container Exits with Code 137

Check:

```bash
docker compose ps -a
```

and:

```bash
docker inspect <container-name>
```

Exit code `137` with `OOMKilled=true` indicates that the container exceeded available memory.

Possible solutions:

- Increase Docker Desktop memory allocation
- Use a machine with more RAM
- Use supported quantized inference
- Use a smaller model
- Deploy inference to a GPU-backed environment
- Use explicit mock mode only for integration testing

### 5. API Startup Takes a Long Time

In real mode, startup includes:

- Tokenizer loading
- Base-model loading
- LoRA adapter loading
- Adapter merging

Large-model startup can take minutes on CPU systems.

### 6. `/generate` Is Slow on CPU

CPU inference for a 7B model can be slow.

Use a small `max_new_tokens` value for initial testing and use GPU-backed infrastructure for practical serving.

### 7. HTTP 422 from `/generate`

Check that:

- `instruction` is present
- `instruction` is not empty or whitespace-only
- `max_new_tokens >= 1`
- `max_new_tokens <= 1024`
- The request body contains valid JSON

### 8. Tests Accidentally Load the Real Model

Run tests with the existing test suite.

API tests explicitly force:

```text
INFERENCE_MODE=mock
```

before FastAPI lifespan startup.

Do not create a global TestClient that starts the application in real mode.

---

## FAQ

### Why use LoRA instead of full fine-tuning?

LoRA trains a small number of additional parameters while keeping the base model weights frozen. This significantly reduces training memory and storage requirements compared with full fine-tuning.

### Why use 4-bit model loading?

Quantization reduces model memory usage and makes fine-tuning or inference more practical on constrained GPU hardware.

### Why merge the LoRA adapter for inference?

Merging simplifies inference by combining adapter updates with the base model representation used for generation.

### Why must inference prompts match the training format?

The model was fine-tuned on a specific instruction/context/response structure. Consistent formatting reduces train-serving skew.

### Why load the model during FastAPI lifespan startup?

Loading once during startup avoids reloading the large model for every request and allows the API to reuse one inference engine.

### Why use a generation lock?

Large-model generation is resource-intensive. Serializing access protects the shared model instance from uncontrolled concurrent generation requests in the current architecture.

### Why is there a mock inference mode?

Mock mode enables API, validation, lifespan, Docker, and health-check integration testing without requiring the real 7B model.

### Does mock mode prove the model works inside Docker?

No.

Mock mode validates the Docker-to-FastAPI-to-endpoint integration path. Real-model Docker inference requires enough memory and must be validated separately on suitable hardware.

### Was real model inference tested?

Yes. The Mistral-7B base model and LoRA adapter were loaded and merged successfully for local inference.

### Why are evaluation metrics relatively low?

BLEU and ROUGE primarily measure lexical overlap with reference answers. Open-ended generation can produce semantically related text with low lexical overlap.

Qualitative inspection is also necessary. In this project, evaluation samples showed that some model outputs were verbose, incorrect, or poorly aligned, indicating that future improvements to training data, hyperparameters, prompting, and evaluation methodology would be beneficial.

### Can this project be deployed to production as-is?

The project demonstrates a production-oriented architecture, but real production deployment would require additional work such as:

- GPU-backed serving infrastructure
- Authentication and authorization
- Rate limiting
- Request timeouts
- Observability and metrics
- Structured tracing
- Load testing
- Security hardening
- CI/CD
- Model versioning
- Deployment-specific secrets management

---

## Project Validation Summary

The following project stages were completed:

| Phase | Component | Status |
|---|---|---|
| Data Preparation | Dataset processing and formatting | Completed |
| Training | LoRA fine-tuning pipeline | Completed |
| Evaluation | BLEU, ROUGE-L, perplexity, comparison report | Completed |
| Inference | Local model loading, adapter merging, generation | Completed |
| FastAPI | Health and generation endpoints | Completed |
| Docker | API integration validated in mock mode | Completed |
| Testing | 19 automated tests passing | Completed |
| Documentation | Setup, architecture, usage, troubleshooting, FAQ | Completed |

---

## Future Improvements

Potential improvements include:

- Higher-quality instruction datasets
- Larger and more diverse training data
- Hyperparameter optimization
- Improved generation configuration
- Better semantic and task-specific evaluation metrics
- Streaming token generation
- Request batching
- Authentication
- Rate limiting
- Structured observability
- GPU-backed Docker deployment
- CI/CD pipelines
- Model registry and versioning
- Kubernetes deployment
- Load and performance testing

---

## Security

- Never commit `.env`.
- Never commit Hugging Face tokens.
- Never commit Weights & Biases API keys.
- Rotate any secret that is accidentally exposed.
- Keep large model artifacts outside Git.
- Use deployment-specific secret management for production environments.

---

## License

Add a license file before public distribution if the repository does not already contain one.

Also review and comply with the license and usage terms of the base model, training dataset, and third-party dependencies used by the project.

---

## Final Status

This repository implements an end-to-end LoRA fine-tuning workflow covering data preparation, model training, evaluation, local inference, FastAPI serving, Docker integration testing, and automated tests.

The project is designed to demonstrate practical ML engineering concerns including modularity, reproducibility, configuration management, large-model serving constraints, API validation, containerization, testing, and transparent documentation of hardware limitations.