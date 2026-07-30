# GGUFy CPU quantization API

The API wraps the native GGUFy executable with path isolation, exact tensor
planning, output validation, and one CPU conversion subprocess per request.
It always produces safetensors.

## Development

Use Python 3.12 and `uv`:

```bash
uv venv --python 3.12
uv sync --all-groups
uv run uvicorn ggufy_api.main:app --reload
uv run pytest
```

The service defaults to `/data/input`, `/data/output`, and `/data/tmp`. For a
local checkout:

```bash
export GGUFY_DATA_ROOT="$PWD/data"
export GGUFY_BINARY="$PWD/zig-out/bin/ggufy"
uv run uvicorn ggufy_api.main:app --host 127.0.0.1 --port 8000
```

## Docker

The default image is CPU-only and runs as UID/GID `10001`.

```bash
mkdir -p input output tmp
docker build -t ggufy-api .
docker run --rm --user "$(id -u):$(id -g)" -p 8000:8000 \
  -v "$PWD/input:/data/input:ro" \
  -v "$PWD/output:/data/output" \
  -v "$PWD/tmp:/data/tmp" \
  ggufy-api
```

The user override keeps bind-mounted output files owned by the host user and
still runs the service without root privileges. `docker compose up --build`
applies the same UID/GID pattern; set `GGUFY_UID` and `GGUFY_GID` if the host
user is not `1000:1000`.

The old CLI-only image is still available as a build target:

```bash
docker build --target cli-runtime -t ggufy-cli .
```

## Convert with ordered tensor rules

Rules use first-match-wins semantics. `PRESERVE` retains the source dtype.
Unmatched tensors are preserved by default. The API resolves rules to exact
tensor names before starting GGUFy.

Default cluster types follow GGUFy's safety behavior: small tensors,
non-`.weight` tensors, token embeddings, and format-incompatible shapes remain
at source precision. Explicit rules and copied schemas are authoritative and
are therefore validated strictly rather than silently downgraded.

```bash
curl -sS http://localhost:8000/v1/conversions \
  -H 'content-type: application/json' \
  -d '{
    "input_path": "model.safetensors",
    "output_path": "model-mixed.safetensors",
    "default_type": "SCALED_F8_E4M3",
    "rules": [
      {
        "pattern": "*.attn*.weight",
        "match": "glob",
        "target_type": "INT8_CONVROT"
      },
      {
        "pattern": "*.norm*.weight",
        "match": "glob",
        "target_type": "PRESERVE"
      }
    ],
    "threads": 8
  }'
```

Supported matching modes are `exact`, `glob`, and `regex`. INT8 ConvRot rules
are rejected unless the tensor is two-dimensional and its final dimension is
divisible by 256.

## Copy a Hugging Face schema

This endpoint uses HTTP range requests. It fetches the safetensors header and,
when present, the small ComfyUI `.comfy_quant` marker payloads. Model weights
are not downloaded.

```bash
curl -sS http://localhost:8000/v1/schemas/huggingface \
  -H 'content-type: application/json' \
  -d '{
    "repo_id": "Comfy-Org/example-model",
    "filename": "diffusion_models/model_int8_convrot.safetensors",
    "revision": "main",
    "read_markers": true
  }' > schema.json
```

For gated repositories, set `HF_TOKEN` in the service environment. Tokens are
not accepted in request bodies.

Pass the returned schema directly to a conversion:

```bash
jq -n --slurpfile schema schema.json '{
  input_path: "source.safetensors",
  output_path: "copied-schema.safetensors",
  schema: $schema[0],
  schema_match: "exact",
  unmatched: "error"
}' |
curl -sS http://localhost:8000/v1/conversions \
  -H 'content-type: application/json' \
  --data-binary @-
```

`schema_match: "suffix"` supports sources with an additional namespace prefix.
Suffix matching still requires a unique match and exact shape equality.

## Resolve without converting

Use `POST /v1/plans/resolve` with the same policy fields to inspect every
resolved tensor type before spending CPU time.

Other endpoints:

- `GET /health`
- `POST /v1/schemas/local`
- `POST /v1/schemas/huggingface`
- `POST /v1/plans/resolve`
- `POST /v1/conversions`

Interactive OpenAPI documentation is available at `/docs`.

## Runtime configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GGUFY_DATA_ROOT` | `/data` | Parent data directory |
| `GGUFY_INPUT_ROOT` | `/data/input` | Only accepted input root |
| `GGUFY_OUTPUT_ROOT` | `/data/output` | Only accepted output root |
| `GGUFY_TEMP_ROOT` | `/data/tmp` | Temporary exact templates |
| `GGUFY_BINARY` | `ggufy` | Native executable path |
| `GGUFY_MAX_CONCURRENT_JOBS` | `1` | Concurrent conversion processes |
| `GGUFY_MAX_THREADS` | CPU count | Per-job thread ceiling |
| `HF_TOKEN` | unset | Optional gated Hugging Face access |

Keep `GGUFY_MAX_CONCURRENT_JOBS=1` unless the worker has enough RAM for
multiple simultaneous largest-tensor working sets.
