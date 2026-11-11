# Hyde Trace Mining

Hyde Trace Mining analyzes records of LLM calls and groups them into user sessions. It builds structural fingerprints for recurring interaction patterns, reports token usage with its provenance, and can optionally use the Hyde GLM API to discover a business taxonomy and label sessions.

The pipeline produces session- and call-level assignments, taxonomy and quality diagnostics, validation reports, token analytics, and an HTML hierarchy view. Without live GLM mode it runs locally and makes no API calls. With live GLM mode, it sends compressed session summaries to the configured inference endpoint; do not enable it for data you are not authorized to share.

## What it does

- Reads Parquet, JSON, or JSONL call records and reconstructs sessions from their trace and call IDs.
- Summarizes user intent and conversation context, then fingerprints sessions by structure for archetype analysis.
- Separates recorded, estimated, and unknown token usage rather than treating missing usage as zero.
- Optionally discovers a taxonomy and annotates training, holdout, and selected residual sessions using the GLM API.
- Can replay saved taxonomy and annotation checkpoints without making API calls.
- Uses local classification and structural propagation only as suggestions; these remain marked `needs_review` where appropriate. Validation metrics compare against GLM labels, not independently verified ground truth.

## Requirements

- Python 3.11 or newer
- Dependencies listed in `v07/requirements.txt`

## Install

Run these commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r v07/requirements.txt
```

## Run locally

```sh
python -m v07.main \
  --input /path/to/traces.parquet \
  --output output
```

The input can also be `.json`, `.jsonl`, or `.ndjson`. Each call record needs a trace/session ID (`trace_id` or `session_id`) and a call ID (`span_id` or `call_id`). Calls for a session must be grouped together in the input; interleaved sessions are rejected.

The default mode makes no network requests. Outputs are written to the selected directory. If expected output files already exist, choose another directory or pass `--overwrite` to replace them.

## Optional live GLM mode

Set an API key in your environment, then enable `--live-glm`:

```sh
export HYDE_API_KEY="your-authorized-api-key"
python -m v07.main \
  --input /path/to/traces.parquet \
  --output output \
  --live-glm
```

The default model is `hyde/glm-5.3-flash` at `https://inference.hyde.tech/v1`. You can configure the API with `HYDE_API_BASE` and `HYDE_MODEL`. `HYDE_ANNOTATION_CUTOFF_MINUTES` sets the annotation time limit (default: 64 minutes). The key can also be supplied as `OPENAI_API_KEY` if `HYDE_API_KEY` is unset. Never commit API keys, source traces, or private annotation data.

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--budget` | `300` | Maximum selected direct training sessions |
| `--residual-budget` | `10000` | Maximum residual sessions eligible for live annotation |
| `--max-archetypes` | `30` | Maximum archetypes selected for analysis |
| `--taxonomy-file PATH` | — | Reuse a saved taxonomy |
| `--reuse-annotations-from DIR` | — | Replay training and validation annotation checkpoints from a directory |
| `--overwrite` | off | Replace existing expected output files |

For annotation replay, pass both `--taxonomy-file` and `--reuse-annotations-from`.

## Generated files

- `results.json` — run summary, validation summary, and token analytics
- `taxonomy.json` and `taxonomy_audit.json` — discovered/reused taxonomy and quality flags
- `session_assignments.json` and `trace_assignments.json` — session- and call-level labels
- `validation_report.json` and `diagnostic_report.json` — validation metrics and archetype diagnostics
- `hierarchy.html` — browsable taxonomy and assignment visualization
- `insights.json` — reserved for independently validated business insights; currently reports that none are included

## Tests

Install pytest if it is not already available, then run from the repository root:

```sh
python -m pip install pytest
python -m pytest
```
