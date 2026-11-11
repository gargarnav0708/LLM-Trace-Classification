"""Input loading. Supports supplied load_data.iter_calls and direct Parquet."""
import json
from pathlib import Path

JSON_FIELDS = ("input_messages_json", "output_messages_json", "tool_definitions_json", "extra_json")

def _decode(v):
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v

def iter_calls(path, batch_size=16):
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError("Parquet input needs pyarrow; install requirements.txt") from e
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                for key in JSON_FIELDS:
                    if key in row:
                        row[key.removesuffix("_json")] = _decode(row.pop(key))
                yield row
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    for key in JSON_FIELDS:
                        if key in row:
                            row[key.removesuffix("_json")] = _decode(row.pop(key))
                    yield row
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            rows = json.load(f)
        yield from rows
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
