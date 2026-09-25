"""Output identity and validation for safe shard-level resume."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SIGNATURE_KEY = b"crispedit_mask_input_signature"


def signature(raw_path, sidecars, settings):
    raw = Path(raw_path)
    stat = raw.stat()
    payload = {"raw": [str(raw.resolve()), stat.st_size, stat.st_mtime_ns],
               "settings": settings, "sidecars": []}
    for path in sidecars:
        if path:
            p = Path(path)
            payload["sidecars"].append(hashlib.sha256(p.read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def signed_schema(schema, digest):
    return schema.with_metadata({**(schema.metadata or {}), SIGNATURE_KEY: digest.encode()})


def reusable_table(path, digest, schema, expected_indices):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        pf = pq.ParquetFile(path)
        if (pf.schema_arrow.metadata or {}).get(SIGNATURE_KEY) != digest.encode():
            return None
        if not set(schema.names).issubset(pf.schema_arrow.names):
            return None
        table = pf.read()
        if table["row_idx"].to_pylist() != list(expected_indices):
            return None
        if "ERROR" in table["qc_flag"].to_pylist():
            return None
        if "ground_parse_ok" in table.column_names:
            columns = ["qc_flag", "ground_parse_ok"]
            if "grounding_status" in table.column_names:
                columns.append("grounding_status")
            for row in table.select(columns).to_pylist():
                recoverable_parse_error = (
                    row.get("qc_flag") == "GROUND_FAIL"
                    and row.get("grounding_status") == "PARSE_ERROR"
                )
                if (row["qc_flag"] != "PREFILTER_SKIP"
                        and not row["ground_parse_ok"]
                        and not recoverable_parse_error):
                    return None
        return table
    except (OSError, pa.ArrowException, ValueError):
        return None


def check_output_location(output, *inputs):
    output = Path(output).resolve()
    for value in inputs:
        if value is None:
            continue
        path = Path(value).resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError(f"output must be separate from input: {output}, {path}")
