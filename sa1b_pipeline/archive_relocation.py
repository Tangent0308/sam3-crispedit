"""Read archived path references without mutating frozen run contracts."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MAP = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/"
    "SAMTokEdit/data_pipeline/path_map.json"
)


class ArchivePaths:
    def __init__(self, mapping):
        self.rules = sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)
        for old, new in self.rules:
            if not Path(old).is_absolute() or not Path(new).is_absolute():
                raise ValueError("archive mapping requires absolute paths")

    @classmethod
    def load(cls, path=DEFAULT_MAP):
        return cls(json.loads(Path(path).read_text()))

    def path(self, value):
        if not isinstance(value, str) or not value.startswith("/"):
            return value
        for old, new in self.rules:
            if value == old or value.startswith(old + "/"):
                return new + value[len(old):]
        return value

    def record(self, value):
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                relocated = self.path(key)
                if relocated in out:
                    raise ValueError("path relocation would collide dictionary keys")
                out[relocated] = self.record(item)
            return out
        if isinstance(value, list):
            return [self.record(item) for item in value]
        return self.path(value)

    def materialize(self, source, output, *, jsonl=False):
        """Create a separate read/visualization view, never an in-place checkpoint."""
        source, output = Path(self.path(str(source))), Path(output)
        if source.resolve() == output.resolve():
            raise ValueError("cannot overwrite an archived original")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects both original records and existing views.
        with source.open() as inp, output.open("x") as out:
            if jsonl:
                for line in inp:
                    if line.strip():
                        out.write(json.dumps(self.record(json.loads(line)), ensure_ascii=False) + "\n")
            else:
                json.dump(self.record(json.load(inp)), out, ensure_ascii=False, indent=2)
                out.write("\n")
        return output
