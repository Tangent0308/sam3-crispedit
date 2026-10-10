"""Encode SA-1B masks with the official SAMTok VQ-SAM2 tokenizer.

The tokenizer implementation is intentionally loaded from the pinned official
Sa2VA checkout.  The 871 MB checkpoint is the Qwen3-VL-8B-SAMTok release and
must not be replaced by a coordinate or bbox heuristic: its two residual
codes are the supervision consumed by the SAMTok text encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _official_imports(repo_root: Path):
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from projects.samtok.models import DirectResize, SAM2Config, VQ_SAM2, VQ_SAM2Config
    except ImportError as exc:  # pragma: no cover - depends on the isolated runtime
        raise RuntimeError(
            "SAMTok dependencies are unavailable. Use scripts/setup_sa1b_samtok.sh "
            "or install the official Sa2VA projects/samtok environment."
        ) from exc
    return DirectResize, SAM2Config, VQ_SAM2, VQ_SAM2Config


def _sha256(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _mask_and_box(mask_path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, torch.Tensor]:
    width, height = image_size
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask.shape != (height, width):
        raise ValueError(f"mask {mask_path} has shape {mask.shape}, expected {(height, width)}")
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError(f"mask {mask_path} is empty")
    box = torch.tensor(
        [[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]],
        dtype=torch.float32,
    )
    box /= torch.tensor([width, height, width, height], dtype=torch.float32)
    return mask.astype(np.uint8), box


class SAMTokEncoder:
    """One official Qwen3-VL SAMTok encoder resident on a CUDA device."""

    def __init__(self, model_dir: Path, repo_root: Path, device: str = "cuda:0") -> None:
        DirectResize, SAM2Config, VQ_SAM2, VQ_SAM2Config = _official_imports(repo_root)
        self.model_dir = model_dir.resolve()
        self.repo_root = repo_root.resolve()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SAMTok encoding requires a CUDA device")
        weights = self.model_dir / "mask_tokenizer_256x2.pth"
        sam_weights = self.model_dir / "sam2.1_hiera_large.pt"
        if not weights.is_file() or not sam_weights.is_file():
            raise FileNotFoundError(f"expected SAMTok weights in {self.model_dir}")
        config = VQ_SAM2Config(
            sam2_config=SAM2Config(ckpt_path=str(sam_weights)),
            codebook_size=256,
            codebook_depth=2,
            shared_codebook=False,
            latent_dim=256,
        )
        self.model = VQ_SAM2(config).to(self.device).eval()
        state = torch.load(weights, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state)
        del state
        self.preprocessor = DirectResize(1024)
        self.weights_sha256 = _sha256(weights)

    @torch.inference_mode()
    def encode(self, image_path: Path, mask_path: Path) -> tuple[str, list[int]]:
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image)
        mask, box = _mask_and_box(mask_path, image.size)
        resized = self.preprocessor.apply_image(array)
        pixels = torch.from_numpy(resized).permute(2, 0, 1).contiguous().unsqueeze(0)
        gt_mask = torch.from_numpy(mask).unsqueeze(0)
        output = self.model(
            pixels.to(self.device),
            [gt_mask.to(self.device)],
            box.to(self.device),
            reconstruct_mask=False,
        )
        code_pair = [int(value) for value in output.quant_codes[0, 0].detach().cpu().tolist()]
        if len(code_pair) != 2 or not (0 <= code_pair[0] < 256 and 0 <= code_pair[1] < 256):
            raise ValueError(f"invalid SAMTok code pair: {code_pair}")
        remapped = [code_pair[0], 256 + code_pair[1]]
        span = "<|mt_start|>" + "".join(f"<|mt_{code:04d}|>" for code in remapped) + "<|mt_end|>"
        return span, remapped


def encode_manifest(
    generated_manifest: Path,
    output: Path,
    model_dir: Path,
    repo_root: Path,
    device: str = "cuda:0",
) -> list[dict[str, Any]]:
    encoder = SAMTokEncoder(model_dir, repo_root, device)
    rows: list[dict[str, Any]] = []
    with generated_manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            code, pair = encoder.encode(Path(row["source_image"]), Path(row["mask"]))
            rows.append(
                {
                    "case_id": row["case_id"],
                    "unit_id": row["unit_id"],
                    "mask_code": code,
                    "code_pair": pair,
                    "mask_tokenizer": str(encoder.model_dir / "mask_tokenizer_256x2.pth"),
                    "mask_tokenizer_sha256": encoder.weights_sha256,
                    "samtok_repo": str(encoder.repo_root),
                    "samtok_repo_revision": _git_revision(encoder.repo_root),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--samtok-root", type=Path, default=Path(os.environ.get("SAMTOK_REPO_ROOT", "/opt/Sa2VA")))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    rows = encode_manifest(args.generated, args.output, args.model_dir, args.samtok_root, args.device)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
