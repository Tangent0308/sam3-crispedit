"""Deployment font checks must work before Pillow or OS fonts are installed."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image, ImageFont

from sa1b_pipeline import fonts
from sa1b_pipeline.group_audit import _member_comparison


def test_checksum_preflight_needs_only_standard_library():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, '-S', '-m', 'sa1b_pipeline.fonts'], cwd=repo,
        env={**os.environ, 'PYTHONPATH': str(repo)}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'SHA-256 verified' in result.stdout


def test_audit_labels_do_not_depend_on_system_fonts(monkeypatch, tmp_path):
    original = ImageFont.truetype
    calls = []

    def bundled_only(path, *args, **kwargs):
        assert Path(path).parent == fonts.FONT_DIR
        calls.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(ImageFont, 'truetype', bundled_only)
    fonts.verify_fonts(render=True)
    source = tmp_path / 'input.png'
    Image.new('RGB', (100, 100), 'gray').save(source)
    result = _member_comparison([str(source), str(source)], tmp_path / 'audit.jpg')
    with Image.open(result) as panel:
        assert panel.size == (1536, 520)
    assert set(calls) == set(fonts.FONT_NAMES)


def test_missing_font_fails_with_actionable_path(monkeypatch, tmp_path):
    monkeypatch.setattr(fonts, 'FONT_DIR', tmp_path)
    with pytest.raises(FileNotFoundError, match='Bundled font missing:.*DejaVuSans-Bold'):
        fonts.load_font(22, bold=True)


def test_checksum_rejects_corrupted_font(monkeypatch, tmp_path):
    import shutil
    for path in fonts.FONT_DIR.iterdir():
        shutil.copyfile(path, tmp_path / path.name)
    (tmp_path / fonts.FONT_NAMES[0]).write_bytes(b'not a valid font')
    monkeypatch.setattr(fonts, 'FONT_DIR', tmp_path)
    with pytest.raises(RuntimeError, match='checksum mismatch'):
        fonts.verify_fonts()
