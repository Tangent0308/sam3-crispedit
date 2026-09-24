"""Regression guards for the Arnold libGL failure and differing node installs."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('crispedit_preflight',
    Path(__file__).resolve().parents[1] / 'scripts/preflight_crispedit_env.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def reports():
    return [dict(hostname=f'worker{rank}', commit='abc', code_sha256='def', versions={'cv2': '4.11.0'},
                 vllm_probe=True, gpus=8, python='/opt/clone/.venv-crispedit/bin/python',
                 base_python='/opt/clone/.uv-python/python') for rank in range(4)]


def test_four_local_node_reports_are_accepted():
    preflight.validate_cluster_reports(reports())


@pytest.mark.parametrize('key,value,message', [
    ('hostname', 'worker0', 'distinct physical'),
    ('commit', 'other_commit', 'disagree on commit'),
    ('code_sha256', 'dirty_code', 'disagree on code_sha256'),
    ('versions', {'cv2': '5.0.0'}, 'disagree on versions'),
    ('gpus', 4, 'Incomplete GPU'),
    ('vllm_probe', False, 'Incomplete GPU'),
    ('base_python', '/mnt/bn/old/python', 'local Python'),
])
def test_bad_node_stops_before_data_planning(key, value, message):
    values = copy.deepcopy(reports())
    values[1][key] = value
    with pytest.raises(ValueError, match=message):
        preflight.validate_cluster_reports(values)


def test_mixed_opencv_distributions_are_rejected(monkeypatch):
    monkeypatch.setattr(preflight.importlib.metadata, 'distributions', lambda: [
        SimpleNamespace(metadata={'Name': name}) for name in ('opencv-python', 'opencv-python-headless')])
    with pytest.raises(RuntimeError, match='Only opencv-python-headless'):
        preflight.check_opencv()


def test_gui_opencv_binary_is_rejected_even_if_metadata_claims_headless(monkeypatch):
    import cv2
    monkeypatch.setattr(preflight.importlib.metadata, 'distributions', lambda: [
        SimpleNamespace(metadata={'Name': 'opencv-python-headless'})])
    monkeypatch.setattr(cv2, 'getBuildInformation', lambda: '  GUI: QT5\n')
    with pytest.raises(RuntimeError, match='without GUI'):
        preflight.check_opencv()
