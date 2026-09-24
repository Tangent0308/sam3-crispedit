"""Deployment paths only; never change model or sampling policy."""
import os


def runtime_path(name, legacy):
    return os.environ.get('SAMTOK_' + name, legacy)


def qwen38_model():
    return runtime_path('QWEN38_MODEL', '/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B')


def editor_model():
    return runtime_path('QWEN21_MODEL', '/tmp/tanyue_qwen_image_21')
