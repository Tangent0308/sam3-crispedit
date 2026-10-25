#!/usr/bin/env bash
# Packages the evidence production loads from directories next to the locked environments, with the
# versions used in validation (dependencies come from the locks):
#   ocr-lib     RapidOCR for --text-verify (editor env; text_verification.py reads SA1B_OCR_LIB)
#   feeder-lib  crc32c and tfd-utils (TFRecord checks and Example decoding) for the shard feeder
#               (mllm env, whose lock has numpy and protobuf; SA1B_FEEDER_LIB)
set -euo pipefail
runtime=${1:?usage: setup_extras.sh RUNTIME_ROOT}
: "${UV_BIN:=/home/tiger/.local/bin/uv}"
install() {  # role target check packages...
  local python="$runtime/$1/bin/python" target="$runtime/$2" check=$3; shift 3
  [[ -f "$target/ready" ]] && return 0
  rm -rf "$target.partial"
  "$UV_BIN" pip install --python "$python" --target "$target.partial" --no-deps --link-mode=copy \
    --index-url https://bytedpypi.byted.org/simple/ "$@"
  rm -rf "$target"; mv "$target.partial" "$target"
  LIB="$target" "$python" -c "import os,sys; sys.path.append(os.environ['LIB']); $check"
  date > "$target/ready"
}
install editor ocr-lib 'from rapidocr_onnxruntime import RapidOCR; RapidOCR()' \
  rapidocr_onnxruntime==1.4.4 pyclipper==1.4.0 shapely==2.2.0 pyyaml==6.0.3 six==1.17.0
install mllm feeder-lib 'import crc32c; assert crc32c.crc32c(b"123456789") == 0xe3069283; from tfd_utils.pb2 import Example' \
  crc32c==2.9.post0 tfd-utils==1.3.0
