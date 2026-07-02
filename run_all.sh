#!/usr/bin/env bash
# Run the whole pipeline with a small, fast configuration for learning.
# Assumes you've run setup.sh. Uses the venv's python directly.
set -e
cd "$(dirname "$0")"
PY=./.venv/bin/python

echo "==> device"
$PY common/device.py

echo "==> stage 0: download data"
$PY stage0_data/download_data.py --stories 10000

echo "==> stage 1: train tokenizer"
$PY stage1_tokenizer/train_tokenizer.py

echo "==> stage 2: prepare + pretrain"
$PY stage2_pretrain/prepare_dataset.py
# Small run for a laptop. Increase --max-iters for better quality.
$PY stage2_pretrain/train.py --max-iters 2000 --eval-interval 250

echo "==> stage 2: sample from the base model"
$PY stage2_pretrain/generate.py --prompt "Once upon a time" --max-new-tokens 120

echo "==> stage 3: chat fine-tune"
$PY stage3_chat_finetune/build_chat_data.py --n 4000
$PY stage3_chat_finetune/train_chat.py --epochs 3

echo "==> stage 4: tool fine-tune"
$PY stage4_tool_tuning/build_tool_data.py --n 4000
$PY stage4_tool_tuning/train_tools.py --epochs 3

echo "==> stage 6: export to Hugging Face format"
$PY stage6_publish/export_to_hf.py --ckpt checkpoints/tools.pt --out hf_model

echo
echo "All done!"
echo "  Chat:        $PY stage3_chat_finetune/chat_cli.py"
echo "  Chat+tools:  $PY stage5_chat_interface/chat_with_tools.py"
echo "  Web UI:      $PY stage5_chat_interface/web_app.py --ckpt checkpoints/tools.pt --tools"
echo "  Publish:     $PY stage6_publish/push_to_hub.py --repo <you>/gpt-nano"
