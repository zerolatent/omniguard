# VLM Safety Fine-tuning (OmniVinci)

## Quickstart

### 1) Build chat datasets

```bash
PYTHONPATH=/Users/sidreddy/dev/hackathon/vlm-finetune python -m src.data_utils.build_sft_dataset
```

This generates `src/data/train_chat.jsonl` and `src/data/test_chat.jsonl` with conversation+JSON targets.

### 2) Train with QLoRA (PEFT)

```bash
PYTHONPATH=/Users/sidreddy/dev/hackathon/vlm-finetune python -m src.train.train_sft \
  --model_path /Users/sidreddy/dev/hackathon/vlm-finetune/models/omnivinci \
  --train_path /Users/sidreddy/dev/hackathon/vlm-finetune/src/data/train_chat.jsonl \
  --eval_path /Users/sidreddy/dev/hackathon/vlm-finetune/src/data/test_chat.jsonl \
  --output_dir /Users/sidreddy/dev/hackathon/vlm-finetune/outputs/sft-omnivinci
```

### 3) Evaluate (schema validity & exact match)

```bash
PYTHONPATH=/Users/sidreddy/dev/hackathon/vlm-finetune python -m src.eval.infer \
  --model_path /Users/sidreddy/dev/hackathon/vlm-finetune/models/omnivinci \
  --test_path /Users/sidreddy/dev/hackathon/vlm-finetune/src/data/test_chat.jsonl
```

### Notes
- The dataset expects `src/data/train.jsonl`, `src/data/test.jsonl`, and `src/data/annotations/*.json` as provided.
- Default config uses `num_video_frames=64` and loads audio.

## Setup on H100 (Ubuntu)

```bash
chmod +x scripts/setup_h100.sh
./scripts/setup_h100.sh
# then
source .venv/bin/activate
PYTHONPATH=$(pwd) torchrun --standalone --nproc_per_node=1 \
  -m src.train.train_sft --deepspeed ./deepspeed_config.json
PYTHONPATH=$(pwd) python -m src.eval.infer --model_path ./outputs/sft-omnivinci --base_model_path ./models/omnivinci
```


