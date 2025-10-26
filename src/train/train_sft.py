import os
import json
from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model


@dataclass
class Args:
    model_path: str
    train_path: str
    eval_path: str
    output_dir: str
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_video_frames: int = 64
    audio_chunk_length: str = "max_3600"


def build_chat_text(processor, conversation: List[Dict]) -> str:
    return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)


def collate_fn(features: List[Dict], processor):
    texts = [build_chat_text(processor, f["conversation"]) + f["response"] for f in features]
    inputs = processor(texts)
    return inputs


def main():
    import argparse

    p = argparse.ArgumentParser()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_model = os.path.join(project_root, "models", "omnivinci")
    default_train = os.path.join(project_root, "src", "data", "train_chat.jsonl")
    default_eval = os.path.join(project_root, "src", "data", "test_chat.jsonl")
    default_out = os.path.join(project_root, "outputs", "sft-omnivinci")

    p.add_argument("--model_path", type=str, default=default_model)
    p.add_argument("--train_path", type=str, default=default_train)
    p.add_argument("--eval_path", type=str, default=default_eval)
    p.add_argument("--output_dir", type=str, default=default_out)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--num_video_frames", type=int, default=64)
    p.add_argument("--audio_chunk_length", type=str, default="max_3600")
    p.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON (optional)")
    args_ns = p.parse_args()

    args = Args(**vars(args_ns))

    # Hopper (H100) supports BF16; enable TF32 for speed as well.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model.config.num_video_frames = args.num_video_frames
    processor.config.num_video_frames = args.num_video_frames
    model.config.audio_chunk_length = args.audio_chunk_length
    processor.config.audio_chunk_length = args.audio_chunk_length

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    data_files = {"train": args.train_path, "validation": args.eval_path}
    ds = load_dataset("json", data_files=data_files)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        logging_steps=10,
        evaluation_strategy="epoch",
        report_to=[],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        deepspeed=args_ns.deepspeed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=lambda batch: collate_fn(batch, processor),
        tokenizer=processor.tokenizer,
        packing=False,
        max_seq_length=4096,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()


