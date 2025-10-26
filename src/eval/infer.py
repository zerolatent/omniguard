import argparse
import json
import os
from typing import Dict, List

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel


SCHEMA = {
    "incident_type": str,
    "safety_status": str,
    "probability": float,
    "safety_response": str,
    "action_plan": str,
}


def parse_json_strict(s: str) -> Dict:
    s = s.strip()
    if "{" in s and "}" in s:
        s = s[s.find("{") : s.rfind("}") + 1]
    return json.loads(s)


def is_valid_schema(obj: Dict) -> bool:
    for k, t in SCHEMA.items():
        if k not in obj:
            return False
        if t is float:
            try:
                float(obj[k])
            except Exception:
                return False
        elif not isinstance(obj[k], t):
            return False
    return True


def _load_model_and_processor(model_path: str, base_model_path: str | None):
    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_adapter:
        base_path = base_model_path or "/Users/sidreddy/dev/hackathon/vlm-finetune/models/omnivinci"
        base = AutoModelForCausalLM.from_pretrained(base_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
        model = PeftModel.from_pretrained(base, model_path)
        processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def generate_for_video(model_path: str, video_path: str, prompt: str, num_video_frames: int = 64, audio_chunk_length: str = "max_3600", base_model_path: str | None = None) -> str:
    model, processor = _load_model_and_processor(model_path, base_model_path)
    model.config.num_video_frames = num_video_frames
    processor.config.num_video_frames = num_video_frames
    model.config.audio_chunk_length = audio_chunk_length
    processor.config.audio_chunk_length = audio_chunk_length

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor([text])
    out_ids = model.generate(
        input_ids=inputs.input_ids,
        media=getattr(inputs, "media", None),
        media_config=getattr(inputs, "media_config", None),
        max_new_tokens=256,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
    )
    return processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]


def evaluate(model_path: str, test_jsonl: str, base_model_path: str | None) -> Dict:
    prompt = (
        "Analyze this construction safety video and return strict JSON with keys "
        "incident_type, safety_status, probability, safety_response, action_plan. Reply with JSON only."
    )
    em = 0
    valid = 0
    total = 0
    preds: List[Dict] = []
    with open(test_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            video_path = item["conversation"][0]["content"][0]["video"]
            gold = json.loads(item["response"]) if isinstance(item["response"], str) else item["response"]

            text = generate_for_video(model_path, video_path, prompt, base_model_path=base_model_path)
            try:
                obj = parse_json_strict(text)
                if is_valid_schema(obj):
                    valid += 1
                if obj == gold:
                    em += 1
                preds.append({"video": video_path, "pred": obj, "gold": gold})
            except Exception:
                preds.append({"video": video_path, "pred_text": text, "gold": gold})
            total += 1

    return {"exact_match": em / max(total, 1), "schema_valid": valid / max(total, 1), "total": total, "preds": preds}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_model = os.path.join(project_root, "models", "omnivinci")
    default_test = os.path.join(project_root, "src", "data", "test_chat.jsonl")
    ap.add_argument("--model_path", type=str, default=default_model)
    ap.add_argument("--test_path", type=str, default=default_test)
    ap.add_argument("--base_model_path", type=str, default=None, help="Base model path when --model_path is a LoRA adapter directory")
    args = ap.parse_args()
    res = evaluate(args.model_path, args.test_path, args.base_model_path)
    print(json.dumps({k: v for k, v in res.items() if k != "preds"}, indent=2))
    out = "/Users/sidreddy/dev/hackathon/vlm-finetune/outputs/eval_preds.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res["preds"], f, indent=2)


