import json
import os
from dataclasses import dataclass
from typing import Dict, Iterator, List

from src.data_utils.safety_labels import INCIDENT_TYPES, SAFETY_STATUS, derive_action_plan


@dataclass
class Sample:
    conversation: List[Dict]
    target_json: Dict
    meta: Dict


INSTRUCTION = (
    "Analyze this construction safety video and return strict JSON with keys "
    "incident_type, safety_status, probability, safety_response, action_plan. "
    "Reply with JSON only."
)


def _load_index(jsonl_path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def _read_annotation(root: str, rel_path: str) -> Dict:
    abs_path = os.path.join(root, "src", "data", rel_path) if not os.path.isabs(rel_path) else rel_path
    with open(abs_path, "r") as f:
        return json.load(f)


def _abs_video_path(root: str, rel_path: str) -> str:
    return os.path.join(root, "src", "data", rel_path) if not os.path.isabs(rel_path) else rel_path


def _canonicalize_target(ann: Dict) -> Dict:
    incident = ann.get("predictions", {}).get("incident_type")
    status = ann.get("safety_status")
    prob = ann.get("predictions", {}).get("probability", 0.9)
    response = ann.get("safety_response", "")

    if incident not in INCIDENT_TYPES:
        incident = INCIDENT_TYPES[0]
    if status not in SAFETY_STATUS:
        status = SAFETY_STATUS[0]

    action = derive_action_plan(incident, status)

    return {
        "incident_type": incident,
        "safety_status": status,
        "probability": float(prob),
        "safety_response": response,
        "action_plan": action,
    }


def iter_samples(root: str, split: str) -> Iterator[Sample]:
    idx_path = os.path.join(root, "src", "data", f"{split}.jsonl")
    for row in _load_index(idx_path):
        ann = _read_annotation(root, row["annotation"])  # type: ignore[index]
        target = _canonicalize_target(ann)
        video_abs = _abs_video_path(root, row["video"])  # type: ignore[index]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_abs},
                    {"type": "text", "text": INSTRUCTION},
                ],
            }
        ]
        yield Sample(conversation=conversation, target_json=target, meta={"video_id": row.get("video_id")})


def build_supervised_examples(root: str, split: str) -> List[Dict]:
    examples: List[Dict] = []
    for s in iter_samples(root, split):
        examples.append(
            {
                "conversation": s.conversation,
                "response": json.dumps(s.target_json, ensure_ascii=False),
                "meta": s.meta,
            }
        )
    return examples


def save_examples(root: str, split: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = build_supervised_examples(root, split)
    with open(out_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_examples(project_root, "train", os.path.join(project_root, "src", "data", "train_chat.jsonl"))
    save_examples(project_root, "test", os.path.join(project_root, "src", "data", "test_chat.jsonl"))


