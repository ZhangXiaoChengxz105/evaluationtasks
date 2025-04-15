# === dataloader.py ===
# Loads ScienceQA dataset from HuggingFace and saves as local JSONL + image files

import os
import json
from typing import List, Dict
from PIL import Image
from datasets import load_dataset

def load_scienceqa(split="train") -> List[Dict]:
    """
    Load the ScienceQA dataset from HuggingFace and convert it into a format compatible with MCTS reasoning search.
    Also saves each image as a separate PNG file named by question ID.
    Args:
        split (str): Which split to load (e.g., 'train', 'validation', 'test')
    Returns:
        List of processed QA examples as dictionaries
    """
    dataset = load_dataset("derek-thomas/ScienceQA", split=split)
    processed = []

    image_output_dir = "images"
    os.makedirs(image_output_dir, exist_ok=True)

    for idx, item in enumerate(dataset):
        image_filename = None
        image_data = item.get("image", None)

        if hasattr(image_data, "save"):
            image_filename = f"{idx}.png"
            image_path = os.path.join(image_output_dir, image_filename)
            try:
                image_data.save(image_path)
            except Exception as e:
                print(f"Failed to save image for item {idx}: {e}")
                image_filename = None

        qa_example = {
            "id": str(idx),
            "question": item["question"],
            "choices": item.get("choices", []),
            "answer": item.get("answer", -1),
            "hint": item.get("hint", ""),
            "lecture": item.get("lecture", ""),
            "solution": item.get("solution", ""),
            "image_path": image_filename,
            "metadata": {
                "subject": item.get("subject", ""),
                "grade": item.get("grade", ""),
                "topic": item.get("topic", ""),
                "category": item.get("category", ""),
                "skill": item.get("skill", ""),
                "task": item.get("task", "")
            }
        }
        processed.append(qa_example)

    # Save output as JSONL
    output_path = "scienceqa_mcts_data.jsonl"
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in processed:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(processed)} samples to {output_path}")
    return processed

# === Example usage ===
if __name__ == "__main__":
    data = load_scienceqa("train")
    print("Loaded", len(data), "examples")
    print("Sample for MCTS:", data[0])
