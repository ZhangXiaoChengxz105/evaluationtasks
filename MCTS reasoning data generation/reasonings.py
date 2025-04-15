import json
import random
import base64
from collections import defaultdict
from typing import List, Dict
import math
import random
from datasets import load_dataset
from openai import OpenAI
import os
client = OpenAI(api_key="apikey", base_url="https://api.deepseek.com")# you would need to insert your secret key
gpt_client= OpenAI(api_key="apikey")# you would need to insert your secret key
image_path = os.path.join("images", '1.png')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def qa_reasoning(example):
    print("qa reasoning")
    question = example['question']
    choices = example['choices']
    prev_reasoning = example.get("reasoning", "")

    prev_reasoning = prev_reasoning.strip()

    context_text = prev_reasoning if prev_reasoning else ""

    prompt = (
        "You are an expert educational reasoning assistant. Given a question, its choices, and optionally some prior reasoning steps, provide an explanation of the question and options but do not indicate anything about your answers in the reasoning part. You must include the information of the options in the reasonings, such as 'the options given are 0:...,1:...,2:...,' at the start of the reasoning.  \n"
        "Return your response in the format:\n"
        "REASONING: <your reasoning>\n"
        "ANSWER: <an index from 0-3 if confident, otherwise null>\n\n"
        f"Question: {question}\n"
        f"Choices: {choices}\n"
        f"Previous Reasoning: {context_text}\n"
        "\nYour response:"
    )



    messages = [
        {"role": "system", "content": "You are a helpful educational assistant."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=False
    )

    output = response.choices[0].message.content.strip()

    reasoning = ""
    answer = None
    for line in output.splitlines():
        if line.strip().startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
        elif line.strip().startswith("ANSWER:"):
            ans_text = line.replace("ANSWER:", "").strip()
            if ans_text.lower() != "null":
                try:
                    answer = int(ans_text)
                except ValueError:
                    answer = None

    updated_example = dict(example)
    if prev_reasoning:
        updated_example["reasoning"] = prev_reasoning.strip() + "\n" + reasoning
    else:
        updated_example["reasoning"] = reasoning
    # print(updated_example)
    # print(answer)

    return updated_example, answer

def meta_reasoning(example,prev_reasoning=[]):
    print("meta reasoning")
    question = example['question']
    metadata = example.get('metadata', {})

    subject = metadata.get('subject', 'unknown')
    grade = metadata.get('grade', 'unknown')
    topic = metadata.get('topic', 'unknown')
    category = metadata.get('category', 'unknown')
    skill = metadata.get('skill', 'unknown')
    task = metadata.get('task', 'unknown')
    prev_reasoning = example.get("reasoning", "")

    prev_reasoning = prev_reasoning.strip()

    context_text = prev_reasoning if prev_reasoning else ""
    

    prompt = (
        "You are an expert educational reasoning assistant. Given a question, its metadata context, and optionally some prior reasoning steps, provide a detailed explanation based on the metadata (e.g., subject knowledge), but do not indicate anything about your answers in the reasoning part. You must include the metadate you received in the reasoning, such as 'the subject of this question is ... and the topic of the question is..'. and make sure to contain all metadata given.\n"
        "Return your response in the format:\n"
        "REASONING: <your reasoning>\n"
        "ANSWER: <an index from 0-3 if you are confident, otherwise null>\n\n"
        f"Subject: {subject}\n"
        f"Grade: {grade}\n"
        f"Topic: {topic}\n"
        f"Category: {category}\n"
        f"Skill: {skill}\n"
        f"Task: {task}\n"
        f"Question: {question}\n"
        f"Previous Reasoning: {context_text}\n"
        "\nYour response:"
    )

    messages = [
        {"role": "system", "content": "You are a helpful educational assistant."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=False
    )

    output = response.choices[0].message.content.strip()

    reasoning = ""
    answer = None
    for line in output.splitlines():
        if line.strip().startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
        elif line.strip().startswith("ANSWER:"):
            ans_text = line.replace("ANSWER:", "").strip()
            if ans_text.lower() != "null":
                try:
                    answer = int(ans_text)
                except ValueError:
                    answer = None

    updated_example = dict(example)
    if prev_reasoning:
        updated_example["reasoning"] = prev_reasoning.strip() + "\n" + reasoning
    else:
        updated_example["reasoning"] = reasoning


    return updated_example, answer

def pic_reasoning(example):
    print("pic reasoning")
    question = example['question']
    image_filename = example.get("image_path", None)
    prev_reasoning = example.get("reasoning", "")
    print("Previous Reasoning:", prev_reasoning)
    context_text = prev_reasoning.strip()

    if not image_filename:
        print("No image path found in the example.")
        # Handle the case where there is no image path
        return example, None  # skip if no image path

    # Assume images are in ./images/ directory
    image_path = os.path.join("images", image_filename)

    base64_image = encode_image(image_path)

    prompt = (
        "You are an expert educational reasoning assistant. Given a question, a related, and optionally some prior reasoning steps,first provide a description of the picture, then tell in detail how this picture can help solve the problem, then try to give the answer only whe you are very sure about it otherwise say null. Do not indicate anything about your answers in the reasoning part. You must start with 'There is a picture given that '.\n"
        "You must not provide the final answer in the reasoning section.\n"
        "Return your response in the format:\n"
        "REASONING: <your reasoning>\n"
        "ANSWER: <an index from 0-3 if you are confident, otherwise null>\n\n"
        f"Question: {question}\n"
        f"Previous Reasoning: {context_text}\n"
        "\nYour response:"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]

    response = gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=False
    )

    output = response.choices[0].message.content.strip()

    reasoning = ""
    answer = None
    for line in output.splitlines():
        if line.strip().startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
        elif line.strip().startswith("ANSWER:"):
            ans_text = line.replace("ANSWER:", "").strip()
            if ans_text.lower() != "null":
                try:
                    answer = int(ans_text)
                except ValueError:
                    answer = None

    updated_example = dict(example)
    if prev_reasoning:
        updated_example["reasoning"] = prev_reasoning.strip() + "\n" + reasoning
    else:
        updated_example["reasoning"] = reasoning

    print(updated_example)
    print(answer)

    return updated_example, answer