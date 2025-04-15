# sk-069167067d3e44ebb96c62013040e597

import random
from datasets import load_dataset
from openai import OpenAI


client = OpenAI(api_key="sk-069167067d3e44ebb96c62013040e597", base_url="https://api.deepseek.com")

def rewrite_question(question, choices, answer, subject, context=""):

    prompt = (
        f"Rewrite (or expand) the following MMLU question with deeper reasoning, in a more elaborate style, with the structure of subject, question, choices(in the form of 0 to 3), answer(in the form of 0 to 3) and one single paragraph of reasoning.\n\n"
        f"Subject: {subject}\n"
        f"Original Question: {question}\n"
        f"Choices: {choices}\n"
        f"Answer: {answer}\n\n"
        "Rewritten (with deeper reasoning) Question (please rephrase the question, choices, and answer in a way that shows more reasoning):"
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        stream=False
    )
    rewritten = response.choices[0].message.content.strip()
    if "Rewritten (with deeper reasoning) Question:" in rewritten:
        rewritten = rewritten.split("Rewritten (with deeper reasoning) Question:")[-1].strip()
    return rewritten

def main():
    dataset = load_dataset("cais/mmlu", "all", split="test")
    data_list = list(dataset)
    random.shuffle(data_list)
    chosen_samples = data_list[:20]
    results = []
    for sample in chosen_samples:
        question = sample.get("question", "")
        choices = sample.get("choices", "")
        answer = sample.get("answer", "")
        subject = sample.get("subject", "")
        rewritten = rewrite_question(question, choices, answer, subject)
        results.append({
            "subject": subject,
            "original_question": question,
            "choices": choices,
            "answer": answer,
            "rewritten_question": rewritten
        })
        print(f"Rewritten Question complete")
        print(f"Rewritten Question: {rewritten}")
    with open("output.txt", "w", encoding="utf-8") as f:
        for idx, item in enumerate(results, start=1):
            f.write(f"--- Sample {idx} ---\n")
            f.write(f"Subject: {item['subject']}\n")
            f.write(f"Original Question: {item['original_question']}\n")
            f.write(f"Choices: {item['choices']}\n")
            f.write(f"Answer: {item['answer']}\n")
            f.write(f"Rewritten Question: {item['rewritten_question']}\n")
            f.write("\n")
            
    for idx, item in enumerate(results, start=1):
        print(f"--- Sample {idx} ---")
        print("Subject:", item["subject"])
        print("Original Question:", item["original_question"])
        print("Choices:", item["choices"])
        print("Answer:", item["answer"])
        print("Rewritten Question:", item["rewritten_question"])
        print()

if __name__ == "__main__":
    main()