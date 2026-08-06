import os
import logging
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)

def format_prompt(row):
    return f"""<|system|>
You are a senior customer support agent for a premium brand. Reply politely, professionally, and resolve the user's issue based ONLY on the provided context.
<|user|>
Context: {row.get('category', 'General')} - {row['intent']}
Query: {row['instruction']}
<|assistant|>
{row['response']}"""

def main():
    os.makedirs("data", exist_ok=True)
    logging.info("Downloading dataset...")
    
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    dataset = dataset.shuffle(seed=42).select(range(5000)) 
    
    df = pd.DataFrame(dataset)
    df['text'] = df.apply(format_prompt, axis=1)
    
    output_path = "data/train.jsonl"
    df[['text']].to_json(output_path, orient='records', lines=True)
    logging.info(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    main()