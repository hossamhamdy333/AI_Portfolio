import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def train():
    os.makedirs("outputs", exist_ok=True)
    
    # Load Model in 4-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=512, 
        load_in_4bit=True,
    )

    # Apply LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", 
    )

    # Load Data & Train
    dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=60, 
            learning_rate=2e-4,
            optim="adamw_8bit",
            output_dir="outputs/support_lora",
        ),
    )

    trainer.train()
    model.save_pretrained("outputs/support_lora")
    tokenizer.save_pretrained("outputs/support_lora")
    print("Training Complete. LoRA weights saved!")

if __name__ == "__main__":
    train()