---
base_model: ALLaM-AI/ALLaM-7B-Instruct-preview
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- qlora
- peft
- arabic
- legal
---

<div align="center">

# ALLaM-7B QLoRA Adapter — Arabic Legal Instruction Following

`peft` `transformers` `bitsandbytes` `ALLaM-AI/ALLaM-7B-Instruct-preview`

</div>

---

LoRA adapter for `ALLaM-AI/ALLaM-7B-Instruct-preview`, fine-tuned with QLoRA (4-bit base) on Arabic legal instruction data: article analysis, plain-language simplification, and judgment prediction.

This folder is the adapter's model card. The data, training procedure, LLM-judged results (base vs. v1 vs. v2), and limitations are in the parent project's [README](../../../README.md).

> **Weights are not in this repo.** The adapter weights are hosted on the Hugging Face Hub as [`hossam3759180/allam-qlora-legal-adapter`](https://huggingface.co/hossam3759180/allam-qlora-legal-adapter). This folder holds the adapter config and tokenizer files only.

## Adapter configuration

From `adapter_config.json`:

| Setting | Value |
|---|---|
| PEFT type / task | LoRA / `CAUSAL_LM` |
| Rank / alpha / dropout | 32 / 64 / 0.1 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Bias | none |
| Saved with PEFT | 0.20.0 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id = "ALLaM-AI/ALLaM-7B-Instruct-preview"
adapter_id = "hossam3759180/allam-qlora-legal-adapter"

# Use the same 4-bit quantization settings as training (see the parent README).
bnb = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id)
```

## Limitations

Evaluated with an LLM judge (Gemini) on a fixed 150-row sample, not against human-labeled data. Not legal advice; see the parent README's limitations section for what was and wasn't verified.
