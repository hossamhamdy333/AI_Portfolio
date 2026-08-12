"""
Hugging Face Spaces entry point for the AI Support Copilot.

IMPORTANT: this file, not src/app.py, is what runs on Hugging Face Spaces.
ZeroGPU (HF's free shared-GPU tier) only schedules onto Gradio SDK Spaces --
Docker and Static Spaces cannot use it. So this app is built with Gradio,
and the model is loaded at module scope with .to("cuda") eagerly (the
documented ZeroGPU pattern), while the actual generation call is wrapped in
@spaces.GPU so it only claims a GPU slot for the few seconds it's needed.

src/app.py (FastAPI) is kept separately for deploying to a platform with a
real, always-on GPU (e.g. an Azure VM/Container App) later -- it does NOT
use @spaces.GPU since that decorator is HF-specific.
"""

import logging
import os

import gradio as gr
import spaces
import torch
from unsloth import FastLanguageModel

from src.retriever import KBRetriever
from src.evaluate import evaluate_faithfulness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_REPO = os.environ.get("MODEL_REPO", "hossamhamdy333/support-copilot-llama3-lora")
MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 200

SYSTEM_PROMPT = (
    "You are a senior customer support agent for a premium brand. "
    "Reply politely, professionally, and resolve the user's issue based "
    "ONLY on the provided context."
)

# --- Load once at module scope (the documented ZeroGPU pattern) ---
logger.info("Loading fine-tuned model from %s ...", MODEL_REPO)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_REPO,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
logger.info("Model loaded.")

logger.info("Building KB retriever...")
retriever = KBRetriever()
logger.info("Retriever ready.")


@spaces.GPU(duration=60)
def generate(query: str) -> tuple[str, str]:
    """The only part of a request that needs the GPU. Returns (response, context)."""
    context = retriever.retrieve(query)

    prompt = f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
Context: {context}
Query: {query}
<|assistant|>
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>")[-1]
    answer = decoded.strip()

    if os.environ.get("ENABLE_EVAL") == "1":
        eval_result = evaluate_faithfulness(query, context, answer)
        if eval_result and not eval_result.get("is_faithful", True):
            logger.warning("Faithfulness check flagged this response: %s", eval_result.get("reason"))

    return answer, context


def respond(message: str, history: list[dict]) -> tuple[list[dict], str]:
    """Gradio chat callback: run generation, append to history, surface the retrieved context."""
    if not message.strip():
        return history, ""
    try:
        answer, context = generate(message)
    except Exception as e:
        logger.exception("Generation failed")
        answer, context = f"⚠️ Something went wrong: {e}", ""

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, context


CUSTOM_CSS = """
#chatbot { min-height: 480px; }
.context-box textarea { font-size: 12px !important; opacity: 0.85; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS, title="AI Support Copilot") as demo:
    gr.Markdown(
        "## 🎧 AI Support Copilot\n"
        "Fine-tuned Llama-3 (8B, QLoRA) + RAG retrieval over a customer support knowledge base."
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        type="messages",
        elem_id="chatbot",
        value=[{"role": "assistant", "content": "Hello! I'm your AI Support Copilot. How can I help?"}],
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="e.g. Where is my refund?",
            show_label=False,
            scale=4,
        )
        send = gr.Button("Send", variant="primary", scale=1)

    with gr.Accordion("🔍 Retrieved knowledge base context (last response)", open=False):
        context_box = gr.Textbox(label="", interactive=False, elem_classes="context-box")

    send.click(respond, inputs=[msg, chatbot], outputs=[chatbot, context_box]).then(
        lambda: "", None, msg
    )
    msg.submit(respond, inputs=[msg, chatbot], outputs=[chatbot, context_box]).then(
        lambda: "", None, msg
    )

if __name__ == "__main__":
    demo.queue().launch()
