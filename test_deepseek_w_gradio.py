import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_SIZE = 1.5  # 1.5, 7
MAX_LEN = 500  # 100
TEMPERATURE = 0.7  # 0.7
MODEL_ID = f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{MODEL_SIZE}B"

# Load with explicit MPS configuration
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="mps",  # Directly specify MPS
    torch_dtype=torch.float16,  # Force FP16 for MPS compatibility
    low_cpu_mem_usage=True  # Essential for 8GB RAM
).eval()  # Set to eval mode immediately

# No need for model.to("mps") since device_map handles it


def generate_text(prompt, max_length=MAX_LEN, temperature=TEMPERATURE):
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Gradio UI
def launch_gradio():
    demo = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.Textbox(lines=3, placeholder="Enter your prompt..."),
            gr.Slider(50, 500, value=MAX_LEN, label="Max Length"),
            gr.Slider(0.1, 1.0, value=TEMPERATURE, label="Temperature")
        ],
        outputs="text",
        title=f"DeepSeek-R1-Distill-Qwen-{MODEL_SIZE}B Demo",
        description=f"A distilled {MODEL_SIZE}B parameter model for efficient local AI."
    )
    demo.launch(share=True)  # Access via http://localhost:7860


if __name__ == '__main__':
    launch_gradio()

