import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_size = 1.5  # 1.5, 7
model_name = f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{model_size}B"

# Load with explicit MPS configuration
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",  # Directly specify MPS
    torch_dtype=torch.float16,  # Force FP16 for MPS compatibility
    low_cpu_mem_usage=True  # Essential for 8GB RAM
).eval()  # Set to eval mode immediately

# No need for model.to("mps") since device_map handles it


def generate_text(prompt, max_length=100, temperature=0.7):
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
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your prompt..."),
        gr.Slider(50, 500, value=100, label="Max Length"),
        gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
    ],
    outputs="text",
    title=f"DeepSeek-R1-Distill-Qwen-{model_size}B Demo",
    description=f"A distilled {model_size}B parameter model for efficient local AI."
)

demo.launch(share=True)  # Access via http://localhost:7860

