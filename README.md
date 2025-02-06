# test_ollama
https://ollama.com/blog/python-javascript-libraries


# References for ollama
* https://github.com/ollama/ollama-python
* https://pypi.org/project/ollama/
* https://ollama.com/library/deepseek-r1:1.5b

# References for Gradio UI version
* https://medium.com/@harshithaparitala1/deepseekais-pocket-sized-revolution-running-open-source-1-5b-model-locally-bb34082bddac

# Requirement for silicon mac
* python                    3.10.16
* ollama                    0.4.7 
* transformers              4.48.2 
* tokenizers                0.21.0 
* gradio                    5.14.0 
* torch                     2.6.0 
* accelerate                1.3.0 

## 1. Setting up the Environment for silicon mac 
    conda create -n ollama python=3.10
    conda activate ollama 

    pip install ollama

    # Install PyTorch with MPS support (optimized for Apple Silicon)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install Hugging Face libraries and Gradio for UI
    pip install transformers tokenizers gradio
    pip install accelerate==1.3.0  # pip install 'accelerate>=0.26.0'

