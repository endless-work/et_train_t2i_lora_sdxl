# et_train_t2i_lora_sdxl

## Environment Setup

When creating a pod, use the template:  
**Runpod PyTorch 2.4.0**  
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`  

⚠️ The disk should be **40–50GB**.

---

## Installation

Update pip:
```bash

#VS code SSH launch
ssh wuc7nou671g0u8-6441196d@ssh.runpod.io -i ~/.ssh/id_ed25519

# --- Install dependencies ---
pip install --upgrade pip && \
pip install \
    transformers==4.44.2 \
    accelerate==1.1.1 \
    datasets==2.21.0 \
    safetensors==0.4.5 \
    pillow \
    tqdm \
    peft \
    bitsandbytes \
    git+https://github.com/huggingface/diffusers

# --- Check installation ---
python -c "import diffusers; print(diffusers.__version__)"

# --- Update transformers --
pip install "transformers>=4.55.0,<5.0.0" --upgrade

# --- Login to Hugging Face ---
huggingface-cli login

# --- Create working directories ---
mkdir -p /workspace/fine-tuning/scripts /workspace/fine-tuning/outputs
cd /workspace/fine-tuning/scripts


# --- Download the training script ---
wget https://raw.githubusercontent.com/endless-work/et_train_t2i_lora_sdxl/main/et_train_t2i_lora_sdxl.py



