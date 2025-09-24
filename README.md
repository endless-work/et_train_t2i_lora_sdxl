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

# --- Login to Hugging Face ---
huggingface-cli login

# --- Create working directories ---
mkdir -p /workspace/fine-tuning/scripts /workspace/fine-tuning/outputs
cd /workspace/fine-tuning/scripts


# --- Download the training script ---
wget https://raw.githubusercontent.com/endless-work/et_train_t2i_lora_sdxl/main/et_train_t2i_lora_sdxl.py



# Launch training 
accelerate launch et_train_t2i_lora_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --dataset_name=endlesstools/et-sdxl-lora-wood \
  --caption_column=text \
  --image_column=image \
  --resolution=1024 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --rank=16 \
  --learning_rate=1e-4 \
  --max_train_steps=3000 \
  --checkpointing_steps=500 \
  --output_dir="/workspace/fine-tuning/outputs/lora-wood-xbkdtgv"


# create push_hf_trained.py for push on HF model repo
cat > push_hf_trained.py << 'EOF'
from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/workspace/fine-tuning/outputs/lora-wood-xbkdtgv",
    repo_id="endlesstools/et-sdxl-w-wood",
    repo_type="model",
    commit_message="Upload LoRA weights"
)

print("✅ All files uploaded to Hugging Face Hub")
EOF


# start push_hf_trained.py
python push_hf_trained.py

