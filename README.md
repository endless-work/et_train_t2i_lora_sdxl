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

# Install dependencies:
pip install --upgrade pip
pip install transformers==4.44.2 accelerate==1.1.1 datasets==2.21.0 safetensors==0.4.5
pip install git+https://github.com/huggingface/diffusers
# Additional tools (images, progress, LoRA support):
pip install pillow tqdm peft bitsandbytes
# Check installation:
python -c "import diffusers; print(diffusers.__version__)"

# Login to Hugging Face:
huggingface-cli login

# Create working directories:
mkdir -p /workspace/fine-tuning/scripts
mkdir -p /workspace/fine-tuning/outputs
cd /workspace/fine-tuning/scripts

# Download the training script:
wget https://raw.githubusercontent.com/endless-work/et_train_t2i_lora_sdxl/main/et_train_t2i_lora_sdxl.py

# 
accelerate launch /workspace/fine-tuning/scripts/et_train_t2i_lora_sdxl.py \
  --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
  --dataset_name=endlesstools/et-sdxl-lora-wood \
  --caption_column=text \
  --image_column=image \
  --resolution=1024 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-5 \
  --num_train_epochs=50 \
  --validation_prompt "bhfaqhh, seamless tileable PBR texture, dynamic tigerwood with bold orange and dark brown streaks, continuous layered surface, flowing linear pattern" \
  --num_validation_images=2 \
  --validation_epochs=1 \
  --output_dir="/workspace/fine-tuning/outputs/sdxl-wood-lora"


# create push_hf_trained.py for push on HF model repo
cat > push_hf_trained.py << 'EOF'
from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/workspace/fine-tuning/outputs/sdxl-wood-lora",
    repo_id="endlesstools/et-sdxl-lora-weights-v2",
    repo_type="model",
    commit_message="Upload LoRA weights"
)

print("✅ All files uploaded to Hugging Face Hub")
EOF

# start push_hf_trained.py
python push_hf_trained.py