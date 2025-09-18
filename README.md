# et_train_t2i_lora_sdxl

# При создании под нужно брать темплейт
Runpod Pytorch 2.4.0 
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Диск должен быть 40/50гб 

# Обновим 
pip install --upgrade pip 

# Устанавливаем diffusers, transformers, accelerate и прочие 
pip install transformers==4.44.2 accelerate==1.1.1 datasets==2.21.0 safetensors==0.4.5 
pip install git+https://github.com/huggingface/diffusers 

# проверка 
python -c "import diffusers; print(diffusers.__version__)" 

# Для изображений и прогресса 
pip install pillow tqdm peft bitsandbytes 

# Авторизация в Hugging Face 
huggingface-cli login 

# Создаём рабочие директории: 
mkdir -p /workspace/fine-tuning/scripts 
mkdir -p /workspace/fine-tuning/outputs 

# заходим 
cd /workspace/fine-tuning/scripts 

# нужно качать именно et_train_t2i_lora_sdxl.py 
wget https://raw.githubusercontent.com/endless-work/et_train_t2i_lora_sdxl/main/et_train_t2i_lora_sdxl.py