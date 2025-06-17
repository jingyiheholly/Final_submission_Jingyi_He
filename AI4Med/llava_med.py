from transformers import AutoModelForCausalLM
CACHE_DIR="hf_models"
from llava.model.builder import load_pretrained_model
model_path='microsoft/llava-med-v1.5-mistral-7b'
model_base=None
model_name='llava-med-v1.5-mistral-7b'
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device="cuda")