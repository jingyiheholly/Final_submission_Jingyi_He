from transformers import AutoTokenizer, AutoModelForCausalLM

cache_directory = "/root/autodl-tmp/hf_model"  # change this to your desired location

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    cache_dir=cache_directory
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    cache_dir=cache_directory
)
