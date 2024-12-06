import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import FastAPI
import os

# Set the HF_HOME environment variable
# os.environ["HF_HOME"] = "/root/autodl-tmp"

print("Hugging Face cache directory:", os.environ["HF_HOME"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ID = "nahgetout/test"
#'meta-llama/Meta-Llama-3-8B-Instruct'
#"nahgetout/test"
#'LoftQ/Meta-Llama-3-8B-Instruct-4bit-64rank'

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

def chat_with_model(prompt, max_length=1000):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = chat_with_model(user_input)
    print(f"Chatbot: {response}")