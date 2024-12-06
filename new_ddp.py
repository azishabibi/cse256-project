import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = "/root/autodl-tmp"

print("Hugging Face cache directory:", os.environ["HF_HOME"])

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

# Setup for Distributed Data Parallel (DDP)
def setup_ddp():
    # Get environment variables set by torch.distributed.launch
    rank = int(os.environ["RANK"])  # Global rank of the process
    local_rank = int(os.environ["LOCAL_RANK"])  # Local rank of the process
    torch.cuda.set_device(local_rank)  # Assign process to the correct GPU
    torch.distributed.init_process_group(backend="nccl")  # Initialize process group
    device = torch.device("cuda", local_rank)  # Define the device for this process
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    print(f"[init] Local Rank: {local_rank}, Global Rank: {rank}, Device: {device}")
    return local_rank, rank, device

# Initialize DDP
local_rank, rank, device = setup_ddp()

MODEL_ID = "LoftQ/Meta-Llama-3-8B-Instruct-4bit-64rank"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4',
    ),
)

peft_model = PeftModel.from_pretrained(
    base_model,
    MODEL_ID,
    subfolder="loftq_init",
    is_trainable=True,
)
peft_model = peft_model.to(device) 

# Wrap PEFT model with DDP
ddp_model = DDP(peft_model, device_ids=[local_rank], output_device=local_rank)

print(f"[DDP setup complete] Rank: {rank}, Local Rank: {local_rank}, Device: {device}")

del base_model
torch.cuda.empty_cache()
tokenizer.pad_token = tokenizer.eos_token

dataset2 = load_dataset("Multilingual-Multimodal-NLP/McEval-Instruct")
print(len(dataset2['train']))

def preprocess_function(example):
    instruction = example['instruction']
    response = example['output']
    input_text = instruction + tokenizer.eos_token + response + tokenizer.eos_token
    
    tokenized = tokenizer(
        input_text,
        max_length=2048,
        truncation=True,
        padding='max_length',
    )
    
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    
    instruction_tokenized = tokenizer(
        instruction + tokenizer.eos_token,
        max_length=2048,
        truncation=True,
        padding=False,
    )
    instruction_length = len(instruction_tokenized['input_ids'])
    # Create labels by masking the instruction part with -100
    labels = input_ids.copy()
    labels[:instruction_length] = [-100] * instruction_length
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

tokenized_dataset = dataset2.map(preprocess_function, remove_columns=dataset2['train'].column_names)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args = TrainingArguments(
    output_dir="/root/autodl-tmp/fine_tuned_model", # Change the line if needed
    evaluation_strategy="no",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,  # Warm up for 10% of training
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="/root/autodl-tmp/logs", # Change the line if needed
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,  
    push_to_hub=False,
    local_rank=local_rank 
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()

if local_rank == 0:  # Save only on main process
    trainer.save_model("/root/autodl-tmp/fine_tuned_model_realuser_instruct")
    tokenizer.save_pretrained("/root/autodl-tmp/fine_tuned_model_realuser_instruct")
