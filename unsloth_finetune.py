from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from huggingface_hub import login
import os
import argparse
import wandb
import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# instruction = '''
# ### Instruction:
# You are an expert in Neo4j graph databases and Cypher query generation. 
# Your task is to generate a precise and executable Cypher query based solely on the provided QUESTION and SCHEMA.

# - Use only the Relationship properties and Node properties present in the SCHEMA.
# - Ensure the Cypher query answers the QUESTION using only the SCHEMA information.
# - Include final Cypher query after "### Cypher:", with no explanation or additional text.

# ### QUESTION:
# {question}

# ### SCHEMA:
# {schema}

# ### Response:
# <think>
# {CoT}
# </think>
# ### CYPHER:
# {cypher}
# '''
instruction = '''
### Instruction:
You are an expert in Neo4j graph databases and Cypher query generation. 
Your task is to generate a precise and executable Cypher query based solely on the provided QUESTION and SCHEMA.

- Use only the Relationship properties and Node properties present in the SCHEMA.
- Ensure the Cypher query answers the QUESTION using only the SCHEMA information.
- Include final Cypher query after "### Cypher:", with no explanation or additional text.

QUESTION:{question}.

SCHEMA:{schema}.

### Response:
<think>
{CoT}
</think>
### CYPHER:
{cypher}
'''

def formatting_prompts(examples, eos_token):
    question = examples["question"]
    schema = examples["schema"]
    output_cypher = examples['generated_cypher']
    CoT = examples['reasoning']
    texts = []
    for q, s, c, r in zip(question, schema, output_cypher, CoT):
        text = instruction.format(question=q, schema=s, cypher=c, CoT=r) + eos_token
        texts.append(text)

    return {
        "text": texts,
    }

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--wandb_project_name", type=str, default='llama-8b-instruct-finetune')
    parser.add_argument("--wandb_run_name", type=str, default='llama-r16-trueset-20percent-new')
    parser.add_argument("--train_on_correct_df", type=str, default="False")
    parser.add_argument("--adapter_save_path", type=str, default='finetuned_model/llama-r16-trueset-20percent-new')
    args = parser.parse_args()
    print(f"finetuning model name: {args.model_name}")
    print(f"train with only correct df: {args.train_on_correct_df}")
    print(f"the adapter save path: {args.adapter_save_path}")
    
    hf = os.environ.get('hf_token')
    login(hf)

    wb = os.environ.get('wandb_token')
    wandb.login(key=wb)
    run = wandb.init(
        project=args.wandb_project_name, 
        job_type="training", 
        anonymous="allow",
        name=args.wandb_run_name,
    )

    with open("configs/lora_finetune_config.yaml", "r") as f:
        config = yaml.safe_load(f)


    absolute_path = ''

    NEO4J_DATASET = os.path.join(absolute_path, 'evaluated_dataset/text2cypher_deepseekr1_sorted.csv')
    print("CSV Path:", NEO4J_DATASET)

    if os.path.exists(NEO4J_DATASET):
        df = pd.read_csv(NEO4J_DATASET)
        print('dataset loaded!')
    else:
        print(f"Error: File not found - {NEO4J_DATASET}")

    df = df[df['error_analysis'] == 'NoError']
    if args.train_on_correct_df == "True":
        df = df[df['exact_match'] == 1]
    else:
        df = df[df['exact_match'] == 0]
    # lower = df['completion_tokens'].quantile(0.4)
    # upper = df['completion_tokens'].quantile(0.9)
    # filtered_df = df[(df['completion_tokens'] > lower) & (df['completion_tokens'] < upper)]

    filtered_df = df.groupby('database_reference_alias').sample(frac=0.2, random_state=42)
    finetune_dataset = Dataset.from_pandas(filtered_df)
    print(f'len of df: {len(filtered_df)}')

    max_seq_length = 3072
    dtype = torch.bfloat16
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        model_name = args.model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    EOS_TOKEN = tokenizer.eos_token
    finetune_dataset = finetune_dataset.map(
        lambda batch: formatting_prompts(batch, eos_token=EOS_TOKEN),
        batched=True,
    )
    print(finetune_dataset[0]['text'])
    print(f'r is {config["lora_r"]}')
    print(f'lora_alpha is {config["lora_alpha"]}')
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config["lora_r"]),  
        target_modules=config["target_modules"],
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),  
        bias="none",  
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=int(config["seed"]),
        use_rslora=False,  
        loftq_config=None,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=finetune_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        args=TrainingArguments(
            # per_device_train_batch_size=2,
            per_device_train_batch_size=int(config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            num_train_epochs = int(config['num_train_epochs']), 
            warmup_ratio = float(config["warmup_ratio"]),
            # warmup_steps=5,
            # max_steps=10,
            learning_rate=float(config["learning_rate"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=int(config["logging_steps"]),
            optim=config["optim"],
            weight_decay=float(config["weight_decay"]),
            lr_scheduler_type="linear",
            seed=int(config["seed"]),
            output_dir=config["output_dir"],
        ),
    )

    print(f'device_batch_size: {trainer.args.per_device_train_batch_size}')
    print(f'gradient_accumulation_steps: {trainer.args.gradient_accumulation_steps}')
    start_time = time.time()

    print("Start training...")
    trainer_stats = trainer.train()

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_in_minutes = elapsed_time / 3600
    print(f"Training completed in {elapsed_time_in_minutes:.2f} hours")

    new_model_local = args.adapter_save_path
    model.save_pretrained(new_model_local) 
    tokenizer.save_pretrained(new_model_local)
    print(f"Model saved to {new_model_local}")
    # save merged model
    model = model.merge_and_unload()
    model.save_pretrained(f"{new_model_local}_merged")
    tokenizer.save_pretrained(f"{new_model_local}_merged")
    print(f"Model saved to {new_model_local}_merged")

if __name__ == "__main__":
    main()