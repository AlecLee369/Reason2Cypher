from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from huggingface_hub import login
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import argparse
import yaml
from peft import PeftModel, PeftConfig
from datasets import load_dataset, Dataset
from tqdm import tqdm
import time

hf = os.environ.get('hf_token')
login(hf)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
# !nvidia-smi

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
'''
def prepare_chat_prompt(question, schema) -> list[dict]:
    chat = [
        {
            "role": "user",
            "content": instruction.format(
                schema=schema, question=question
            ),
        }
    ]
    return chat
def _postprocess_output_cypher(_output_cypher: str) -> str:
    output_cypher = _output_cypher
    
    partition_by = "### Cypher:"
    if partition_by in output_cypher:
        _, _, output_cypher = output_cypher.partition(partition_by)
    partition_by = "### CYPHER:"
    if partition_by in output_cypher:
        _, _, output_cypher = output_cypher.partition(partition_by)
    output_cypher = output_cypher.strip("`\n")
    output_cypher = output_cypher.lstrip("cypher\n")
    output_cypher = output_cypher.strip("`\n ")
    output_cypher = output_cypher.replace(';', '')
    return output_cypher

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llama31-8b-r64")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_path", type=str, default='output_dataset/r64_generate_cypher.csv')
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    print(f"finetuning model name: {args.model_path}")
    print(f"batch size: {args.batch_size}")
    print(f"the output path: {args.output_path}")

    absolute_path = '' # download orginal text2cypher dataset: https://huggingface.co/datasets/neo4j/text2cypher-2024v1
    NEO4J_DATASET = os.path.join(absolute_path, 'neo4j_dataset/text2cypher-2024v1/data/test-00000-of-00001.parquet')
    print("CSV Path:", NEO4J_DATASET)

    if os.path.exists(NEO4J_DATASET):
        df = pd.read_parquet(NEO4J_DATASET, engine='pyarrow')
        print("DataFrame loaded successfully.")
    else:
        print(f"Error: File not found - {NEO4J_DATASET}")

    df_with_graph = df.dropna(subset=['database_reference_alias'])
    df_with_graph = df_with_graph.sort_values(by=['database_reference_alias'])
    df_with_graph["generated_cypher"] = None
    df_with_graph["CoT"] = None 

    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    print("dtype:", dtype)
    load_in_4bit = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    model.eval()

    BATCH_SIZE = args.batch_size
    cypher_outputs = []  # generated cypher output
    num_rows = len(df_with_graph)
    print(f'len of df: {num_rows}')

    model_generate_parameters = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    # model_generate_parameters = {
    #     "max_new_tokens": args.max_new_tokens,
    #     "top_p": 0.9,
    #     "temperature": 0.5,
    #     "do_sample": True,
    #     "pad_token_id": tokenizer.eos_token_id,
    # }
    print("Generating Cypher queries...")
    start_time = time.time()
    for batch_start in tqdm(range(0, num_rows, BATCH_SIZE), desc="Generating Cypher"):
        batch_end = min(batch_start + BATCH_SIZE, len(df_with_graph))
        batch_questions = df_with_graph.iloc[batch_start:batch_end]["question"].tolist()
        #print(batch_questions)
        batch_schemas = df_with_graph.iloc[batch_start:batch_end]["schema"].tolist()

        # Prepare prompts for batch
        batch_prompts = [
            instruction.format(question=q, schema=s)
            for q, s in zip(batch_questions, batch_schemas)
        ]

        # Tokenize inputs
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        # Generate Cypher queries in batch
        with torch.no_grad():
            tokens = model.generate(**inputs, **model_generate_parameters)
            tokens = tokens[:, inputs["input_ids"].shape[1]-3:]  # Remove input prompt from output
            raw_outputs = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            # store CoT
            df_with_graph.iloc[batch_start:batch_end, df_with_graph.columns.get_loc("CoT")] = pd.Series(raw_outputs).values
            output = [_postprocess_output_cypher(output) for output in raw_outputs]
            # Store results back in DataFrame
            df_with_graph.iloc[batch_start:batch_end, df_with_graph.columns.get_loc("generated_cypher")] = pd.Series(output).values
        if batch_start % 100 == 0:
            df_with_graph.to_csv(args.output_path, index=False)
            print(f"Processed batch {batch_start} - {batch_end}, progress saved!")
    
    df_with_graph.to_csv(args.output_path, index=False)
    print(f"All batches processed and saved to {args.output_path}.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = elapsed_time / 3600
    print(f"Elapsed time: {elapsed_time:.2f} hours")
    
if __name__ == "__main__":
    main()