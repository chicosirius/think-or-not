import os
import json
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Configuration
DATASET_NAME = 'openai/gsm8k'
DATASET_SPLIT = 'main'
MODEL_ID = 'meta-llama/Llama-3.1-70B-Instruct'
OUTPUT_FILENAME = 'rewrite_gsm8k_10.log'
NUM_SAMPLES_TO_GENERATE = 10
MAX_TOKENS = 1024
TEMPERATURE = 0.8
TOP_P = 1.0
REPETITION_PENALTY = 1.05

# Determine output path in current working directory
output_path = os.path.join(os.getcwd(), OUTPUT_FILENAME)

# Load dataset
print('Loading dataset...')
ds = load_dataset(DATASET_NAME, DATASET_SPLIT)

# Initialize tokenizer and model
print('Initializing tokenizer and model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = LLM(
    model=MODEL_ID,
    tensor_parallel_size=torch.cuda.device_count(),
    trust_remote_code=True
)

# Prompt template builder
prompt_template = (
    'You will be given a problem-solving process. '
    'Please rewrite this process without changing its logic or content. '
    'Ensure that the output includes only the rewritten process and nothing else.\n\n'
    '**Problem-Solving Process:**\n{input}\n\n**Rewritten Process:**'
)

def rewrite_process(original: str, k: int) -> list[str]:
    prompt = prompt_template.format(input=original)
    chat = [
        {'role': 'system', 'content': 'You are a helpful AI assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        max_tokens=MAX_TOKENS
    )
    outputs = []
    with torch.no_grad():
        for _ in range(k):
            out = model.generate(inputs, params)
            text = out[0].outputs[0].text.strip()
            outputs.append(text)
    return outputs

# Main script
if __name__ == '__main__':
    total = len(ds['test'])
    print(f'Rewriting {total} examples...')

    # Ensure log file exists
    open(output_path, 'a', encoding='utf-8').close()

    with open(output_path, 'a', encoding='utf-8') as f:
        for idx, sample in enumerate(tqdm(ds['test'], desc='Processing samples'), start=1):
            question = sample['question']
            true_answer = sample['answer']
            candidates = rewrite_process(true_answer, NUM_SAMPLES_TO_GENERATE)

            f.write(
                f"idx:{idx}/{total}\t"
                f"Question: {[question]}\t"
                f"True Answer: {[true_answer]}\t"
                f"Generated Answers: {candidates}\n"
            )

    print(f'All results have been saved to {output_path}.')
