import os
import random
import torch
import requests
import json
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
import numpy as np
import time
from eval_datasets.types.gsm8k import GSM8KDataset
from eval_datasets.types.commonsenseqa import CommonSenseQADataset
from eval_datasets.types.musr import MuSRDataset
from eval_datasets.types.mmlu_pro import MMLUProDataset
from eval_datasets.types.prontoqa import ProntoQADataset
from eval_datasets.types.aime2025 import AIME2025Dataset

SERVER_URL = "http://localhost:{}/v1/chat/completions"  # Dynamic port in the URL

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "commonsenseqa": CommonSenseQADataset,
    "musr": MuSRDataset,
    "mmlu_pro": MMLUProDataset,
    "prontoqa": ProntoQADataset,
    "aime2025": AIME2025Dataset
}

def get_parser():
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--iter', type=int, default=5, help="Number of iterations for each question")
    parser.add_argument('--sample', type=int, default=10000, help="Number of questions to sample")
    parser.add_argument('--port', type=int, default=4000, help="Port to run the server on")
    return parser

def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_chat_request(messages, max_tokens):
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    }

def get_model_response(messages, port, think_mode=False):
    url = SERVER_URL.format(port)  # Use the port from the command-line argument
    headers = {
        "Content-Type": "application/json"
    }

    data = create_chat_request(messages, max_tokens=32768)
    data["think_mode"] = think_mode  # Add think_mode to the request data
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

def main():
    # Set random seed for reproducibility
    set_random_seed(42)

    # Load the dataset
    if args.dataset not in DATASET_MAP:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    dataset = DATASET_MAP[args.dataset]()

    # Open a file to write the results
    current_directory = os.getcwd()  # Get current working directory
    output_file_path = os.path.join(current_directory, "vanilla_thinkornot_results", f"{args.dataset.split('/')[-1]}_{args.model.split('/')[-1]}.log")
    ourdir = os.path.dirname(output_file_path)
    os.makedirs(ourdir, exist_ok=True)
    if os.path.exists(output_file_path):
        output_file_path = os.path.join(current_directory, "vanilla_thinkornot_results", f"{args.dataset.split('/')[-1]}_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d%H%M%S')}.log")
    f = open(output_file_path, 'w', encoding='utf-8')

    # Initialize statistics
    stats = {
        "idx": 0,
        "correct_count": 0,
        "token_num": 0,
        "not_extracted_count": 0,
        "think_count": 0,
        "nothink_count": 0,
        "think_correct_count": 0,
        "nothink_correct_count": 0,
        "correct_indices" : [],
        "incorrect_indices" : []
    }

    # Evaluate the model on the dataset
    # Sample a subset of the dataset if specified
    if args.sample > 0 and args.sample < len(dataset):
        dataset.sample(args.sample)
        print(f"Sampled {args.sample} questions from the dataset.")
    else:
        print(f"Using the entire dataset with {len(dataset)} questions.")

    for question in tqdm(dataset):
        stats["idx"] += 1
        if args.dataset == "gsm8k":
            messages = question['zs_cot_messages']
            # print(f"Question: {messages}")
        else:
            messages = question['message']

        think_prompt = [
            {"role": "system", "content": DEEP_THINK_PROMPT},
            {"role": "user", "content": "Does this question require deep thinking to answer? Please first respond with 'yes' or 'no', then provide your reasons."},
            {"role": "user", "content": question['question']}
        ]

        try_times = 0
        think_error_count = 0  # Counter for "Think mode is disabled" errors
        format_error_count = 0  # Counter for format errors
        while True:
            try:
                think_flag = False
                try_times += 1
                think_response = get_model_response(think_prompt, args.port, think_mode=False)
                think_decision = think_response["content"].strip().lower()
                # print(f"Think Decision: {think_decision}")

                if "yes" in think_decision:
                    # If thinking is required, use the think model
                    model_answer = get_model_response(messages, args.port, think_mode=True)
                    think_flag = True
                    # print(f"Think Mode: {model_answer['content']}")
                else:
                    # If thinking is not required, use the regular model
                    while True:
                        model_answer = get_model_response(messages, args.port, think_mode=False)
                        think_flag = False
                        print(f"No Think Mode: {model_answer['content']}")
                        if "</think>" in model_answer["content"]:
                            think_error_count += 1
                            if think_error_count > 5:
                                print(f"Warning: Ignoring 'Think mode is disabled' error after 5 occurrences for question {stats['idx']}.")
                                think_error_count = 0
                                break
                            else:
                                print(f"Think mode is disabled, but </think> tag found in the response. Retrying...")
                        else:
                            break

                model_response = [model_answer["content"]]
                token_count = model_answer["token_count"]
                metrics = dataset.evaluate_response(model_response, question)
                answer_span_in_response = metrics[0]["answer_span"]
                extracted_answer = metrics[0]["model_response"][answer_span_in_response[0]:answer_span_in_response[1]]
                break  # If all lines succeed, break out of the loop
            except Exception as e:
                if try_times <= 5:
                    print(f"Error occurred: {e}. Retrying...")
                else:
                    extracted_answer = "Not Extracted!"
                    stats["not_extracted_count"] += 1
                    metrics[0]['correct'] = False
                    break
        
        # Update the statistics
        if think_flag:
            stats["think_count"] += 1
        else:
            stats["nothink_count"] += 1

        if metrics[0]['correct']:
            stats["correct_count"] += 1
            if "yes" in think_decision:
                stats["think_correct_count"] += 1
            else:
                stats["nothink_correct_count"] += 1
        stats["token_num"] += token_count

        # Write the results to the output file
        f.write(f"idx:{stats['idx']}/{len(dataset)}\t" + f"{metrics[0]['correct']}\t" + f"{extracted_answer}\t" + f"{question['answer']}\t" + f"{token_count}\t" + f"Question: {[question['question']]}\t" + f"Model Response: {model_response}\n")
    
    # Calculate the accuracy
    accuracy = stats["correct_count"] / stats["idx"]
    token_avg = stats["token_num"] / stats["idx"]
    think_accuracy = stats["think_correct_count"] / stats["think_count"] if stats["think_count"] > 0 else 0
    nothink_accuracy = stats["nothink_correct_count"] / stats["nothink_count"] if stats["nothink_count"] > 0 else 0
    f.write(f"-----------------------------------------\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Token Average: {token_avg}\n")
    f.write(f"Not Extracted Count: {stats['not_extracted_count']}\n")
    f.write(f"Think Count: {stats['think_count']}\n")
    f.write(f"No Think Count: {stats['nothink_count']}\n")
    f.write(f"Think Accuracy: {think_accuracy}\n")
    f.write(f"No Think Accuracy: {nothink_accuracy}\n")
    f.write(f"-----------------------------------------\n")
    f.close()


DEEP_THINK_PROMPT = """
You are an intelligent reasoning assistant. Upon receiving a question, you must determine whether it requires Deep Think Mode—which involves rigorous, multi-step, and systematic complex reasoning.

### **Evaluation Criteria (At least TWO must be met to trigger Deep Think Mode)**  
1. **Cannot be answered directly based on the question itself**  
   - The answer is not immediately apparent from general knowledge, simple reasoning, or single-step calculations.  
   - The question requires combining multiple knowledge points, hidden conditions, or assumptions.  

2. **Multi-step reasoning & information integration**  
   - The solution involves **sequential logical steps**, where each step depends on previous conclusions.  
   - Multiple data sources, conditions, or assumptions must be **synthesized** to derive the final answer.  

3. **Strict mathematical/logical proof or recursive deduction**  
   - The problem requires **formal proof** (e.g., deductive reasoning, axiomatic proofs).  
   - It involves recursive reasoning, mathematical induction, or constructing counterexamples.  

4. **Non-trivial strategy or non-unique solution**  
   - The question requires evaluating multiple potential solutions and **choosing the optimal one** (e.g., game theory, complex planning).  
   - There may be **multiple valid approaches**, requiring deep analysis and comparison.  

5. **Systematic reasoning & hypothesis-based deduction**  
   - The question requires establishing hypotheses and **systematically deriving** conclusions (e.g., scientific theories, economic models).  
   - Multiple variables and complex relationships are involved, requiring a rigorous analytical process.  

### **Output Format**  
- **"YES"** (Deep Think Mode required)  
  - If the question meets **at least 2 criteria**, return **"YES"** and briefly explain why.  
- **"NO"** (Deep Think Mode not required; standard reasoning is sufficient)  
  - If the question only requires basic or short-step reasoning, return **"NO"** and explain why it can be answered directly.  

### **Examples**  
#### **Requires Deep Think**  
- **Input**: "Let A, B, and C be three sets. Prove that A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)."  
  - **Output**: "YES - This problem involves set operations and requires a formal mathematical proof with multi-step logical deductions."  

- **Input**: "If the speed of light is the cosmic limit, but the universe is expanding, is it possible for two regions to be permanently unobservable from each other?"  
  - **Output**: "YES - This question involves relativity, cosmology, and hypothesis-based deduction, requiring systematic reasoning."  

- **Input**: "On an 8x8 chessboard, if two opposite corners are removed, can it be completely covered by 2x1 dominoes?"  
  - **Output**: "YES - This requires constructing a counterexample, analyzing the board’s parity, and recursive reasoning."  

#### **Does Not Require Deep Think**  
- **Input**: "What is 2 to the power of 10?"  
  - **Output**: "NO - This is a straightforward computation that can be answered directly."  

- **Input**: "Tom is 5 years older than Alice. Alice is 10 years old. How old is Tom?"  
  - **Output**: "NO - This is a basic arithmetic problem that does not require complex reasoning."  

- **Input**: "Why is water heavier than oil?"  
  - **Output**: "NO - This is a factual question about density that can be answered using common knowledge."
"""


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main()