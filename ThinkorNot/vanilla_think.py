import os
import random
import torch
import requests
import json
import argparse
import numpy as np
from tqdm import tqdm
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluation Pipeline")
    parser.add_argument('--dataset', required=True, help="Dataset name (e.g., gsm8k, mmlu)")
    parser.add_argument('--model', required=True, help="Model name or path")
    parser.add_argument('--think', action='store_true', help="Enable think mode")
    parser.add_argument('--iter', type=int, default=5, help="Number of iterations for each question")
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
    """Create the JSON payload for the chat request."""
    return {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    }

def get_model_response(messages, port, think_mode=False):
    """Send a request to the server and get the model response."""
    url = SERVER_URL.format(port)
    headers = {"Content-Type": "application/json"}
    data = create_chat_request(messages, max_tokens=32768)
    data["think_mode"] = think_mode
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

    # Prepare output file
    output_dir = os.path.join(os.getcwd(), "new/base_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{args.dataset}_{args.model.split('/')[-1]}_{args.think}.log")

    correct_indices = []
    incorrect_indices = []

    with open(output_file_path, 'w', encoding='utf-8') as f:
        idx = 0
        return_dict = dict()
        for i in range(args.iter):
            return_dict[i] = {"correct": [], "token_count": []}

        # Iterate through the dataset
        for question in tqdm(dataset):
            idx += 1
            if args.dataset == "gsm8k":
                messages = question['zs_cot_messages']
            else:
                messages = question['message']

            iter_correct = []
            iter_token_counts = []

            attempt = 0
            think_error_count = 0  # Counter for "Think mode is disabled" errors
            format_error_count = 0  # Counter for format errors

            while attempt < args.iter:
                try:
                    model_answer = get_model_response(messages, port=args.port, think_mode=args.think)
                    model_response = [model_answer["content"]]
                    
                    # Check for think mode error
                    if args.think == False and "</think>" in model_response[0]:
                        think_error_count += 1
                        if think_error_count >= 5:
                            print(f"Warning: Ignoring 'Think mode is disabled' error after 5 occurrences for question {idx}.")
                            think_error_count = 0  # Reset the counter
                        else:
                            raise ValueError("Think mode is disabled, but </think> tag found in the response.")

                    token_count = model_answer["token_count"]

                    try:
                        metrics = dataset.evaluate_response(model_response, question)
                        answer_span_in_response = metrics[0]["answer_span"]
                        extracted_answer = metrics[0]["model_response"][answer_span_in_response[0]:answer_span_in_response[1]]

                    except Exception as e:
                        format_error_count += 1
                        if format_error_count >= 5:
                            print(f"Warning: Ignoring format error after 5 occurrences for question {idx}.")
                            metrics = [
                                {
                                    'model_response': model_response,
                                    'answer_line': None,
                                    'correct': False,
                                    'answer_randomly_sampled': False,
                                    'answer_span': None,
                                    'model_answer': None,
                                    'raw_model_answer': None,
                                    **question
                                }
                            ]
                            format_error_count = 0
                            extracted_answer = None
                        else:
                            raise ValueError(f"Error in evaluating response: {e}")
                
                    iter_correct.append(metrics[0]['correct'])
                    return_dict[attempt]["correct"].append(metrics[0]['correct'])
                    return_dict[attempt]["token_count"].append(token_count)

                    # Write results to the file
                    f.write(
                        f"idx:{idx}/{len(dataset)}\t"
                        f"{metrics[0]['correct']}\t"
                        f"{extracted_answer}\t"
                        f"{question['answer']}\t"
                        f"{token_count}\t"
                        f"Question: {[question['question']]}\t"
                        f"Model Response: {model_response}\n"
                    )

                    attempt += 1  # Increment only on successful iteration
                except Exception as e:
                    print(f"Error occurred during iteration {attempt + 1}: {e}. Skipping this iteration.")

            # Calculate average and variance for correctness and token counts
            avg_correct = np.mean(iter_correct)

            # Determine if the question is overall correct
            overall_correct = avg_correct > 0.5
            if overall_correct:
                correct_indices.append(idx)
            else:
                incorrect_indices.append(idx)
            
            # break  # Uncomment this line to stop after the first question for testing

        # Calculate overall accuracy and token statistics
        overall_accuracy = np.mean([np.mean(return_dict[i]["correct"]) for i in range(args.iter)])
        overall_token_avg = np.mean([np.mean(return_dict[i]["token_count"]) for i in range(args.iter)])
        overall_accuracy_std = np.std([np.mean(return_dict[i]["correct"]) for i in range(args.iter)])
        overall_token_std = np.std([np.mean(return_dict[i]["token_count"]) for i in range(args.iter)])

        f.write(f"-----------------------------------------\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.8f}\n")
        f.write(f"Overall Token Average: {overall_token_avg:.8f}\n")
        f.write(f"Overall Accuracy Standard Deviation: {overall_accuracy_std:.8f}\n")
        f.write(f"Overall Token Standard Deviation: {overall_token_std:.8f}\n")
        f.write(f"Correct Indices: {correct_indices}\n")
        f.write(f"Incorrect Indices: {incorrect_indices}\n")
        f.write(f"-----------------------------------------\n")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main()