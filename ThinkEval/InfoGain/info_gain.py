import os
import re
import requests
import json
import math
from tqdm import tqdm

# Function to query token probabilities from a server endpoint

def get_token_probabilities(server_url: str, input_text: str, target_token=None, top_k: int = 10) -> dict:
    """
    Call the /v1/token/probabilities endpoint to retrieve either the next-token distribution
    or the conditional probability of a specific token sequence.

    Args:
        server_url: Base URL of the server, e.g., "http://localhost:4000".
        input_text: Text prompt to send.
        target_token: Specific token or token list to compute conditional probability for.
        top_k: Number of top tokens to return when target_token is None.

    Returns:
        JSON response as a Python dict, or None on failure.
    """
    endpoint = f"{server_url}/v1/token/probabilities"
    payload = {"input_text": input_text, "top_k": top_k}
    if target_token is not None:
        payload["target_token"] = target_token

    try:
        res = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
        res.raise_for_status()
        return res.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Function to count ordered multiple-choice options (A, B, C, ...)

def count_ordered_options(text: str, is_math: bool) -> int:
    """
    Count consecutive lettered options starting from 'A' in the given text.
    Returns 0 for math questions.
    """
    if is_math:
        return 0

    options = re.findall(r"([A-Z])(?:\.|\))", text)
    if 'A' not in options:
        return 0
    count = 0
    expected = ord('A')
    for opt in options:
        if ord(opt) == expected:
            count += 1
            expected += 1
        else:
            break
    return count

# Function to parse a log file and extract entries

def parse_log_file(file_path: str, mode: str = "multi") -> list[dict]:
    """
    Read a log file and extract fields per entry.
    Supports 'single' (every line) or 'multi' (every 5th line) modes.

    Args:
        file_path: Path to the log file.
        mode: Parsing mode, either 'single' or 'multi'.

    Returns:
        List of dicts with parsed fields.
    """
    is_math = any(tag in file_path for tag in ['gsm8k', 'aime2025'])
    entries = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if mode == 'multi' and idx % 5 != 0:
                continue
            # Regex to capture idx/total, correctness, model answer, ground truth, response length, question, and model response
            pattern = (
                r"idx:(\d+)/(\d+)\s+(True|False)\s+"  # idx, total, correctness
                r"(.+?)\s+(\S+)\s+(\d+)\s+"         # model_answer, ground_truth, length
                r"Question:\s+\[(.*?)\]\s+"         # question
                r"Model Response:\s+\[(.*?)\]"       # model_response
            )
            match = re.match(pattern, line)
            if not match:
                continue

            q_text = match.group(6).strip()
            resp_text = match.group(7).strip()

            entry = {
                'idx': int(match.group(1)),
                'total': int(match.group(2)),
                'correct': match.group(3) == 'True',
                'model_answer': match.group(4),
                'ground_truth': match.group(5),
                'response_length': int(match.group(6)),
                'option_num': count_ordered_options(q_text, is_math),
                'question': q_text,
                'model_response': resp_text
            }
            entries.append(entry)
    return entries

# Function to format steps of a think path for entropy calculations

def format_response(question: str, think_path: str, split_mode: str = "paragraph") -> list[str]:
    """
    Generate a sequence of partial prompts splitting the think path by paragraphs or sentences.
    Each prompt ends with a cue for the correct option letter.
    """
    prompts = []
    if split_mode == 'sentence':
        sentences = [s.strip() + '.' for s in think_path.split('.') if s.strip()]
        for i in range(1, len(sentences)+1):
            partial = ' '.join(sentences[:i])
            prompts.append(f"Question: {question}\n\nThink: {partial}\n\nThe correct option letter is:")
    else:
        paras = [p.strip() for p in think_path.split('\n\n') if p.strip()]
        merged, buf = [], ''
        for p in paras:
            if len(buf) + len(p) < 120:
                buf += (' ' + p).strip()
            else:
                merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        for segment in merged:
            prompts.append(f"Question: {question}\n\n<think>\n{segment}\n</think>\n\nThe correct option letter is:")
    return prompts

# Main function to compute conditional entropy for each think step

def get_conditional_entropy(log_file_path: str, server_url: str, output_dir: str):
    """
    For each entry in the log, compute conditional entropy over all options
    and for the correct answer at each incremental think step.
    Results are saved as a JSON in the output directory.
    """
    entries = parse_log_file(log_file_path)
    results = {}
    for e in tqdm(entries, desc='Processing entries'):
        idx = e['idx']
        options = [f" {chr(i)}" for i in range(65, 65 + e['option_num'])]
        steps = format_response(e['question'], e['model_response'], split_mode='paragraph')

        all_entropy, ans_entropy_norm, ans_prob_norm = [], [], []
        for step in steps:
            prob_data = get_token_probabilities(server_url, step, target_token=options)
            probs = prob_data['probability']
            # entropy over all options
            total_ent = -sum(p / sum(probs) * math.log2(p / sum(probs)) for p in probs if p>0)
            all_entropy.append(total_ent)

            # normalize answer probability
            ans_p = probs[options.index(' ' + e['ground_truth'])]
            norm_p = ans_p / sum(probs)
            ans_prob_norm.append(norm_p)
            ent_norm = -norm_p * math.log2(norm_p) if norm_p>0 else 0
            ans_entropy_norm.append(ent_norm)

        results[idx] = {
            'idx': idx,
            'question': e['question'],
            'ground_truth': e['ground_truth'],
            'model_response': e['model_response'],
            'think_steps': len(steps),
            'all_options_entropy': all_entropy,
            'answer_prob_norm': ans_prob_norm,
            'answer_entropy_norm': ans_entropy_norm,
            'all_options_entropy_avg': sum(all_entropy)/len(all_entropy),
            'answer_entropy_norm_avg': sum(ans_entropy_norm)/len(ans_entropy_norm)
        }

    # Write output to JSON file next to the log
    fname = os.path.basename(log_file_path).replace('.log', '_info_gain.json')
    out_path = os.path.join(output_dir, fname)
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # Server configuration and log paths
    server_url = 'http://localhost:4000'
    cwd = os.getcwd()
    log_file = os.path.join(cwd, 'commonsenseqa_QwQ-32B_True.log')
    output_dir = os.path.join(cwd, 'think_results')

    get_conditional_entropy(log_file, server_url, output_dir)
    