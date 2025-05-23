import os
import re
import requests
import json
import math
from tqdm import tqdm


def get_token_probabilities(server_url: str, input_text: str, target_token=None, top_k: int = 10) -> dict:
    """
    Call the /v1/token/probabilities endpoint to retrieve next-token distribution
    or conditional probability for a specific token or list of tokens.

    Args:
        server_url: Base URL of the probability server (e.g., "http://localhost:4000").
        input_text: Prompt text to send to the server.
        target_token: Token or list of tokens for which to return probabilities.
        top_k: Number of top tokens to request when target_token is None.

    Returns:
        Parsed JSON response as a dict, or None if the request fails.
    """
    endpoint = f"{server_url}/v1/token/probabilities"
    payload = {"input_text": input_text, "top_k": top_k}
    if target_token is not None:
        payload["target_token"] = target_token

    try:
        response = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as err:
        print(f"[Error] Probability request failed: {err}")
        return None


def count_ordered_options(text: str, is_math: bool) -> int:
    """
    Count consecutive lettered options (A, B, C, ...) in a regular multi-choice prompt.
    Returns 0 for math-based problems.

    Args:
        text: Full question text containing options.
        is_math: Flag indicating if the question is math (skip option counting).

    Returns:
        Integer count of options starting from 'A'.
    """
    if is_math:
        return 0

    letters = re.findall(r"([A-Z])(?:\.|\))", text)
    count, expected_ord = 0, ord('A')
    for ch in letters:
        if ord(ch) == expected_ord:
            count += 1
            expected_ord += 1
        else:
            break
    return count


def parse_log_file(file_path: str, mode: str = "multi") -> list:
    """
    Parse a log file of model runs and extract entry fields.

    Args:
        file_path: Path to the .log file.
        mode: 'single' for every-line parsing or 'multi' for every-5th-line parsing.

    Returns:
        List of dict entries with keys: idx, total, correct, model_answer,
        ground_truth, response_length, option_num, question, model_response
    """
    is_math = any(tag in file_path for tag in ['gsm8k', 'aime2025'])
    entries = []

    pattern = (
        r"idx:(\d+)/(\d+)\s+(True|False)\s+"  # idx, total, correct
        r"(.+?)\s+(\S+)\s+(\d+)\s+"          # model_answer, ground_truth, length
        r"Question:\s+\[(.*?)\]\s+"          # question text
        r"Model Response:\s+\[(.*?)\]"         # model_response text
    )

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if mode == 'multi' and i % 5 != 0:
                continue
            m = re.match(pattern, line)
            if not m:
                continue

            q_text = m.group(6).strip()
            resp_text = m.group(7).strip()
            entries.append({
                'idx': int(m.group(1)),
                'total': int(m.group(2)),
                'correct': (m.group(3) == 'True'),
                'model_answer': m.group(4),
                'ground_truth': m.group(5),
                'response_length': int(m.group(6)),
                'option_num': count_ordered_options(q_text, is_math),
                'question': q_text,
                'model_response': resp_text
            })
    return entries


def format_response(question: str, think_path: str, split_mode: str = "paragraph") -> list:
    """
    Split a think-path into incremental prompts for entropy calc.

    Args:
        question: The question text.
        think_path: Full model-generated think sequence.
        split_mode: 'sentence' or 'paragraph' splitting strategy.

    Returns:
        List of partial prompt strings ending with a cue for the correct option.
    """
    prompts = []
    if split_mode == 'sentence':
        sentences = [s.strip() + '.' for s in think_path.split('.') if s.strip()]
        for i in range(1, len(sentences) + 1):
            partial = ' '.join(sentences[:i])
            prompts.append(f"Question: {question}\n\nThink: {partial}\n\nThe correct option letter is:")
    else:
        paras = [p.strip() for p in think_path.split('\n\n') if p.strip()]
        merged, buf = [], ''
        for p in paras:
            if len(buf) + len(p) < 120:
                buf = (buf + ' ' + p).strip()
            else:
                merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)

        for segment in merged:
            prompts.append(
                f"Question: {question}\n\n<think>\n{segment}\n</think>\n\nThe correct option letter is:"
            )
    return prompts


def get_conditional_entropy(log_file_path: str, server_url: str, output_dir: str) -> None:
    """
    Compute conditional entropy of the correct answer at each incremental think step.
    Saves results as JSON in the specified output directory.

    Args:
        log_file_path: Path to input .log.
        server_url: URL of the probability server.
        output_dir: Directory where output JSON will be saved.
    """
    entries = parse_log_file(log_file_path)
    results = {}

    for entry in tqdm(entries, desc='Processing entries'):
        idx = entry['idx']
        # Build list of option tokens with leading space
        options = [f" {chr(code)}" for code in range(65, 65 + entry['option_num'])]
        steps = format_response(entry['question'], entry['model_response'], split_mode='paragraph')

        all_entropy = []
        ans_prob_norm = []
        ans_ent_norm = []

        for step in steps:
            data = get_token_probabilities(server_url, step, target_token=options)
            probs = data.get('probability', [])
            total_p = sum(probs)
            # Compute entropy across all options
            ent_all = -sum((p/total_p) * math.log2(p/total_p) for p in probs if p > 0)
            all_entropy.append(ent_all)

            # Normalized probability for the correct answer
            correct_tok = f" {entry['ground_truth']}"
            p_corr = probs[options.index(correct_tok)]
            p_norm = p_corr / total_p
            ans_prob_norm.append(p_norm)
            ent_norm = -p_norm * math.log2(p_norm) if p_norm > 0 else 0
            ans_ent_norm.append(ent_norm)

        results[idx] = {
            'idx': idx,
            'question': entry['question'],
            'ground_truth': entry['ground_truth'],
            'model_response': entry['model_response'],
            'think_steps': len(steps),
            'all_options_entropy': all_entropy,
            'answer_prob_norm': ans_prob_norm,
            'answer_entropy_norm': ans_ent_norm,
            'all_options_entropy_avg': sum(all_entropy) / len(all_entropy) if all_entropy else 0,
            'answer_entropy_norm_avg': sum(ans_ent_norm) / len(ans_ent_norm) if ans_ent_norm else 0
        }

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(log_file_path).replace('.log', '_info_gain.json')
    out_file = os.path.join(output_dir, base)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # Load server URL and paths
    server_url = 'http://localhost:4000'
    cwd = os.getcwd()
    log_file = os.path.join(cwd, 'commonsenseqa_QwQ-32B_True.log')
    output_dir = os.path.join(cwd, 'think_results')

    get_conditional_entropy(log_file, server_url, output_dir)
