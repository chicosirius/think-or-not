import os
import re
import time
import numpy as np
import requests
import json
import math
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

class ProblemType(Enum):
    """Enumeration for different problem types."""
    MULTIPLE_CHOICE = "multiple_choice"
    MATH = "math"

class ResponseFormat(Enum):
    """Enumeration for different response formats."""
    OPTION_LETTER = "option_letter"
    MATH_BOX = "math_box"

class SplitMode(Enum):
    """Enumeration for different text splitting modes."""
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"

@dataclass
class LogEntry:
    """Data class for storing log entry information."""
    idx: int
    total: int
    correct: bool
    model_answer: str
    ground_truth: str
    response_length: int
    option_num: Optional[int]
    question: str
    model_response: str

class DynamicThinkAnalyzer:
    """Main class for dynamic think analysis of model responses."""
    
    def __init__(self, config: Dict):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.server_url = config.get("server_url", "http://localhost:3000")
        self.top_k = config.get("top_k", 5)
        self.max_steps = config.get("max_steps", 10)
        self.ratio = config.get("ratio", 0.3)
        self.merged_length = config.get("merged_length", 200)
        self.max_entropy = 1 / (math.e * math.log(2))
        self.threshold = self.max_entropy * self.ratio
        self.problem_type = ProblemType.MULTIPLE_CHOICE
        self.response_format = ResponseFormat.OPTION_LETTER
        self.split_mode = SplitMode.PARAGRAPH

    def get_token_probabilities(self, input_text: str, target_token: Optional[str] = None) -> Optional[Dict]:
        """
        Get token probabilities from the server.
        
        Args:
            input_text: Input text to analyze
            target_token: Specific token to get probability for
            
        Returns:
            Dictionary containing probability information or None if request fails
        """
        url = f"{self.server_url}/v1/token/probabilities"
        payload = {"input_text": input_text, "top_k": self.top_k}
        if target_token:
            payload["target_token"] = target_token

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def get_token_count(self, text: str) -> Optional[int]:
        """Get token count for given text."""
        url = f"{self.server_url}/v1/token/count"
        try:
            response = requests.post(url, json={"text": text})
            response.raise_for_status()
            return response.json().get("token_count")
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def count_ordered_options(self, text: str) -> Optional[int]:
        """
        Count ordered options (A, B, C, etc.) in the text.
        
        Args:
            text: Text containing options
            
        Returns:
            Number of ordered options found or None for math problems
        """
        if self.problem_type == ProblemType.MATH:
            return None

        options = re.findall(r"([A-Z])[\.\)]", text)
        try:
            start_index = options.index('A')
        except ValueError:
            return 0

        count = 1
        current_char_code = ord('A') + 1
        for opt in options[start_index + 1:]:
            if ord(opt) == current_char_code:
                count += 1
                current_char_code += 1
            if current_char_code > ord('Z'):
                break
        return count

    def parse_log_file(self, file_path: str, mode: str = "multi") -> List[LogEntry]:
        """
        Parse log file and extract entries.
        
        Args:
            file_path: Path to log file
            mode: Parsing mode ('single' or 'multi')
            
        Returns:
            List of parsed log entries
        """
        # Determine problem type from filename
        if "gsm8k" in file_path or "aime2025" in file_path:
            self.problem_type = ProblemType.MATH
            self.response_format = ResponseFormat.MATH_BOX

        entries = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if mode == "multi" and i % 5 != 0:
                    continue
                    
                match = re.match(
                    r"idx:(\d+)/(\d+)\t+(True|False)\t+([\s\S]*?)\t+(.+?)\t+(\d+)\t+Question:\s+\[(.*?)\]\t+Model Response:\s+\[(.*?)\]",
                    line
                )
                if match:
                    q = line.split("Question: [")[1].split("]	Model Response: [")[0]
                    r = line.split("Model Response: [")[1]
                    
                    entry = LogEntry(
                        idx=int(match.group(1)),
                        total=int(match.group(2)),
                        correct=match.group(3) == "True",
                        model_answer=match.group(4),
                        ground_truth=match.group(5),
                        response_length=int(match.group(6)),
                        option_num=self.count_ordered_options(q),
                        question=q.strip("\'").strip("\"").encode('utf-8').decode('unicode_escape', errors='ignore'),
                        model_response=r.strip("]").strip("\'").strip("\"").encode('utf-8').decode('unicode_escape', errors='ignore'),
                    )
                    entries.append(entry)
        return entries

    def format_response(self, question: str, think_path: str) -> List[str]:
        """
        Format response for analysis based on problem type.
        
        Args:
            question: The question text
            think_path: The model's thinking path
            
        Returns:
            List of formatted response steps
        """
        if self.response_format == ResponseFormat.OPTION_LETTER:
            prompt = f"Question: {question}\n\nThe correct option letter is:"
        else:
            prompt = f"Question: {question}\n\nPlease box your final answer via $\\boxed{{your answer}}. The correct answer is: \\boxed{{"

        response = [prompt]
        
        if self.split_mode == SplitMode.SENTENCE:
            sentences = think_path.split('.')
            think_path_formatted = ""
            for step in sentences:
                think_path_formatted += step + "."
                response.append(f"Question: {question}\n\nThink: {think_path_formatted}\n\nThe correct option letter is:")
        
        elif self.split_mode == SplitMode.PARAGRAPH:
            paragraphs = think_path.split('\n\n')
            paragraphs_merged = []
            para = ""
            
            for step in paragraphs:
                if not para:
                    para = step.strip()
                else:
                    para = para + " " + step.strip()
                
                if len(para) >= self.merged_length or step == paragraphs[-1]:
                    paragraphs_merged.append(para)
                    para = ""

            think_path_formatted = ""
            for step in paragraphs_merged:
                think_path_formatted += step + " "
                if "</think>" in think_path_formatted:
                    response.append(f"Question: {question}\n\n<think>\n{think_path_formatted}\n\n{prompt}")
                else:
                    response.append(f"Question: {question}\n\n<think>\n{think_path_formatted}\n</think>\n\n{prompt}")
        
        return response

    @staticmethod
    def calculate_entropy(probabilities: List[float]) -> float:
        """
        Calculate entropy of a probability distribution.
        
        Args:
            probabilities: List of probabilities
            
        Returns:
            Calculated entropy value
        """
        sum_probs = sum(probabilities)
        if sum_probs == 0:
            return 0.0
            
        normalized = [p / sum_probs for p in probabilities if p > 0]
        return -sum(p * math.log2(p) for p in normalized)

    def analyze_sequence(self, context: str) -> Dict:
        """
        Analyze token sequence probabilities using beam search.
        
        Args:
            context: Initial context to analyze
            
        Returns:
            Dictionary containing top tokens and their probabilities
        """
        probabilities = {"target_token": [], "probability": []}
        beam = [("", context, 1.0)]
        completed_sequences = {}

        for _ in range(self.max_steps):
            new_beam = []
            for partial_str, current_context, cumulative_prob in beam:
                try:
                    token_probs = self.get_token_probabilities(current_context)
                    if not token_probs:
                        continue
                        
                    next_tokens = token_probs.get("top_k_tokens_with_probabilities", [])
                    
                    for token_info in next_tokens:
                        token = token_info["token"]
                        prob = token_info["probability"]
                        new_str = partial_str + token
                        new_ctx = current_context + token

                        if (self.response_format == ResponseFormat.MATH_BOX and "}" in token) or \
                           (self.response_format == ResponseFormat.OPTION_LETTER and token.strip() in [chr(i) for i in range(65, 91)]):
                            if partial_str not in completed_sequences:
                                completed_sequences[partial_str] = (partial_str, cumulative_prob * prob)
                        else:
                            new_beam.append((new_str, new_ctx, cumulative_prob * prob))
                except Exception as e:
                    print(f"Error in token probability analysis: {e}")
                    continue

            if not new_beam:
                break
                
            beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:self.top_k]

        top_candidates = sorted(completed_sequences.values(), key=lambda x: x[1], reverse=True)[:self.top_k]
        for seq, prob in top_candidates:
            probabilities["target_token"].append(seq)
            probabilities["probability"].append(prob)
            
        return probabilities

    def process_entry(self, entry: LogEntry) -> Tuple[str, int, bool]:
        """
        Process a single log entry.
        
        Args:
            entry: Log entry to process
            
        Returns:
            Tuple containing (model_response, token_count, is_correct)
        """
        formatted_steps = self.format_response(entry.question, entry.model_response)
        model_response = ""
        final_token = None

        for step in formatted_steps:
            if self.problem_type == ProblemType.MULTIPLE_CHOICE:
                options = [f" {chr(i)}" for i in range(65, 65 + (entry.option_num or 0))]
                probs = self.get_token_probabilities(step, target_token=options)
            else:
                probs = self.analyze_sequence(step)

            if not probs or not probs.get("probability"):
                continue

            avg_entropy = self.calculate_entropy(probs["probability"]) / len(probs["probability"])
            max_prob_idx = probs["probability"].index(max(probs["probability"]))
            model_response = step + probs['target_token'][max_prob_idx]
            final_token = probs['target_token'][max_prob_idx]

            if avg_entropy < self.threshold:
                break

        model_response = model_response.replace(f"Question: {entry.question}\n\n", "")
        token_count = self.get_token_count(model_response) or 0
        is_correct = final_token == entry.ground_truth if final_token else False
        
        return model_response, token_count, is_correct

    def dynamic_think_analysis(self, log_file_path: str, output_file_path: str) -> None:
        """
        Perform dynamic think analysis on log file and save results.
        
        Args:
            log_file_path: Path to input log file
            output_file_path: Path to save results
        """
        entries = self.parse_log_file(log_file_path)
        results = {"correct": [], "token_count": []}
        correct_indices = []
        incorrect_indices = []

        # Handle duplicate output files
        if os.path.exists(output_file_path):
            base_name = os.path.basename(output_file_path)
            name, ext = os.path.splitext(base_name)
            new_name = f"{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}{ext}"
            output_file_path = os.path.join(os.path.dirname(output_file_path), new_name)
            print(f"File exists. Renaming to {new_name}")

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for entry in tqdm(entries, desc="Processing entries"):
                response, token_count, is_correct = self.process_entry(entry)
                
                results["correct"].append(is_correct)
                results["token_count"].append(token_count)
                
                if is_correct:
                    correct_indices.append(entry.idx)
                else:
                    incorrect_indices.append(entry.idx)

                f.write(
                    f"idx:{entry.idx}/{entry.total}\t"
                    f"{is_correct}\t"
                    f"{response[-1] if response else 'N/A'}\t"
                    f"{entry.ground_truth}\t"
                    f"{token_count}\t"
                    f"Question: {[entry.question]}\t"
                    f"Model Response: {[response]}\n"
                )

            # Calculate and write summary statistics
            accuracy = np.mean(results["correct"]) if results["correct"] else 0.0
            token_avg = np.mean(results["token_count"]) if results["token_count"] else 0.0
            accuracy_std = np.std(results["correct"]) if results["correct"] else 0.0
            token_std = np.std(results["token_count"]) if results["token_count"] else 0.0

            f.write(f"\n{'='*40}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"Average Token Count: {token_avg:.1f}\n")
            f.write(f"Accuracy Std Dev: {accuracy_std:.4f}\n")
            f.write(f"Token Count Std Dev: {token_std:.1f}\n")
            f.write(f"Correct Indices: {correct_indices}\n")
            f.write(f"Incorrect Indices: {incorrect_indices}\n")
            f.write(f"{'='*40}\n")

if __name__ == "__main__":
    # Configuration for multiple choice problems
    mc_config = {
        "server_url": "http://localhost:3000",
        "top_k": 5,
        "max_steps": 10,
        "ratio": 0.1,
        "merged_length": 120
    }

    # Configuration for math problems
    math_config = {
        "server_url": "http://localhost:3000",
        "top_k": 5,
        "max_steps": 10,
        "ratio": 0.1,
        "merged_length": 200
    }

    # Example usage:
    analyzer = DynamicThinkAnalyzer(mc_config)
    log_path = "/path/to/log_file.log"
    output_path = "/path/to/output_results.log"
    analyzer.dynamic_think_analysis(log_path, output_path)