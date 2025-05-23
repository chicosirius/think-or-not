import json
import os
from typing import List, Dict
from datasets import load_dataset
from datasets import concatenate_datasets

class AIME2025Dataset:
    sys_prompt = 'You are a helpful AI assistant that will answer questions. You must end your response with "\\boxed{your answer}" everytime!'
    usr_prompt = '\n\nRemember to box your final answer via $\\boxed{your answer}$.'

    def __init__(self, data_split: str = "test"):
        dataset1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split=data_split)
        dataset2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split=data_split)
        self.dataset = concatenate_datasets([dataset1, dataset2])

    def __len__(self):
        return len(self.dataset)
    
    def sample(self, n: int):
        self.dataset = self.dataset.select(range(n))
    
    def format_question(self, raw_question: Dict) -> str:
        formatted_question = dict()
        question = raw_question["question"]

        formatted_question["question"] = question
        formatted_question["answer"] = raw_question["answer"]
        formatted_question["message"] = [{'role': 'system', 'content': self.sys_prompt}, {'role': 'user', 'content': question + self.usr_prompt}]
        
        return formatted_question
    
    def __iter__(self):
        for raw_question in self.dataset:
            yield self.format_question(raw_question)
    
    def evaluate_response(self, 
                          model_responses, 
                          question: Dict
                          ) -> List[Dict]:
        answer = question["answer"].split("^")[0]
        returned_answers = []

        for resp in model_responses:
            # Check if the model response contains the answer
            correct = False
            ans = None
            slice = None

            try:
                start = resp.rindex("\\boxed{")
                end = resp.index("}", start)
                ans = resp[start + 7:end].strip().split("{")[-1].split("^")[0]
                slice = (start + 7, end)
                correct = ans == answer
            except ValueError:
                raise ValueError("Response does not contain a boxed answer.")

            returned_answers.append({
                'model_response': resp,
                'answer_line': ans,
                'correct': correct,
                'answer_randomly_sampled': False,
                'answer_span': slice,
                'model_answer': ans,
                'raw_model_answer': ans,
                **question
            })

        return returned_answers
