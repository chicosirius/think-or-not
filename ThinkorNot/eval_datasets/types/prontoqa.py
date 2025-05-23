import json
import os
from typing import List, Dict
from datasets import load_dataset

class ProntoQADataset:
    sys_prompt = 'You are a helpful AI assistant that will answer questions. You must end your response with "\\boxed{your answer}" everytime!'
    usr_prompt = '\n\nRemember to box your final answer via $\\boxed{your answer}$. If there is no correct answer, give a random answer.'

    def __init__(self, data_split: str = "validation"):
        self.dataset = load_dataset("renma/ProntoQA", split=data_split)
    
    def __len__(self):
        return len(self.dataset)
    
    def sample(self, n: int):
        self.dataset = self.dataset.select(range(n))

    def format_question(self, raw_question: Dict) -> str:
        formatted_question = dict()
        context = raw_question["context"]
        question = raw_question["question"]
        options = raw_question["options"]

        label = [chr(i) for i in range(65, 65 + len(options))]
        for l, option in zip(label, options):
            question += f"\n{option}"

        formatted_question["question"] = context + "\n" + question
        formatted_question["answer"] = raw_question["answer"]
        formatted_question["message"] = [{'role': 'system', 'content': self.sys_prompt}, {'role': 'user', 'content': context + "\n" + question + self.usr_prompt}]
        
        return formatted_question
    
    def __iter__(self):
        for raw_question in self.dataset:
            yield self.format_question(raw_question)
    
    def evaluate_response(self, 
                          model_responses, 
                          question: Dict
                          ) -> List[Dict]:
        answer = question["answer"]
        returned_answers = []

        for resp in model_responses:
            # Check if the model response contains the answer
            correct = False
            ans = None
            slice = None

            try:
                start = resp.rindex("\\boxed{")
                end = resp.index("}", start)
                ans = resp[start + 7:end].strip()
                slice = (start + 7, end)
                ans = ans.split("{")[-1].split(")")[0]
                if "True" in ans or "true" in ans:
                    ans = "A"
                if "False" in ans or "false" in ans:
                    ans = "B"
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
