import json
import os
from typing import List, Dict
from datasets import load_dataset
from modelscope.msdatasets import MsDataset
from tqdm import tqdm

class GPQADataset:
    sys_prompt = 'You are a helpful AI assistant that will answer questions. You must end your response with "\\boxed{your answer}" everytime!'
    usr_prompt = '\n\nRemember to box your final answer via $\\boxed{your answer}$.'

    def __init__(self, data_split: List[str] = ["gpqa_main"]):
        self.dataset = []
        for split in data_split:
            self.dataset.extend(MsDataset.load("modelscope/gpqa", split=split, trust_remote_code=True))
    
    def __len__(self):
        return len(self.dataset)

    def format_question(self, raw_question: Dict) -> Dict:
        formatted_question = dict()
        narrative = raw_question["narrative"]
        question = raw_question["question"]
        options = raw_question["choices"].strip('[').strip(']').split(',')

        label = [chr(i) for i in range(65, 65 + len(options))]
        for l, option in zip(label, options):
            question += f"\n{l}. {option}"

        formatted_question["question"] = narrative + "\n\n" + question
        formatted_question["answer"] = chr(65 + raw_question["answer_index"])
        formatted_question["message"] = [{'role': 'system', 'content': self.sys_prompt}, {'role': 'user', 'content': narrative + "\n\n" + question + self.usr_prompt}]
        
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
                start = resp.index("\\boxed{")
                end = resp.index("}", start)
                ans = resp[start + 7:end][0].strip()
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
