import json
import os
from typing import List, Dict
from datasets import load_dataset
from modelscope.msdatasets import MsDataset

class CommonSenseQADataset:
    sys_prompt = 'You are a helpful AI assistant that will answer questions. You must end your response with "\\boxed{your answer}" everytime!'
    usr_prompt = '\n\nRemember to box your final answer via $\\boxed{your answer}$. Your answer needs to be the letter of the option. If there is no correct answer, give a random answer.'

    def __init__(self, data_split: str = "validation"):
        self.dataset = MsDataset.load("opencompass/commonsense_qa", split = data_split)
    
    def __len__(self):
        return len(self.dataset)
    
    def sample(self, n: int):
        self.dataset = self.dataset.select(range(n))

    def format_question(self, raw_question: Dict) -> str:
        formatted_question = dict()
        print(raw_question)
        question = raw_question["question"]
        options = raw_question["choices"]
        
        for label, option in zip(options['label'], options['text']):
            question += f"\n{label}. {option}"

        formatted_question["question"] = question
        formatted_question["answer"] = raw_question["answerKey"]
        formatted_question["message"] = [{'role': 'system', 'content': self.sys_prompt}, {'role': 'user', 'content': question + self.usr_prompt}]
        
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
                ans = resp[start + 7:end][0].strip().split("{")[-1]
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
