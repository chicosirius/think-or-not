import argparse
import json
from fastapi import FastAPI, Request
import torch
import uvicorn
import datetime
import os

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["TRITON_MMA"] = "0"

app = FastAPI()

# Automatically select GPUs based on available devices
gpu_count = torch.cuda.device_count()
if gpu_count > 0:
    # Set CUDA_VISIBLE_DEVICES to use all available GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
else:
    print("No GPU found. Defaulting to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.post("/v1/chat/completions")
async def create_item(request: Request):
    global model, tokenizer
    try:
        json_post_raw = await request.json()
        max_length = json_post_raw.get('max_tokens')
        top_p = json_post_raw.get('top_p')
        temperature = json_post_raw.get('temperature')
        messages = json_post_raw.get('messages')
        repetition_penalty = json_post_raw.get('repetition_penalty')
        think_mode = json_post_raw.get('think_mode', False)

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, max_tokens=max_length)

        if args.model == 'Qwen/QwQ-32B':
            if think_mode:
                tokenizer.chat_template = chat_template["QwQ_think_chat_template"]
            else:
                tokenizer.chat_template = chat_template["QwQ_nothink_chat_template"]
        if args.model == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B':
            if think_mode:
                tokenizer.chat_template = chat_template["Deepseek-r1-32b_think_chat_template"]
            else:
                tokenizer.chat_template = chat_template["Deepseek-r1-32b_nothink_chat_template"]

        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print(f"inputs: {inputs}")
        outputs = model.generate(inputs, sampling_params)
        token_count = len(outputs[0].outputs[0].token_ids)
        response = outputs[0].outputs[0].text

        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "think_mode": think_mode,
                    "content": response,
                    "token_count": token_count
                }
            }],
        }
        log = f"[{time}] think mode: {think_mode}, prompt: {messages}, response: {repr(response)}"
        print(log)
        return answer["choices"][0]["message"]

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return {"error": error_message}

def parse_args():
    parser = argparse.ArgumentParser(description="Run FastAPI server with custom port and model")
    parser.add_argument('--model', type=str, required=True, help="Model to load (e.g., 'microsoft/Phi-3-mini-4k-instruct')")
    parser.add_argument('--port', type=int, default=4000, help="Port to run the server on")
    return parser.parse_args()

if __name__ == '__main__':
    # Command line arguments
    args = parse_args()

    # Model and tokenizer loading based on provided model name
    model_dir = args.model

    with open("chat_template.json", "r", encoding="utf-8") as f:
        chat_template = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    model = LLM(model = args.model, 
                tensor_parallel_size=torch.cuda.device_count(), 
                trust_remote_code=True
                )

    # Running the server with the specified port
    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)