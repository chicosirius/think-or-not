import argparse
import json
from fastapi import FastAPI, Request
import torch
import uvicorn
import datetime
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import softmax

os.environ["TRITON_MMA"] = "0"  # Disable Triton MMA

app = FastAPI()

# Automatically select GPUs based on available devices
gpu_count = torch.cuda.device_count()
if gpu_count > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    print(f"Using {gpu_count} GPU(s).")
else:
    print("No GPU found. Defaulting to CPU.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Global variables for model and tokenizer
model = None
tokenizer = None

@app.post("/v1/token/probabilities")
async def get_token_probabilities(request: Request):
    global model, tokenizer
    try:
        json_post_raw = await request.json()
        input_text = json_post_raw.get('input_text')
        target_token = json_post_raw.get('target_token', None)  # Optional: specific token or sequence
        top_k = json_post_raw.get('top_k', 10)  # Default to top-k tokens

        # Tokenize the input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        # Generate logits using the model
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Get the logits for the last token in the sequence
        last_token_logits = logits[0, -1, :]

        # Calculate probabilities using softmax
        probabilities = softmax(last_token_logits, dim=-1)

        if target_token and type(target_token) == str:
            # Tokenize the target token(s)
            target_token_ids = tokenizer.encode(target_token, add_special_tokens=False)
            
            # Calculate the conditional probability for the target token sequence
            conditional_probability = 1.0
            for token_id in target_token_ids:
                conditional_probability *= probabilities[token_id].item()
                # Update probabilities for the next token
                input_ids = torch.cat([input_ids, torch.tensor([[token_id]], device=model.device)], dim=1)
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits
                    last_token_logits = logits[0, -1, :]
                    probabilities = softmax(last_token_logits, dim=-1)

            return {
                "target_token": target_token,
                "probability": conditional_probability
            }
        elif target_token and type(target_token) == list:
            # Tokenize the target token(s)
            target_token_ids_list = []
            for token in target_token:
                target_token_ids = tokenizer.encode(token, add_special_tokens=False)
                target_token_ids_list.extend(target_token_ids)
            
            probability_list = []
            for token_id in target_token_ids_list:
                probability_list.append(probabilities[token_id].item())

            return {
                "target_token": target_token,
                "probability": probability_list
            }
        else:
            # Return the top-k token probabilities
            top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k)
            top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_indices.tolist())

            return {
                "top_k_tokens_with_probabilities": [
                    {"token": token, "probability": prob} 
                    for token, prob in zip(top_k_tokens, top_k_probs.tolist())
                ]
            }

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return {"error": error_message}

@app.post("/v1/token/count")
async def count_tokens(request: Request):
    global tokenizer
    try:
        json_post_raw = await request.json()
        text = json_post_raw.get('text', '')
        if not text:
            return {"error": "No text provided"}

        tokens = tokenizer.encode(text)
        return {"token_count": len(tokens)}
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return {"error": error_message}

def parse_args():
    parser = argparse.ArgumentParser(description="Run FastAPI server with custom port and model")
    parser.add_argument('--model', type=str, required=True, help="Model to load (e.g., 'Qwen/QwQ-32B')")
    parser.add_argument('--port', type=int, default=4000, help="Port to run the server on")
    return parser.parse_args()

if __name__ == '__main__':
    # Command line arguments
    args = parse_args()

    # Load tokenizer and model
    model_dir = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")

    # Running the server with the specified port
    uvicorn.run(app, host='0.0.0.0', port=args.port, workers=1)