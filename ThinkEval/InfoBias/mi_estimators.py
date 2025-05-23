import os
import ast
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm

# Functions for distance and kernel computations

def distmat(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise squared distance matrix.
    """
    if X.dim() == 1:
        X = X.view(-1, 1)
    r = torch.sum(X * X, dim=1).view(-1, 1)
    a = torch.mm(X, X.t())
    D = r.expand_as(a) - 2 * a + r.t().expand_as(a)
    return D.abs()


def sigma_estimation(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Estimate sigma using the median of pairwise distances.
    """
    D = distmat(torch.cat([X, Y], dim=0)).cpu().numpy()
    tril = np.tril_indices(D.shape[0], -1)
    distances = D[tril]
    median = np.median(distances)
    if median <= 0:
        median = np.mean(distances)
    return max(median, 1e-2)


def kernelmat(X: torch.Tensor, sigma: float, ktype: str = 'gaussian') -> torch.Tensor:
    """
    Build centered kernel matrix for input tensor X.
    """
    m = X.size(0)
    H = torch.eye(m) - (1.0 / m) * torch.ones((m, m))

    if ktype == 'gaussian':
        D = distmat(X)
        variance = 2.0 * sigma * sigma * X.size(1)
        K = torch.exp(-D / variance)
    elif ktype == 'linear':
        K = torch.mm(X, X.t())
    elif ktype == 'IMQ':
        D = distmat(X)
        K = (D + 1).rsqrt()
    else:
        raise ValueError(f"Unknown kernel type: {ktype}")

    # Center the kernel matrix
    return torch.mm(K, H)


def hsic_normalized_cca(x: np.ndarray, y: np.ndarray, sigma: float = 50.0, ktype: str = 'gaussian') -> float:
    """
    Compute HSIC-based dependence estimate using normalized CCA formulation.
    """
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    m = x_t.size(0)

    Kx_c = kernelmat(x_t, sigma, ktype)
    Ky_c = kernelmat(y_t, sigma, ktype)

    epsilon = 1e-5
    I = torch.eye(m)
    inv_x = torch.inverse(Kx_c + epsilon * m * I)
    inv_y = torch.inverse(Ky_c + epsilon * m * I)

    Rx = Kx_c.mm(inv_x)
    Ry = Ky_c.mm(inv_y)
    score = torch.sum(Rx * Ry.t())
    return score.item()


def hsic_gaussian(x: np.ndarray, y: np.ndarray, sigma: float = 5.0, ktype: str = 'gaussian') -> float:
    """
    Shortcut to compute HSIC with Gaussian kernel.
    """
    return hsic_normalized_cca(x, y, sigma, ktype)


def extract_model_responses(log_file_path: str):
    """
    Parse model responses and token counts from log file.

    Returns:
        grouped_responses (list[list[str]]): Model outputs grouped by 5.
        avg_tokens (list[float]): Average token count per group.
    """
    responses = []
    tokens = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Model Response:' in line:
                resp = line.split('Model Response:', 1)[1].strip().strip('[]"')
                responses.append(resp)
                token = int(line.split('\t')[4])
                tokens.append(token)

    grouped = [responses[i:i+5] for i in range(0, len(responses), 5)]
    token_groups = [tokens[i:i+5] for i in range(0, len(tokens), 5)]
    avg_tokens = [sum(g)/len(g) for g in token_groups]
    return grouped, avg_tokens


def extract_rewrite_responses(log_file_path: str):
    """
    Parse rewritten answers from log file.

    Returns:
        grouped_rewrites (list[list[str]]): Rewrites grouped by 5.
    """
    rewrites = []
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Generated Answers:' in line:
                answers = ast.literal_eval(line.split('Generated Answers:', 1)[1].strip())
                rewrites.append(answers)
    return rewrites

# Paths (using current working directory for logs)
base_dir = os.getcwd()
model_log = os.path.join(base_dir, 'gsm8k_phi-4_True.log')
rewrite_log = os.path.join(base_dir, 'rewrite_gsm8k_10.log')

# Extract data
model_outputs, token_counts = extract_model_responses(model_log)
rewrite_outputs = extract_rewrite_responses(rewrite_log)

# Initialize embedding model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')

# Prepare results
results = []
for i in tqdm(range(len(model_outputs)), desc='Computing HSIC'):
    emb_model = embedder.encode(model_outputs[i])
    emb_rewrite = embedder.encode(rewrite_outputs[i])

    hsic_val = hsic_gaussian(emb_model, emb_rewrite, sigma=5.0)
    hsic_per_tok = hsic_val / token_counts[i]

    results.append({
        'Index': i+1,
        'AvgToken': token_counts[i],
        'HSIC': hsic_val,
        'HSICPerToken': hsic_per_tok
    })

# Normalize HSIC per token and compute InfoBias
per_tok_vals = [r['HSICPerToken'] for r in results]
min_val, max_val = min(per_tok_vals), max(per_tok_vals)
for r in results:
    if max_val > min_val:
        norm = (r['HSICPerToken'] - min_val) / (max_val - min_val)
    else:
        norm = 0.0
    r['NormHSICPerToken'] = norm
    r['InfoBias'] = 1 - norm

# Save to Excel
df = pd.DataFrame(results)
output_file = os.path.join(os.getcwd(), 'hsic_gsm8k_phi4.xlsx')
df.to_excel(output_file, index=False)
