import torch
import math
from transformers import RobertaTokenizer, RobertaForMaskedLM

# 모델 로드
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForMaskedLM.from_pretrained("roberta-base")
model.eval()

sentence = "It is unbelievableness because I saw it for the first time."
inputs = tokenizer(sentence, return_tensors="pt")
input_ids = inputs["input_ids"][0]

log_likelihood = 0.0
count = 0
token_losses = []

# 토큰화된 토큰 ID와 실제 문자열 확인
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Original Sentence:")
print(sentence, "\n")

print("Tokenized (BPE tokens):")
print(tokens, "\n")

print("Token IDs:")
print(token_ids, "\n")

# Special tokens 포함 (모델 입력용)
encoding = tokenizer(sentence, return_tensors="pt")
print("Input IDs (with <s>, </s>):")
print(encoding["input_ids"])
print("Decoded back:", tokenizer.decode(encoding["input_ids"][0]))

# 각 토큰별 [MASK] 처리 후 확률 추적
for i in range(1, len(input_ids) - 1):  # <s>, </s> 제외
    masked = input_ids.clone()
    masked[i] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(masked.unsqueeze(0))
        logits = outputs.logits

    probs = torch.softmax(logits[0, i], dim=-1)
    true_id = input_ids[i].item()
    token_prob = probs[true_id].item()

    # 토큰 단위 loss & ppl
    token_loss = -math.log(token_prob + 1e-12)
    token_ppl = math.exp(token_loss)

    token_str = tokenizer.decode([true_id])
    print(f"Token: {token_str:10s} | Loss: {token_loss:.4f} | PPL: {token_ppl:.2f}")

    log_likelihood += math.log(token_prob + 1e-12)
    token_losses.append(token_loss)
    count += 1

# 문장 단위 Pseudo-PPL
pseudo_ppl = math.exp(-log_likelihood / count)
avg_loss = sum(token_losses) / len(token_losses)

print(f"\nAverage Token Loss: {avg_loss:.4f}")
print(f"Sentence Pseudo-Perplexity: {pseudo_ppl:.2f}")
