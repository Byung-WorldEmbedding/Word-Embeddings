import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr
import csv

# =============================
# 0. 준비
# =============================
MODEL_NAME = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
embedding_matrix = model.embeddings.word_embeddings.weight
dim = embedding_matrix.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")

# =============================
# 1. Base embedding 함수
# =============================
def get_base_embedding(word, tokenizer, embedding_matrix):
    tokens = tokenizer.tokenize(word)
    if not tokens:
        return None
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    sub_embs = embedding_matrix[token_ids, :]
    return sub_embs.mean(dim=0)  # (hidden_dim,)

# =============================
# 2. SimLex-999 로드
# =============================
def load_simlex999(path="SimLex-999.txt"):
    df = pd.read_csv(path, sep="\t")
    return df[["word1", "word2", "SimLex999"]]

simlex = load_simlex999("SimLex-999.txt")

# =============================
# 3. 단어쌍 유사도 (baseline: base embedding)
# =============================
results = []
for _, row in simlex.iterrows():
    w1, w2, gold = row["word1"], row["word2"], row["SimLex999"]

    emb1 = get_base_embedding(w1, tokenizer, embedding_matrix)
    emb2 = get_base_embedding(w2, tokenizer, embedding_matrix)

    if emb1 is None or emb2 is None:
        continue

    sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

    results.append({
        "word1": w1,
        "word2": w2,
        "gold_score": gold,
        "model_score": sim
    })

# =============================
# 4. Pearson correlation + CSV 저장
# =============================
if results:
    gold_scores = [r["gold_score"] for r in results]
    model_scores = [r["model_score"] for r in results]
    corr, _ = pearsonr(model_scores, gold_scores)
    print(f"[Baseline Evaluation] Pearson correlation on SimLex-999: {corr:.4f} (n={len(results)})")

    out_file = "simlex_baseline_results.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word1", "word2", "gold_score", "model_score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"[Saved] Baseline results saved to {out_file}")
else:
    print("⚠ 평가할 수 있는 단어 쌍이 없습니다.")
