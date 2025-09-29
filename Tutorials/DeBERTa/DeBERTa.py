import torch
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import wordnet as wn
import nltk

# =============================
# 1. 준비
# =============================
nltk.download("wordnet")
nltk.download("omw-1.4")

MODEL_NAME = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

embedding_matrix = model.embeddings.word_embeddings.weight  # (vocab_size, hidden_dim)

# =============================
# 2. 단어 → base embedding
# =============================
def get_base_embedding(word, tokenizer, embedding_matrix):
    """
    주어진 단어를 subword로 분해하고, subword embedding들을 평균내어 base embedding 생성
    """
    tokens = tokenizer.tokenize(word)
    if not tokens:
        return None
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    sub_embs = embedding_matrix[token_ids, :]  # (num_subwords, hidden_dim)
    return sub_embs.mean(dim=0)  # (hidden_dim,)

# =============================
# 3. sense inventory 추출
# =============================
def build_sense_inventory(word_list, tokenizer, embedding_matrix, max_words=100):
    """
    word_list의 단어들에 대해 sense inventory와 gloss를 추출하고
    base embedding과 매핑
    """
    inventory = {}

    for w in word_list[:max_words]:  # 샘플링: max_words 개만
        synsets = wn.synsets(w)
        if not synsets:
            continue

        base_emb = get_base_embedding(w, tokenizer, embedding_matrix)
        if base_emb is None:
            continue

        senses = []
        for s in synsets:
            gloss = s.definition()
            senses.append({
                "sense_id": s.name(),
                "pos": s.pos(),
                "gloss": gloss,
            })

        inventory[w] = {
            "base_embedding": base_emb,
            "senses": senses
        }

    return inventory

# =============================
# 4. 실행 예시
# =============================
# WordNet에서 단어 목록 가져오기
import random
words = list(wn.words())
sample_words = random.sample(words, 5)  # 무작위 5개 선택
inventory = build_sense_inventory(sample_words, tokenizer, embedding_matrix)

# 작은 샘플로 테스트
inventory = build_sense_inventory(sample_words, tokenizer, embedding_matrix, max_words=10)

# 확인
for w, info in inventory.items():
    print(f"\nWord: {w}")
    print("Base Embedding Shape:", info["base_embedding"].shape)
    for s in info["senses"]:
        print(f"  - {s['sense_id']} ({s['pos']}): {s['gloss']}")
