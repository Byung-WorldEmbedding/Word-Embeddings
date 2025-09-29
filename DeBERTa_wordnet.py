import torch
from nltk.corpus import wordnet as wn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 1. Base embedding 추출
# =============================
def get_base_embedding(word, tokenizer, embedding_matrix):
    tokens = tokenizer.tokenize(word)
    if not tokens:
        return None
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    sub_embs = embedding_matrix[token_ids, :]
    return sub_embs.mean(dim=0)  # (hidden_dim,)

# =============================
# 2. Sense inventory 구축
# =============================
def build_sense_inventory(word, tokenizer, embedding_matrix):
    synsets = wn.synsets(word)
    if not synsets:
        return None

    base_emb = get_base_embedding(word, tokenizer, embedding_matrix)
    if base_emb is None:
        return None

    senses = []
    for s in synsets:
        senses.append({
            "sense_id": s.name(),
            "pos": s.pos(),
            "gloss": s.definition(),
        })

    return {
        "base_embedding": base_emb,
        "senses": senses
    }

# =============================
# 3. Sense embedding 초기화 (noise 추가)
# =============================
def init_sense_embeddings(inventory, dim):
    sense_embs = {}
    for s in inventory["senses"]:
        sid = s["sense_id"]
        base_emb = inventory["base_embedding"].to(device)
        sense_embs[sid] = base_emb + 0.01 * torch.randn(dim, device=device)
    return sense_embs

# =============================
# 4. 정의문 기반 업데이트 (식 1 단순화)
# =============================
def definition_update(sense_id, sense_embs, tokenizer, embedding_matrix, beta=0.3):
    synset = wn.synset(sense_id)
    gloss = synset.definition()
    gloss_tokens = tokenizer.tokenize(gloss)
    if not gloss_tokens:
        return sense_embs[sense_id]
    gloss_ids = tokenizer.convert_tokens_to_ids(gloss_tokens)
    gloss_emb = embedding_matrix[gloss_ids, :].mean(dim=0).to(device)

    v_old = sense_embs[sense_id]
    r_j = (1 - beta) * v_old + beta * gloss_emb
    return r_j

# =============================
# 5. Base alignment (식 4)
# =============================
def base_alignment(r_js, base_emb, gamma=0.5):
    r_stack = torch.stack(r_js)
    r_bar = r_stack.mean(dim=0)
    v_js = []
    for r_j in r_js:
        v_j = r_j + (1 - gamma) * (base_emb - r_bar)
        v_js.append(v_j)
    return v_js

# =============================
# 6. 학습 루프 (1 epoch 예시)
# =============================
def train_one_epoch(inventory, sense_embs, tokenizer, embedding_matrix, beta=0.3, gamma=0.5):
    base_emb = inventory["base_embedding"].to(device)
    senses = inventory["senses"]

    # 정의문 기반 업데이트
    r_js = []
    for s in senses:
        sid = s["sense_id"]
        r_j = definition_update(sid, sense_embs, tokenizer, embedding_matrix, beta)
        r_js.append(r_j)

    if not r_js:
        return sense_embs

    # base alignment
    v_js = base_alignment(r_js, base_emb, gamma)

    # sense embedding 갱신
    for s, v_j in zip(senses, v_js):
        sense_embs[s["sense_id"]] = v_j

    return sense_embs
