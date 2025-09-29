from DeBERTa_wordnet import get_base_embedding, build_sense_inventory, init_sense_embeddings, definition_update, train_one_epoch
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from transformers import AutoTokenizer, AutoModel
import nltk

# =============================
# 0. 준비
# =============================
nltk.download("wordnet")
nltk.download("omw-1.4")

MODEL_NAME = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

embedding_matrix = model.embeddings.word_embeddings.weight  # (vocab_size, hidden_dim)
print(f"embedding_matrix: {embedding_matrix.shape}")
dim = embedding_matrix.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")

# =============================

word = "red-hot"
inventory = build_sense_inventory(word, tokenizer, embedding_matrix)

if inventory:
    sense_embs = init_sense_embeddings(inventory, dim)
    print(f"\n[Init] {word} sense embeddings initialized")

    # 1 epoch 학습
    sense_embs = train_one_epoch(inventory, sense_embs, tokenizer, embedding_matrix)

    print(f"\n[After 1 Epoch] Sense embeddings for '{word}':")
    for s in inventory["senses"]:
        sid = s["sense_id"]
        gloss = s["gloss"]
        vec = sense_embs[sid]
        print(f" - {sid}: {gloss[:60]}... | shape={vec.shape}")


