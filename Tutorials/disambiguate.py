import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from nltk.corpus import wordnet as wn
import nltk

# --- 준비 ---
nltk.download('wordnet')
nltk.download('omw-1.4')

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.eval()

# 문장 임베딩 함수
def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    # [CLS] 토큰 (<s>) 임베딩 사용
    return outputs.last_hidden_state[0, 0, :]

# 단어 sense 정의 임베딩 추출
def get_sense_embeddings(word):
    synsets = wn.synsets(word)
    sense_embeddings = {}
    for syn in synsets:
        definition = syn.definition()
        emb = get_sentence_embedding(definition)
        sense_embeddings[syn] = emb
    return sense_embeddings

# 문맥 기반 sense 판별
def disambiguate(word, context_sentence):
    word_emb = get_sentence_embedding(context_sentence)
    sense_embs = get_sense_embeddings(word)

    scores = []
    for syn, emb in sense_embs.items():
        sim = F.cosine_similarity(word_emb.unsqueeze(0), emb.unsqueeze(0)).item()
        scores.append((syn, sim))

    # 가장 높은 similarity sense 선택
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores

# --- 테스트 ---
context1 = "He went to the bank to deposit his paycheck."
context2 = "The children played near the bank of the river."

for ctx in [context1, context2]:
    results = disambiguate("bank", ctx)
    print(f"\nContext: {ctx}")
    for syn, score in results[:3]:  # 상위 3개 sense 출력
        print(f"  {syn.name()} ({syn.definition()}): sim={score:.4f}")
