import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")   # 처음 한 번만 실행
nltk.download("omw-1.4")   # 다국어 WordNet 확장 (옵션)

word = "bank"
synsets = wn.synsets(word)

for s in synsets:
    print(s.name(), ":", s.definition())
