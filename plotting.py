import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 로드
df = pd.read_csv("results/simlex_baseline_results.csv")

plt.figure(figsize=(7,7))
sns.scatterplot(x="gold_score", y="model_score", data=df, alpha=0.6, edgecolor=None)

# 회귀선 추가
sns.regplot(x="gold_score", y="model_score", data=df,
            scatter=False, color="red", line_kws={"linewidth":2})

plt.xlabel("Gold (Human Similarity Score)")
plt.ylabel("Model (Cosine Similarity)")
plt.title("SimLex-999 Evaluation (Pearson r=0.3974)")
plt.grid(True)
plt.show()
plt.savefig("results/base_simlex_scatter.png", dpi=300)