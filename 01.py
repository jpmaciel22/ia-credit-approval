import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

credit_approval = fetch_ucirepo(id=27)

x_df = credit_approval.data.features.copy()

y_raw = credit_approval.data.targets.copy()

# print("\nVariaveis:")
# print(credit_approval.variables)

y = y_raw.iloc[:,0].map({
    "-": 0,
    "+": 1
})

dados_validos = y.notna()

x_df = x_df[dados_validos].copy()
y = y[dados_validos].astype(int)

# print("\nDistribuicao geral do Y:")
# print(y.value_counts())
# print(y.value_counts(normalize=True))

for coluna in x_df.columns:
    if pd.api.types.is_numeric_dtype(x_df[coluna]):
        x_df[coluna] = x_df[coluna].fillna(x_df[coluna].median())
    else:
        x_df[coluna] = x_df[coluna].fillna("Desconhecido")

x_df = pd.get_dummies(x_df, drop_first=True)

# print("\nFormato do X depois do get_dummies:")
# print(x_df.shape)

x_treinamento, x_test, y_treinamento, y_test = train_test_split(
    x_df,
    y,
    test_size=0.3,
    random_state=1,
    stratify=y
)

# print("\nDistribuicao no treino:")
# print(y_treinamento.value_counts())

# print("\nDistribuicao no teste:")
# print(y_test.value_counts())

modelo = RandomForestClassifier(
    random_state=1,
    n_estimators=100,
    max_depth=10,
    max_leaf_nodes=100,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=2
)

modelo.fit(x_treinamento, y_treinamento)
tree_index = 0
tree_to_visualize = modelo.estimators_[tree_index]

plt.figure(figsize=(30, 15))

plot_tree(
    tree_to_visualize,
    feature_names=x_df.columns.tolist(),
    class_names=["Denied", "Approved"],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3
)

plt.title("Decision Tree - Exemplo de uma arvore da Random Forest")
plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()

# print("\nImagem da arvore salva como: decision_tree.png")
# print("\nClasses do modelo:")
# print(modelo.classes_)

indice_approved = list(modelo.classes_).index(1)

prob_approved = modelo.predict_proba(x_test)[:, indice_approved]

print("\nResumo das probabilidades de Approved:")
print(pd.Series(prob_approved).describe())

print("\nProbabilidades apenas dos casos que realmente eram Approved:")
print(pd.Series(prob_approved[y_test.values == 1]).describe())

threshold_escolhido = 0.58
previsoes = (prob_approved >= threshold_escolhido).astype(int)
matriz = confusion_matrix(y_test, previsoes, labels=[0, 1])

tn, fp, fn, tp = matriz.ravel()

print("\nThreshold escolhido:")
print(threshold_escolhido)

print("\nMatriz de Confusao:")
print(matriz)

print("\nDados da Matriz Final:")
print(f"Denied certo: {tn}")
print(f"Denied marcado como Approved: {fp}")
print(f"Approved marcado como Denied: {fn}")
print(f"Approved certo: {tp}")
accuracy = accuracy_score(y_test, previsoes)

precision_approved = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_approved = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_approved = (
    2 * precision_approved * recall_approved / (precision_approved + recall_approved)
    if (precision_approved + recall_approved) > 0
    else 0
)

precision_weighted = precision_score(
    y_test,
    previsoes,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    y_test,
    previsoes,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    previsoes,
    average="weighted",
    zero_division=0
)

print("\nMetricas gerais:")
print(f"Acuracia: {accuracy:.4f}")
print(f"Precisao weighted: {precision_weighted:.4f}")
print(f"Recall weighted: {recall_weighted:.4f}")
print(f"F1 weighted: {f1_weighted:.4f}")

print("\nMetricas especificas para Approved:")
print(f"Precisao Approved: {precision_approved:.4f}")
print(f"Recall Approved: {recall_approved:.4f}")
print(f"F1 Approved: {f1_approved:.4f}")

print("\nRelatorio de classificacao:")
print(classification_report(
    y_test,
    previsoes,
    labels=[0, 1],
    target_names=["Denied", "Approved"],
    zero_division=0
))

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["Denied", "Approved"]
)

disp.plot(values_format="d")
plt.title(f"Matriz de Confusao - Threshold {threshold_escolhido}")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# print("\nImagem da matriz de confusao salva como: confusion_matrix.png")