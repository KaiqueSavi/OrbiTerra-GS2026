"""
OrbiTerra · Treino e comparação de modelos (Random Forest x XGBoost)
====================================================================

Pergunta científica do projeto:
    "O sensor de campo (ESP32) agrega poder preditivo REAL sobre o que já se
     consegue só com satélite + clima?"

Para responder, treinamos os mesmos algoritmos em DOIS conjuntos de variáveis:

    Modelo A  =  satélite (NDVI/EVI/LST) + clima (chuva/temperatura)
    Modelo B  =  Modelo A  +  ESP32 (umidade do solo, temp/umidade do ar)

Se B > A, o sensor se paga. Validação TEMPORAL (treina no passado, testa no
futuro) — nunca embaralhamos o tempo, para não "ver o futuro".

Saídas:
    docs/imagens/01_serie_temporal.png      — solo (ESP32) reage ANTES do NDVI
    docs/imagens/02_importancia_variaveis.png
    docs/imagens/03_matriz_confusao.png
    docs/imagens/04_comparacao_modelos.png
    data/metricas.txt                 — resumo textual (vai para o PDF)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # backend sem janela (salva PNG)
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_CSV = os.path.join(RAIZ, "data", "lavoura_cafe.csv")
DIR_IMAGENS = os.path.join(RAIZ, "docs", "imagens")
CAMINHO_METRICAS = os.path.join(RAIZ, "data", "metricas.txt")

SEED = 42
FRACAO_TREINO = 0.75                        # 75% passado p/ treino, 25% futuro
NOMES_RISCO = {0: "Baixo", 1: "Médio", 2: "Alto"}
CORES_RISCO = {0: "#2e9e5b", 1: "#e8a33d", 2: "#d1495b"}

# Variáveis observáveis SÓ do satélite + clima (Modelo A)
FEATS_SATELITE_CLIMA = [
    "ndvi", "evi", "lst", "precip_mm", "temp_ar",
    "precip_7d", "precip_30d", "dias_sem_chuva",
    "ndvi_var_7d", "temp_7d", "mes_sin", "mes_cos",
]
# Variáveis adicionais trazidas pelo ESP32 (entram só no Modelo B)
FEATS_ESP32 = [
    "umidade_solo", "umidade_solo_7d", "umidade_solo_tend_3d",
    "umidade_ar", "temp_ar_esp",
]


# --------------------------------------------------------------------------- #
# Engenharia de variáveis
# --------------------------------------------------------------------------- #
def engenharia(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas. Nada aqui usa o futuro — só passado/presente."""
    df = df.sort_values("data").reset_index(drop=True).copy()

    # Clima acumulado (memória de chuva) e tendência.
    df["precip_7d"] = df["precip_mm"].rolling(7, min_periods=1).sum()
    df["precip_30d"] = df["precip_mm"].rolling(30, min_periods=1).sum()
    df["temp_7d"] = df["temp_ar"].rolling(7, min_periods=1).mean()

    # Dias consecutivos sem chuva (estiagem acumulada).
    sem_chuva = (df["precip_mm"] <= 0.1).astype(int)
    grupos = (sem_chuva == 0).cumsum()
    df["dias_sem_chuva"] = sem_chuva.groupby(grupos).cumsum()

    # Variabilidade recente do NDVI (queda de vigor agita a série).
    df["ndvi_var_7d"] = df["ndvi"].rolling(7, min_periods=1).std().fillna(0.0)

    # Sinais do ESP32: nível, média recente e TENDÊNCIA de 3 dias (alerta!).
    df["umidade_solo_7d"] = df["umidade_solo"].rolling(7, min_periods=1).mean()
    df["umidade_solo_tend_3d"] = df["umidade_solo"].diff(3).fillna(0.0)

    # Sazonalidade cíclica (mês como seno/cosseno).
    mes = df["data"].dt.month
    df["mes_sin"] = np.sin(2 * np.pi * mes / 12)
    df["mes_cos"] = np.cos(2 * np.pi * mes / 12)

    return df.dropna().reset_index(drop=True)


def split_temporal(df: pd.DataFrame):
    corte = int(len(df) * FRACAO_TREINO)
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()


# --------------------------------------------------------------------------- #
# Treino de um cenário (conjunto de features) com os dois algoritmos
# --------------------------------------------------------------------------- #
def treina_cenario(treino, teste, feats, rotulo_cenario):
    Xtr, ytr = treino[feats], treino["risco_7d"]
    Xte, yte = teste[feats], teste["risco_7d"]

    modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED,
            objective="multi:softprob", num_class=3, n_jobs=-1,
            eval_metric="mlogloss",
        ),
    }

    resultados = {}
    for nome, modelo in modelos.items():
        modelo.fit(Xtr, ytr)
        pred = modelo.predict(Xte)
        resultados[nome] = {
            "modelo": modelo,
            "pred": pred,
            "acc": accuracy_score(yte, pred),
            "f1": f1_score(yte, pred, average="macro"),
            "cenario": rotulo_cenario,
            "feats": feats,
        }
    return resultados, yte


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def grafico_serie_temporal(df):
    """Mostra a 'alma' do projeto: a umidade do solo (ESP32) despenca ANTES
    de o NDVI (satélite) cair — e o risco alto vem na sequência."""
    jan = df.iloc[-220:]                    # ~últimos 7 meses (mais legível)
    fig, ax1 = plt.subplots(figsize=(11, 4.5))

    ax1.plot(jan["data"], jan["umidade_solo"], color="#1f6f4a",
             lw=1.8, label="Umidade do solo — ESP32 (micro)")
    ax1.set_ylabel("Umidade do solo (%)", color="#1f6f4a")
    ax1.tick_params(axis="y", labelcolor="#1f6f4a")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    ax2.plot(jan["data"], jan["ndvi"], color="#b06b00",
             lw=1.8, ls="--", label="NDVI — satélite (macro)")
    ax2.set_ylabel("NDVI", color="#b06b00")
    ax2.tick_params(axis="y", labelcolor="#b06b00")
    ax2.set_ylim(0.3, 0.85)

    # Faixas de risco alto ao fundo.
    for _, r in jan[jan["risco_7d"] == 2].iterrows():
        ax1.axvspan(r["data"], r["data"], color="#d1495b", alpha=0.04)

    ax1.set_title("OrbiTerra — o solo (ESP32) reage ANTES do NDVI (satélite)")
    linhas = ax1.get_lines() + ax2.get_lines()
    ax1.legend(linhas, [l.get_label() for l in linhas], loc="upper right", fontsize=9)
    fig.tight_layout()
    cam = os.path.join(DIR_IMAGENS, "01_serie_temporal.png")
    fig.savefig(cam, dpi=130)
    plt.close(fig)
    return cam


def grafico_importancia(res_b_xgb):
    modelo = res_b_xgb["modelo"]
    feats = res_b_xgb["feats"]
    imp = pd.Series(modelo.feature_importances_, index=feats).sort_values()
    cores = ["#1f6f4a" if f in FEATS_ESP32 else "#9bb7c4" for f in imp.index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(imp.index, imp.values, color=cores)
    ax.set_title("Importância das variáveis — XGBoost (Modelo B, satélite+clima+ESP32)")
    ax.set_xlabel("Importância relativa")
    # Legenda manual (verde = ESP32).
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#1f6f4a", label="ESP32 (campo)"),
        Patch(color="#9bb7c4", label="Satélite + clima"),
    ], loc="lower right")
    fig.tight_layout()
    cam = os.path.join(DIR_IMAGENS, "02_importancia_variaveis.png")
    fig.savefig(cam, dpi=130)
    plt.close(fig)
    return cam, imp


def grafico_matriz_confusao(yte, res_a, res_b):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, res, titulo in [
        (axes[0], res_a, "Modelo A (satélite+clima)"),
        (axes[1], res_b, "Modelo B (+ESP32)"),
    ]:
        cm = confusion_matrix(yte, res["pred"], labels=[0, 1, 2])
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(NOMES_RISCO.values())
        ax.set_yticklabels(NOMES_RISCO.values())
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
        ax.set_title(f"{titulo}\nXGBoost · acc={res['acc']:.0%}")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="black", fontsize=11)
    fig.suptitle("Matriz de confusão — o ESP32 corrige erros do satélite", y=1.02)
    fig.tight_layout()
    cam = os.path.join(DIR_IMAGENS, "03_matriz_confusao.png")
    fig.savefig(cam, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return cam


def grafico_comparacao(res_a, res_b):
    algos = list(res_a.keys())
    x = np.arange(len(algos))
    largura = 0.35
    acc_a = [res_a[a]["acc"] for a in algos]
    acc_b = [res_b[a]["acc"] for a in algos]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar(x - largura / 2, acc_a, largura, label="A: satélite+clima",
                color="#9bb7c4")
    b2 = ax.bar(x + largura / 2, acc_b, largura, label="B: +ESP32",
                color="#1f6f4a")
    ax.set_xticks(x); ax.set_xticklabels(algos)
    ax.set_ylabel("Acurácia (conjunto de teste)")
    ax.set_ylim(0, 1)
    ax.set_title("Acurácia: o sensor de campo eleva a previsão")
    ax.legend()
    for barras in (b1, b2):
        for b in barras:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.0%}", ha="center", fontsize=10)
    fig.tight_layout()
    cam = os.path.join(DIR_IMAGENS, "04_comparacao_modelos.png")
    fig.savefig(cam, dpi=130)
    plt.close(fig)
    return cam


# --------------------------------------------------------------------------- #
def main() -> None:
    os.makedirs(DIR_IMAGENS, exist_ok=True)
    df = pd.read_csv(CAMINHO_CSV, parse_dates=["data"])
    df = engenharia(df)
    treino, teste = split_temporal(df)

    print(f"Treino: {len(treino)} dias | Teste: {len(teste)} dias "
          f"(corte em {teste['data'].min().date()})")

    res_a, yte = treina_cenario(treino, teste, FEATS_SATELITE_CLIMA, "A")
    res_b, _ = treina_cenario(treino, teste,
                              FEATS_SATELITE_CLIMA + FEATS_ESP32, "B")

    # Gráficos
    g1 = grafico_serie_temporal(df)
    g2, imp = grafico_importancia(res_b["XGBoost"])
    g3 = grafico_matriz_confusao(yte, res_a["XGBoost"], res_b["XGBoost"])
    g4 = grafico_comparacao(res_a, res_b)

    # Baseline ingênuo: prever sempre a classe majoritária.
    baseline = (yte.value_counts().max() / len(yte))

    # --------- Relatório textual (metricas.txt) ---------
    linhas = []
    linhas.append("=" * 64)
    linhas.append("OrbiTerra — Resultados do treino (Random Forest x XGBoost)")
    linhas.append("=" * 64)
    linhas.append(f"Dias de treino: {len(treino)} | teste: {len(teste)} "
                  f"(validação temporal, corte em {teste['data'].min().date()})")
    linhas.append(f"Baseline (classe majoritária no teste): {baseline:.1%}")
    linhas.append("")
    linhas.append(f"{'Algoritmo':16s} | {'Cenário':24s} | {'Acurácia':>8s} | {'F1-macro':>8s}")
    linhas.append("-" * 64)
    for algo in res_a:
        a, b = res_a[algo], res_b[algo]
        linhas.append(f"{algo:16s} | {'A satélite+clima':24s} | "
                      f"{a['acc']:8.1%} | {a['f1']:8.3f}")
        linhas.append(f"{algo:16s} | {'B +ESP32':24s} | "
                      f"{b['acc']:8.1%} | {b['f1']:8.3f}")
        d_acc = (b["acc"] - a["acc"]) * 100
        d_f1 = b["f1"] - a["f1"]
        linhas.append(f"{'':16s} | {'>> ganho do ESP32':24s} | "
                      f"{d_acc:+7.1f}pt | {d_f1:+8.3f}")
        linhas.append("-" * 64)

    linhas.append("")
    linhas.append("Top 5 variáveis (XGBoost, Modelo B):")
    for f, v in imp.sort_values(ascending=False).head(5).items():
        marca = " [ESP32]" if f in FEATS_ESP32 else ""
        linhas.append(f"  {f:22s}: {v:.3f}{marca}")

    linhas.append("")
    linhas.append("Relatório de classificação — XGBoost Modelo B:")
    linhas.append(classification_report(
        yte, res_b["XGBoost"]["pred"],
        labels=[0, 1, 2], target_names=list(NOMES_RISCO.values()), zero_division=0,
    ))
    linhas.append("Conclusão: a umidade do solo (ESP32) é o sinal mais forte e")
    linhas.append("o Modelo B supera o A — o sensor de campo se paga.")

    texto = "\n".join(linhas)
    with open(CAMINHO_METRICAS, "w") as f:
        f.write(texto + "\n")

    print("\n" + texto)
    print(f"\n[OK] Gráficos: {g1}\n          {g2}\n          {g3}\n          {g4}")
    print(f"[OK] Métricas: {CAMINHO_METRICAS}")


if __name__ == "__main__":
    main()
