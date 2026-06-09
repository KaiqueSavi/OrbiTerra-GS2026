"""
OrbiTerra · Painel de decisão (Streamlit)
=========================================

Transforma o dado em DECISÃO: mostra onde está o talhão, a situação atual do
risco hídrico (previsto pela IA para os próximos 7 dias), a série de NDVI x
umidade do solo e a comparação dos modelos.

Executar (a partir da pasta orbiterra_mvp/):
    streamlit run src/dashboard.py

Pré-requisito: rodar antes o simulador e o treino para gerar os dados:
    python src/simulador_dados.py
    python src/treino_modelo.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st

# Reaproveita TODO o pipeline já validado do treino (sem duplicar lógica).
from treino_modelo import (
    engenharia, split_temporal, treina_cenario,
    FEATS_SATELITE_CLIMA, FEATS_ESP32, NOMES_RISCO, CORES_RISCO, CAMINHO_CSV,
)

st.set_page_config(page_title="OrbiTerra · Painel de risco hídrico",
                   page_icon="🛰️", layout="wide")


@st.cache_data
def carrega_dados() -> pd.DataFrame:
    df = pd.read_csv(CAMINHO_CSV, parse_dates=["data"])
    return engenharia(df)


@st.cache_resource
def treina():
    df = carrega_dados()
    treino, teste = split_temporal(df)
    feats_b = FEATS_SATELITE_CLIMA + FEATS_ESP32
    res_a, yte = treina_cenario(treino, teste, FEATS_SATELITE_CLIMA, "A")
    res_b, _ = treina_cenario(treino, teste, feats_b, "B")
    modelo_b = res_b["XGBoost"]["modelo"]
    metricas = {
        "acc_a": res_a["XGBoost"]["acc"], "acc_b": res_b["XGBoost"]["acc"],
        "acc_a_rf": res_a["Random Forest"]["acc"],
        "acc_b_rf": res_b["Random Forest"]["acc"],
    }
    return modelo_b, feats_b, metricas


# --------------------------------------------------------------------------- #
df = carrega_dados()
modelo, feats_b, met = treina()

st.title("🛰️ OrbiTerra — Inteligência espacial contra o estresse hídrico no café")
st.caption("Satélite + sensor de campo (ESP32) + IA · corredor Mogiana Paulista / "
           "Sul de Minas · POC Global Solution 2026.1 (FIAP)")

# ---- Barra lateral: escolha do dia de análise ----
st.sidebar.header("Painel de controle")
datas = df["data"].dt.date.tolist()
dia = st.sidebar.select_slider("Data de análise", options=datas, value=datas[-1])
mask_dia = df["data"].dt.date == dia
linha = df[mask_dia].iloc[0]

# Previsão de risco da IA para o dia selecionado (1 linha, tipos float).
X = df.loc[mask_dia, feats_b].astype(float)
risco_prev = int(modelo.predict(X)[0])
proba = modelo.predict_proba(X)[0]

# ---- Faixa de alerta atual ----
nome_risco = NOMES_RISCO[risco_prev]
cor = CORES_RISCO[risco_prev]
recomendacao = {
    0: "Sem ação imediata. Manter monitoramento de rotina.",
    1: "Atenção: programar irrigação e reavaliar em 48 h.",
    2: "Crítico: irrigar com prioridade e inspecionar o talhão.",
}[risco_prev]

st.markdown(
    f"""<div style="background:{cor};padding:16px 20px;border-radius:10px;color:white">
    <span style="font-size:14px;opacity:.9">RISCO HÍDRICO PREVISTO (próximos 7 dias) — {dia}</span><br>
    <span style="font-size:30px;font-weight:700">{nome_risco.upper()}</span>
    &nbsp;·&nbsp; {recomendacao}
    </div>""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Umidade do solo (ESP32)", f"{linha['umidade_solo']:.0f}%")
c2.metric("NDVI (satélite)", f"{linha['ndvi']:.2f}")
c3.metric("Chuva acum. 7d", f"{linha['precip_7d']:.0f} mm")
c4.metric("Dias sem chuva", f"{int(linha['dias_sem_chuva'])}")

st.write("")

# ---- Duas colunas: mapa + confiança da IA ----
col_mapa, col_conf = st.columns([1.4, 1])
with col_mapa:
    st.subheader("📍 Talhão monitorado")
    st.map(pd.DataFrame({"lat": [linha["lat"]], "lon": [linha["lon"]]}), zoom=8)
with col_conf:
    st.subheader("🤖 Confiança da IA (XGBoost)")
    conf = pd.DataFrame({
        "Classe": [NOMES_RISCO[i] for i in range(3)],
        "Probabilidade": proba,
    }).set_index("Classe")
    st.bar_chart(conf, color="#1f6f4a")

# ---- Série temporal NDVI x umidade do solo ----
st.subheader("📈 NDVI (satélite, atrasado) × Umidade do solo (ESP32, antecipado)")
janela = df[df["data"].dt.date <= dia].tail(180)
serie = janela.set_index("data")[["umidade_solo", "ndvi"]].rename(
    columns={"umidade_solo": "Umidade solo (%)", "ndvi": "NDVI"})
st.line_chart(serie)
st.caption("O solo (ESP32) cai ANTES do NDVI — é essa antecedência que gera o alerta precoce.")

# ---- Comparação de modelos ----
st.subheader("✅ O sensor de campo se paga")
m1, m2, m3 = st.columns(3)
m1.metric("XGBoost — só satélite+clima (A)", f"{met['acc_a']:.0%}")
m2.metric("XGBoost — +ESP32 (B)", f"{met['acc_b']:.0%}",
          delta=f"+{(met['acc_b']-met['acc_a'])*100:.1f} pt")
m3.metric("Random Forest — +ESP32 (B)", f"{met['acc_b_rf']:.0%}",
          delta=f"+{(met['acc_b_rf']-met['acc_a_rf'])*100:.1f} pt")
st.caption("Acurácia no conjunto de teste (validação temporal). "
           "Adicionar o ESP32 eleva a previsão em ambos os algoritmos.")
