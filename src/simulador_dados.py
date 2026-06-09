"""
OrbiTerra · Simulador de dados (satélite + clima + ESP32)
=========================================================

Gera ~3 anos de dados DIÁRIOS, fisicamente plausíveis, para uma lavoura de café
arábica no corredor Mogiana Paulista / Sul de Minas, e rotula o risco de estresse
hídrico dos PRÓXIMOS 7 dias em 3 classes (baixo / médio / alto).

Por que dados sintéticos: mantêm a POC 100% reproduzível (sem depender de
credenciais do Google Earth Engine nem de hardware). Caminho futuro: trocar este
módulo por extração real de NDVI/LST (Sentinel-2/MODIS) e CHIRPS/NASA POWER.

Princípio-chave (e a "alma" do projeto):
    Existe um BALANÇO HÍDRICO DO SOLO latente (a água realmente disponível na zona
    da raiz). Ele é a VERDADE que define o risco — é o ALVO, nunca uma feature.
    - O ESP32 OBSERVA esse balanço quase em tempo real (com ruído) -> indicador
      ANTECIPADO.
    - O NDVI do satélite REAGE a ele com atraso (~10 dias) e ainda sofre lacunas de
      nuvem -> indicador ATRASADO.
    É por isso que, no treino, o sensor de solo agrega poder preditivo real.

Saída: data/lavoura_cafe.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Parâmetros gerais (seed fixa => reprodutível)
# --------------------------------------------------------------------------- #
SEED = 42
DATA_INICIO = "2022-01-01"
N_DIAS = 365 * 3 + 1          # ~3 anos
HORIZONTE_RISCO = 7           # janela de previsão: próximos 7 dias

# Localização aproximada do talhão de referência (Mogiana / Franca-SP)
LAT_TALHAO = -20.78
LON_TALHAO = -47.30

# Limiares de risco sobre a ÁGUA MÍNIMA DO SOLO nos próximos 7 dias (0–100)
LIMIAR_ALTO = 28.0            # min < 28  -> risco ALTO
LIMIAR_MEDIO = 58.0           # 28..58    -> risco MÉDIO ; >=58 -> BAIXO

CAMINHO_SAIDA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "lavoura_cafe.csv",
)


def _estacao_chuvosa(mes: np.ndarray) -> np.ndarray:
    """Estação chuvosa no Sudeste: outubro a março (hemisfério sul)."""
    return np.isin(mes, [10, 11, 12, 1, 2, 3])


def simular() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    datas = pd.date_range(DATA_INICIO, periods=N_DIAS, freq="D")
    doy = datas.dayofyear.to_numpy()
    mes = datas.month.to_numpy()
    chuvosa = _estacao_chuvosa(mes)

    # ---- Temperatura do ar (NASA POWER) -------------------------------------
    # Pico de calor em jan/fev (hemisfério sul), mínimo em jul. Média ~21 °C.
    fase = 2 * np.pi * (doy - 15) / 365.0
    temp_sazonal = 21.0 + 4.5 * np.cos(fase)
    temp_ar = temp_sazonal + rng.normal(0, 1.6, N_DIAS)

    # ---- Precipitação (CHIRPS) ----------------------------------------------
    # Estação chuvosa: chove mais e mais forte. Seca: raro e fraco.
    prob_chuva = np.where(chuvosa, 0.55, 0.12)
    chove = rng.random(N_DIAS) < prob_chuva
    intensidade = np.where(chuvosa, 13.0, 5.0)
    precip = chove * rng.exponential(intensidade, N_DIAS)

    # "Veranicos": janelas secas dentro da estação chuvosa (estresse de verão).
    # Sorteamos alguns inícios de estiagem e zeramos a chuva por alguns dias.
    for _ in range(8):
        ini = rng.integers(0, N_DIAS - 12)
        dur = rng.integers(7, 13)
        precip[ini:ini + dur] = 0.0

    precip = np.round(precip, 1)

    # ---- Balanço hídrico do solo (LATENTE — é o ALVO, não é feature) --------
    # Modelo de "balde": água_disp += chuva_infiltrada - evapotranspiração.
    # ET sobe com o calor; chuva entra com perdas (escoamento/percolação).
    et0 = np.clip(2.4 + 0.16 * (temp_ar - 17.0), 1.2, 6.5)   # demanda diária
    infiltra = 0.85
    agua = np.empty(N_DIAS)
    agua[0] = 72.0
    for t in range(1, N_DIAS):
        # Recarga capilar do subsolo: ascende mais água quando a raiz está seca,
        # freando a queda (evita "chão" prolongado em zero e gera mais transição).
        recarga = 0.9 * max(0.0, (40.0 - agua[t - 1]) / 40.0)
        novo = agua[t - 1] + infiltra * precip[t] + recarga - et0[t]
        agua[t] = float(np.clip(novo, 0.0, 100.0))

    # ---- Sensor ESP32 (olho micro) ------------------------------------------
    # Umidade do solo = leitura quase em tempo real do balanço hídrico + ruído.
    # É o indicador ANTECIPADO: enxerga a água da raiz no dia exato.
    umidade_solo = np.clip(agua + rng.normal(0, 6.5, N_DIAS), 0, 100)
    umidade_solo = np.round(umidade_solo, 1)

    # DHT22 do ESP32: temperatura e umidade relativa do ar (alta frequência).
    temp_ar_esp = np.round(temp_ar + rng.normal(0, 0.8, N_DIAS), 1)
    # Umidade relativa cai quando está quente e o solo seco (ar mais seco).
    umidade_ar = np.clip(
        82 - 0.9 * (temp_ar - 21) - 0.25 * (100 - agua) + rng.normal(0, 4, N_DIAS),
        20, 100,
    )
    umidade_ar = np.round(umidade_ar, 1)

    # ---- Satélite óptico (olho macro) — REAGE COM ATRASO --------------------
    # NDVI "real" segue uma versão SUAVIZADA e ATRASADA da água do solo
    # (a folha só murcha dias depois que a raiz sente a seca) + sazonalidade.
    agua_suave = pd.Series(agua).rolling(10, min_periods=1).mean().to_numpy()
    agua_atrasada = np.concatenate([np.full(7, agua_suave[0]), agua_suave[:-7]])
    ndvi_real = (
        0.42
        + 0.0035 * agua_atrasada                 # vigor segue a água (atrasado)
        + 0.04 * np.cos(fase)                     # leve ciclo de copa
        + rng.normal(0, 0.015, N_DIAS)
    )
    ndvi_real = np.clip(ndvi_real, 0.15, 0.92)

    # Lacuna de nuvem: o satélite só "enxerga" em dias limpos; nos demais,
    # carregamos a última imagem válida (NDVI fica defasado/"velho").
    revisita_ok = rng.random(N_DIAS) < np.where(chuvosa, 0.28, 0.7)
    revisita_ok[0] = True
    ndvi_obs = ndvi_real.copy()
    ultimo = ndvi_real[0]
    for t in range(N_DIAS):
        if revisita_ok[t]:
            ultimo = ndvi_real[t]
        ndvi_obs[t] = ultimo
    ndvi_obs = np.round(ndvi_obs, 3)

    # EVI: índice irmão, escala um pouco menor.
    evi_obs = np.round(np.clip(0.65 * ndvi_obs + 0.02, 0.05, 0.85), 3)

    # LST (MODIS): temperatura de superfície — solo seco esquenta mais.
    lst = temp_ar + 6.0 + 0.09 * (100 - agua) + rng.normal(0, 1.2, N_DIAS)
    lst = np.round(lst, 1)

    # ---- RÓTULO: risco de estresse hídrico nos PRÓXIMOS 7 dias --------------
    # Olhamos a MENOR água do solo na janela futura (o pior momento que vem).
    serie_agua = pd.Series(agua)
    agua_min_futura = (
        serie_agua.shift(-1)
        .rolling(HORIZONTE_RISCO, min_periods=HORIZONTE_RISCO)
        .min()
        .to_numpy()
    )
    risco = np.full(N_DIAS, -1, dtype=int)
    valido = ~np.isnan(agua_min_futura)
    risco[valido & (agua_min_futura < LIMIAR_ALTO)] = 2           # ALTO
    risco[valido & (agua_min_futura >= LIMIAR_ALTO)
                 & (agua_min_futura < LIMIAR_MEDIO)] = 1          # MÉDIO
    risco[valido & (agua_min_futura >= LIMIAR_MEDIO)] = 0         # BAIXO

    df = pd.DataFrame({
        "data": datas,
        "lat": LAT_TALHAO,
        "lon": LON_TALHAO,
        # --- satélite + clima (Modelo A) ---
        "ndvi": ndvi_obs,
        "evi": evi_obs,
        "lst": lst,
        "precip_mm": precip,
        "temp_ar": np.round(temp_ar, 1),
        # --- ESP32 / campo (entra só no Modelo B) ---
        "umidade_solo": umidade_solo,
        "temp_ar_esp": temp_ar_esp,
        "umidade_ar": umidade_ar,
        # --- rótulo ---
        "risco_7d": risco,
    })

    # Descartamos as últimas linhas sem janela futura completa (rótulo = -1).
    df = df[df["risco_7d"] >= 0].reset_index(drop=True)
    return df


def main() -> None:
    df = simular()
    os.makedirs(os.path.dirname(CAMINHO_SAIDA), exist_ok=True)
    df.to_csv(CAMINHO_SAIDA, index=False)

    nomes = {0: "BAIXO", 1: "MÉDIO", 2: "ALTO"}
    dist = df["risco_7d"].value_counts().sort_index()
    print(f"[OK] {len(df)} dias gerados -> {CAMINHO_SAIDA}")
    print(f"Período: {df['data'].min().date()} a {df['data'].max().date()}")
    print("Distribuição do risco (próximos 7 dias):")
    for k, v in dist.items():
        print(f"  {nomes[k]:6s}: {v:4d}  ({100 * v / len(df):4.1f}%)")
    print(f"NDVI: {df['ndvi'].min():.2f}–{df['ndvi'].max():.2f} | "
          f"Umidade solo: {df['umidade_solo'].min():.0f}–{df['umidade_solo'].max():.0f}%")


if __name__ == "__main__":
    main()
