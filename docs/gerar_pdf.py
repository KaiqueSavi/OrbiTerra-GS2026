"""
OrbiTerra · Gerador do PDF de entrega (Global Solution 2026.1)
==============================================================

Monta o PDF único exigido pelo edital, com:
  - 1ª página: nomes dos integrantes (+ espaço para "QUERO CONCORRER", se for ao pódio);
  - estrutura obrigatória: Introdução, Desenvolvimento, Resultados Esperados, Conclusões;
  - códigos principais EM TEXTO (nunca print/screenshot);
  - gráficos do projeto embutidos;
  - links do repositório e do vídeo ao final.

Requisito extra (só para gerar o PDF): pip install reportlab
Uso:
    python docs/gerar_pdf.py
Saída:
    docs/OrbiTerra_Entrega_GS2026.pdf
"""

from __future__ import annotations

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, Image,
    Table, TableStyle, PageBreak, HRFlowable,
)

DIR = os.path.dirname(os.path.abspath(__file__))
DIR_IMG = os.path.join(DIR, "imagens")
SAIDA = os.path.join(DIR, "OrbiTerra_Entrega_GS2026.pdf")

VERDE = colors.HexColor("#1f6f4a")
AZUL = colors.HexColor("#14304a")
CINZA = colors.HexColor("#5b6b73")

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
ss = getSampleStyleSheet()
S = {
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], textColor=AZUL,
                         fontSize=16, spaceBefore=10, spaceAfter=8),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], textColor=VERDE,
                         fontSize=12.5, spaceBefore=10, spaceAfter=5),
    "body": ParagraphStyle("body", parent=ss["BodyText"], fontSize=10,
                           leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7),
    "cap": ParagraphStyle("cap", parent=ss["BodyText"], fontSize=8.5,
                          textColor=CINZA, alignment=TA_CENTER, spaceAfter=10),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=7.6,
                           leading=9.4, textColor=colors.HexColor("#1b1b1b")),
    "capa_titulo": ParagraphStyle("ct", parent=ss["Title"], textColor=AZUL,
                                  fontSize=30, leading=34, spaceAfter=4),
    "capa_sub": ParagraphStyle("cs", parent=ss["Title"], textColor=VERDE,
                               fontSize=14, leading=18, spaceAfter=2),
    "center": ParagraphStyle("center", parent=ss["BodyText"], fontSize=10.5,
                             leading=15, alignment=TA_CENTER),
}


def P(txt, st="body"):
    return Paragraph(txt, S[st])


def code(txt):
    return Preformatted(txt.strip("\n"), S["code"])


def imagem(nome, largura_cm=15.8, legenda=None):
    """Insere imagem mantendo proporção, com legenda opcional."""
    cam = os.path.join(DIR_IMG, nome)
    iw, ih = ImageReader(cam).getSize()
    w = largura_cm * cm
    h = w * ih / iw
    flow = [Image(cam, width=w, height=h)]
    if legenda:
        flow.append(P(legenda, "cap"))
    return flow


def hr():
    return HRFlowable(width="100%", thickness=0.6, color=VERDE,
                      spaceBefore=4, spaceAfter=8)


# --------------------------------------------------------------------------- #
# Trechos de código (em TEXTO) — fiéis aos arquivos em src/
# --------------------------------------------------------------------------- #
COD_BALANCO = """
# src/simulador_dados.py — balanço hídrico do solo (LATENTE: é o ALVO, não feature)
et0 = np.clip(2.4 + 0.16 * (temp_ar - 17.0), 1.2, 6.5)        # demanda diária (ET)
agua = np.empty(N_DIAS); agua[0] = 72.0
for t in range(1, N_DIAS):
    recarga = 0.9 * max(0.0, (40.0 - agua[t-1]) / 40.0)       # capilaridade do subsolo
    novo = agua[t-1] + 0.85 * precip[t] + recarga - et0[t]    # entra chuva, sai ET
    agua[t] = float(np.clip(novo, 0.0, 100.0))

# Sensor ESP32: OBSERVA a água do solo com ruido (indicador ANTECIPADO)
umidade_solo = np.clip(agua + rng.normal(0, 6.5, N_DIAS), 0, 100)
"""

COD_ROTULO = """
# src/simulador_dados.py — rótulo: risco nos PRÓXIMOS 7 dias (3 classes)
# Olhamos a MENOR água do solo na janela futura (o pior momento que vem).
agua_min_futura = (pd.Series(agua).shift(-1)
                   .rolling(7, min_periods=7).min().to_numpy())
risco[agua_min_futura < 28] = 2                               # ALTO
risco[(agua_min_futura >= 28) & (agua_min_futura < 58)] = 1   # MEDIO
risco[agua_min_futura >= 58] = 0                              # BAIXO
"""

COD_FEATURES = """
# src/treino_modelo.py — engenharia de variáveis (nada usa o futuro)
# Sinais do ESP32: nivel, media recente e TENDENCIA de 3 dias (alerta precoce!)
df["umidade_solo_7d"]     = df["umidade_solo"].rolling(7, min_periods=1).mean()
df["umidade_solo_tend_3d"] = df["umidade_solo"].diff(3).fillna(0.0)
# Clima acumulado (memoria de chuva) e estiagem
df["precip_7d"]  = df["precip_mm"].rolling(7,  min_periods=1).sum()
df["precip_30d"] = df["precip_mm"].rolling(30, min_periods=1).sum()
"""

COD_TREINO = """
# src/treino_modelo.py — dois algoritmos, validacao TEMPORAL (treina passado, testa futuro)
modelos = {
  "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                     min_samples_leaf=5, class_weight="balanced", random_state=42),
  "XGBoost":       XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                     subsample=0.9, colsample_bytree=0.9, num_class=3),
}
corte = int(len(df) * 0.75)               # 75% do tempo para treino, 25% final para teste
treino, teste = df.iloc[:corte], df.iloc[corte:]
# Cenarios:  A = satelite + clima   |   B = A + ESP32  -> comparamos A x B
"""

COD_ESP32 = """
# src/emulador_esp32.py — firmware do ESP32 SIMULADO (alerta na borda + JSON via WiFi)
def nivel_alerta(umidade_solo):
    if umidade_solo < 28:  return "ALTO"
    if umidade_solo < 58:  return "MEDIO"
    return "BAIXO"

# Cada leitura sai como um JSON, igual a placa enviaria por HTTP/MQTT:
# {"id_estacao": "ESP32-MOGIANA-01", "umidade_solo": 31.4,
#  "temp_ar": 21.0, "umidade_ar": 68.6, "alerta_local": "ALTO"}
"""


# --------------------------------------------------------------------------- #
def construir():
    doc = SimpleDocTemplate(
        SAIDA, pagesize=A4,
        leftMargin=1.9 * cm, rightMargin=1.9 * cm,
        topMargin=1.8 * cm, bottomMargin=1.7 * cm,
        title="OrbiTerra — Entrega Global Solution 2026.1",
        author="Grupo OrbiTerra — FIAP",
    )
    E = []  # elementos

    # ----------------------- CAPA / 1ª PÁGINA -----------------------
    E += [Spacer(1, 1.2 * cm)]
    E += [P("GLOBAL SOLUTION 2026.1 &nbsp;·&nbsp; FIAP &nbsp;·&nbsp; ECONOMIA ESPACIAL", "cap")]
    E += [P("OrbiTerra", "capa_titulo")]
    E += [P("Inteligência espacial contra o estresse hídrico no café", "capa_sub")]
    E += [P("Satélite + sensor de campo (ESP32) + IA — transformando dado em decisão", "cap")]
    E += [Spacer(1, 0.5 * cm), hr()]

    E += [P("Grupo: <b>[nome do grupo]</b>", "center")]
    E += [Spacer(1, 0.3 * cm)]
    E += [P("<b>Integrantes:</b>", "center")]
    integrantes = [
        "Kaique — RM562072",
        "Willian — RM571013",
    ]
    for nome in integrantes:
        E += [P(nome, "center")]
    E += [Spacer(1, 0.4 * cm)]
    # Espaço reservado para a frase do pódio (preencher se for concorrer).
    E += [P('<i>Para concorrer ao pódio, inserir aqui a frase: '
            '<b>"QUERO CONCORRER"</b>.</i>', "center")]
    E += [Spacer(1, 0.5 * cm), hr()]
    E += [P("<b>Tema:</b> Economia espacial · Observação da Terra (downstream)<br/>"
            "<b>Foco da POC:</b> Café arábica — corredor Mogiana Paulista / Sul de Minas<br/>"
            "<b>Documento:</b> Entrega final (POC) — Introdução, Desenvolvimento, "
            "Resultados Esperados e Conclusões", "center")]
    E += [PageBreak()]

    # ----------------------- 1. INTRODUÇÃO -----------------------
    E += [P("1. Introdução", "h1"), hr()]
    E += [P(
        "O <b>OrbiTerra</b> é uma Prova de Conceito (POC) desenvolvida para a "
        "<b>Global Solution 2026.1</b> da FIAP, cujo tema é a <b>economia espacial</b>. "
        "A GS pede uma solução que responda: <i>“Como a tecnologia espacial pode ser "
        "utilizada para melhorar a vida das pessoas, tornar processos mais eficientes e "
        "criar novas oportunidades na Terra ou fora dela?”</i>")]
    E += [P(
        "Nossa resposta parte do maior mercado <i>downstream</i> da economia espacial — a "
        "<b>observação da Terra</b> — e o coloca a serviço de um problema real e mensurável: "
        "a <b>perda de produtividade do café por falta d'água</b>. O café arábica tem faixa "
        "de tolerância climática estreita, e boa parte da produção vem de pequenas e médias "
        "propriedades onde a decisão de “quando irrigar” ainda é tomada por intuição. O foco "
        "é o corredor cafeeiro da <b>Mogiana Paulista / Sul de Minas</b>, próximo à equipe e "
        "estratégico para o país.")]
    E += [P(
        "A ideia central é combinar <b>três olhares</b> sobre a lavoura: o <b>satélite</b> "
        "(visão macro — índices de vegetação NDVI/EVI e temperatura de superfície LST), uma "
        "<b>estação de campo com ESP32</b> (visão micro — umidade do solo na zona da raiz e "
        "temperatura/umidade do ar) e a <b>Inteligência Artificial</b> (XGBoost e Random "
        "Forest), que funde tudo e classifica o <b>risco de estresse hídrico dos próximos 7 "
        "dias</b> em três níveis: Baixo, Médio e Alto. O resultado é exibido em um "
        "<b>painel de decisão</b> que indica onde está o risco e quando agir.")]

    # ----------------------- 2. DESENVOLVIMENTO -----------------------
    E += [P("2. Desenvolvimento", "h1"), hr()]

    E += [P("2.1 Arquitetura da solução (5 camadas)", "h2")]
    E += [P(
        "A solução tem cinco camadas: (1) <b>satélite óptico</b> (Sentinel-2/Landsat via HLS) "
        "para NDVI/EVI e LST; (2) <b>produto térmico</b> (MODIS LST); (3) <b>clima</b> (CHIRPS "
        "para chuva, NASA POWER para temperatura); (4) <b>estação de campo ESP32</b> (umidade "
        "do solo + DHT22), o indicador antecipado; e (5) o <b>pipeline em Python/Pandas → "
        "modelo (XGBoost/Random Forest) → painel (Streamlit)</b>. O princípio-chave: o solo "
        "reage <b>antes</b> do NDVI — por isso o sensor agrega valor preditivo.")]
    E += [P(
        "Na POC, o <b>ESP32 é simulado</b> (o edital permite) e os <b>dados são sintéticos, "
        "mas fisicamente plausíveis</b>, com semente fixa (100% reproduzível). O caminho "
        "futuro é plugar dados reais via Google Earth Engine.")]

    E += [P("2.2 Geração dos dados e o rótulo de risco", "h2")]
    E += [P(
        "O simulador modela um <b>balanço hídrico do solo</b> (modelo tipo “balde”): a água "
        "disponível na raiz sobe com a chuva infiltrada e cai com a evapotranspiração (que "
        "cresce com o calor). Esse balanço latente é a <b>verdade</b> que define o risco — ele "
        "é o <b>alvo</b>, nunca uma variável de entrada (evitando vazamento de dados). O ESP32 "
        "apenas o <i>observa com ruído</i>.")]
    E += [code(COD_BALANCO)]
    E += [P(
        "O <b>rótulo</b> é o risco dos próximos 7 dias, definido pela menor umidade do solo na "
        "janela futura — o “pior momento que vem”:")]
    E += [code(COD_ROTULO)]

    E += [P("2.3 Engenharia de variáveis e treino dos modelos", "h2")]
    E += [P(
        "Com Pandas, criamos variáveis derivadas que dão <b>memória</b> e <b>tendência</b> às "
        "séries — chuva acumulada, dias sem chuva, média e tendência de 3 dias da umidade do "
        "solo (o sinal de alerta precoce) e sazonalidade. Nada usa o futuro.")]
    E += [code(COD_FEATURES)]
    E += [P(
        "Treinamos <b>Random Forest</b> e <b>XGBoost</b> em dois cenários — <b>Modelo A</b> "
        "(satélite + clima) e <b>Modelo B</b> (A + ESP32) — com <b>validação temporal</b>: "
        "treina nos primeiros 75% do tempo e testa nos 25% finais, sem nunca embaralhar as "
        "datas. Assim medimos honestamente quanto o sensor de campo agrega.")]
    E += [code(COD_TREINO)]

    E += [P("2.4 A estação ESP32 (simulada) e o painel", "h2")]
    E += [P(
        "O emulador faz o papel do firmware: gera leituras dos sensores em “tempo real”, "
        "aplica um <b>alerta na borda</b> (mesmo offline a estação já avisa) e transmite cada "
        "leitura em <b>JSON</b>, como a placa enviaria por WiFi (HTTP/MQTT). O painel "
        "(Streamlit) consome os dados e o modelo para mostrar mapa, risco previsto, a série "
        "NDVI × umidade do solo e a recomendação de manejo.")]
    E += [code(COD_ESP32)]

    E += [PageBreak()]

    # ----------------------- 3. RESULTADOS ESPERADOS -----------------------
    E += [P("3. Resultados Esperados", "h1"), hr()]
    E += [P(
        "O experimento confirma a tese: <b>adicionar o ESP32 eleva a previsão</b> em ambos os "
        "algoritmos, com validação temporal. A tabela compara o Modelo A (satélite + clima) "
        "com o Modelo B (+ESP32):")]

    tabela = [
        ["Algoritmo", "Modelo A\n(satélite+clima)", "Modelo B\n(+ESP32)", "Ganho"],
        ["Random Forest", "83,9%", "97,4%", "+13,6 pt"],
        ["XGBoost", "78,8%", "93,0%", "+14,3 pt"],
        ["Baseline (classe maj.)", "68,9%", "—", "—"],
    ]
    t = Table(tabela, colWidths=[5.2 * cm, 4.0 * cm, 3.4 * cm, 3.2 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), AZUL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#eaf3ee")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c7cf")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    E += [t, Spacer(1, 0.3 * cm)]
    E += [P(
        "A <b>variável mais importante</b> do Modelo B é a umidade do solo do ESP32 "
        "(<font face='Courier'>umidade_solo_7d</font> ≈ 0,51, seguida de "
        "<font face='Courier'>umidade_solo</font>). Mais do que acurácia, o ganho é de "
        "<b>segurança</b>: o Modelo A <b>subestima</b> o risco — classifica 49 dias de risco "
        "Médio como Baixo, o que deixaria o produtor sem irrigar — e o <b>Modelo B corrige</b> "
        "esse erro perigoso (ver matriz de confusão).")]

    E += imagem("01_serie_temporal.png", 15.4,
                "Fig. 1 — A umidade do solo (ESP32) cai ANTES do NDVI (satélite), que é "
                "defasado e tem lacunas de nuvem. Essa antecedência gera o alerta precoce.")
    E += imagem("04_comparacao_modelos.png", 11.5,
                "Fig. 2 — Acurácia no teste: o sensor de campo eleva a previsão nos dois "
                "algoritmos.")
    E += [PageBreak()]
    E += imagem("02_importancia_variaveis.png", 15.0,
                "Fig. 3 — Importância das variáveis (XGBoost, Modelo B): os sinais do ESP32 "
                "(verde) dominam o topo.")
    E += imagem("03_matriz_confusao.png", 15.4,
                "Fig. 4 — Matriz de confusão: o Modelo A troca “Médio” por “Baixo” (subestima "
                "o risco); o Modelo B corrige.")

    # ----------------------- 4. CONCLUSÕES -----------------------
    E += [PageBreak()]
    E += [P("4. Conclusões", "h1"), hr()]
    E += [P(
        "O OrbiTerra demonstra, de ponta a ponta, como a <b>tecnologia espacial</b> pode "
        "virar <b>decisão prática</b> no campo. A POC integra dados de satélite, clima e um "
        "sensor de campo (ESP32) e prova, com um experimento controlado e validação temporal, "
        "que <b>o sensor de campo se paga</b>: ele eleva a acurácia (RF +13,6 pt; XGB "
        "+14,3 pt) e, sobretudo, corrige o erro mais perigoso — subestimar o risco hídrico.")]
    E += [P(
        "A solução responde às três partes da pergunta da GS: <b>melhora a vida das pessoas</b> "
        "(protege a renda do cafeicultor e o preço ao consumidor), <b>torna processos mais "
        "eficientes</b> (irrigar na hora certa, sem desperdício) e <b>cria oportunidades</b> "
        "(um serviço de inteligência agrícola replicável e de baixo custo, que pode alimentar "
        "crédito rural e seguro paramétrico).")]
    E += [P(
        "Como POC, assumimos limites com transparência: ESP32 simulado e dados sintéticos "
        "(porém plausíveis e reproduzíveis), sem vazamento de dados. A evolução natural é "
        "substituir o simulador por dados reais via Google Earth Engine (Sentinel-2/MODIS/"
        "CHIRPS) e instalar a placa física no talhão — o código já está preparado para receber "
        "leituras reais via HTTP/MQTT.")]
    E += [P(
        "<b>Integração de disciplinas (Fases 3–4):</b> Lógica e Python; Análise de dados "
        "(Pandas); Visualização (Matplotlib e painel Streamlit); Machine Learning (XGBoost e "
        "Random Forest); IoT/embarcados (ESP32); e Banco de dados/APIs (JSON e CSV).")]

    E += [Spacer(1, 0.4 * cm), hr()]
    E += [P("Links da entrega", "h2")]
    E += [P("• <b>Repositório do projeto:</b> [inserir link do repositório aqui]")]
    E += [P("• <b>Vídeo demonstrativo (YouTube, “Não listado”):</b> [inserir link do vídeo aqui]")]
    E += [P("• <b>Documento de conceito:</b> docs/OrbiTerra_Conceito.pdf  ·  "
            "<b>Arquitetura/fluxograma:</b> docs/arquitetura.md  ·  "
            "<b>Roteiro do vídeo:</b> docs/roteiro_video.md")]

    doc.build(E, onFirstPage=_rodape, onLaterPages=_rodape)
    return SAIDA


def _rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(CINZA)
    canvas.drawString(1.9 * cm, 1.0 * cm,
                      "OrbiTerra · Global Solution 2026.1 · FIAP")
    canvas.drawRightString(A4[0] - 1.9 * cm, 1.0 * cm, f"Página {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    caminho = construir()
    tam = os.path.getsize(caminho) / 1024
    print(f"[OK] PDF gerado: {caminho} ({tam:.0f} KB)")
