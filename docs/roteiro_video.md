# Roteiro do vídeo demonstrativo (até 5 min)

> Configurar no YouTube como **"Não listado"**. Colar o link no README e ao final do PDF.
> **Para concorrer ao pódio:** dizer o **nome do grupo** e a frase **"QUERO CONCORRER"** nos
> primeiros segundos (e mantê-la no início do PDF). Se NÃO for concorrer, basta omitir a frase.

**Tempo-alvo:** 4min30. Divisão sugerida abaixo (todos os integrantes devem aparecer/falar).

---

### 0:00–0:30 · Abertura e equipe
- "Somos o grupo **[nome]**, da FIAP." *(se for concorrer:)* "**QUERO CONCORRER**."
- Apresentar os integrantes (rápido).
- Pitch em uma frase: *"OrbiTerra usa tecnologia espacial para prever o estresse hídrico no café —
  do satélite ao pé da planta."*

### 0:30–1:15 · O problema e a pergunta da GS
- Tema da GS: **economia espacial**; nossa aposta é a **observação da Terra** (downstream).
- Problema real: o café arábica é **muito sensível à seca**; quem decide "quando irrigar" hoje vai
  no feeling. A Mogiana / Sul de Minas é perto da equipe e estratégica.
- A pergunta que respondemos: *como antecipar a falta d'água antes que vire perda?*

### 1:15–2:15 · A ideia e a arquitetura (mostrar `docs/arquitetura.md`)
- **Três olhares:** satélite (NDVI/LST — macro, mas atrasado e com nuvem), ESP32 (umidade do solo —
  micro, em tempo real) e a **IA** que funde os dois.
- Destacar a tese: **o solo reage ANTES do NDVI** → alerta antecipado (~6 dias).
- Mostrar o **ESP32 simulado** rodando: `python src/emulador_esp32.py` (leituras em JSON + alerta).

### 2:15–3:30 · Demonstração prática (rodar ao vivo)
- `python src/simulador_dados.py` → explicar a base sintética plausível (balanço hídrico do solo).
- `python src/treino_modelo.py` → mostrar o resultado-chave:
  - **Modelo A (só satélite+clima) × Modelo B (+ESP32)**, validação temporal.
  - **RF 83,9% → 97,4% (+13,6 pt)** · **XGB 78,8% → 93,0% (+14,3 pt)**; baseline 68,9%.
  - Gráfico de **importância**: a **umidade do solo do ESP32** é a variável nº 1.
  - **Matriz de confusão**: o Modelo A subestima o risco; o B corrige (não troca "Médio" por "Baixo").
- `streamlit run src/dashboard.py` → mostrar mapa, risco previsto, série NDVI × umidade do solo e a
  recomendação de manejo.

### 3:30–4:15 · Integração de disciplinas (critério do pódio)
- Python/lógica, **análise de dados (Pandas)**, **visualização (Matplotlib + Streamlit)**,
  **Machine Learning (XGBoost/Random Forest)**, **IoT/embarcados (ESP32)** e **dados/APIs (JSON/CSV)**.
- Frisar a honestidade: dados sintéticos reproduzíveis, **sem vazamento** (o alvo não é feature),
  caminho real via Google Earth Engine.

### 4:15–4:30 · Fechamento
- "O sensor de campo **se paga**: o produtor ganha dias de antecedência para irrigar."
- Agradecer e reexibir o link do repositório.

---

## Checklist de gravação
- [ ] Vídeo ≤ 5 min, configurado como **Não listado** no YouTube.
- [ ] Nome do grupo dito no início *(+ "QUERO CONCORRER" se for ao pódio)*.
- [ ] Demonstração real dos scripts e do dashboard aparece na tela.
- [ ] Link do vídeo colado no README e ao final do PDF.
- [ ] Áudio audível; tela legível (aumentar fonte do terminal/navegador).
