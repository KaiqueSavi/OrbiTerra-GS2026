"""
OrbiTerra · Emulador do ESP32 (estação de campo simulada)
=========================================================

Decisão do projeto: a POC usa o ESP32 SIMULADO (sem placa física). Este script
faz o papel do firmware embarcado — gera leituras dos sensores em "tempo real" e
as transmite como a placa faria, em JSON, a cada intervalo.

Sensores emulados (mesmos de uma estação real):
    - Umidade do solo na zona da raiz (sensor capacitivo)   -> %
    - Temperatura do ar (DHT22)                              -> °C
    - Umidade relativa do ar (DHT22)                         -> %

Cada leitura recebe um ALERTA LOCAL (lógica "na borda"/edge): mesmo offline, a
estação já avisa quando o solo está crítico — exatamente o ganho de antecedência
que o satélite (defasado e com nuvem) não dá.

Uso:
    python src/emulador_esp32.py                 # 12 leituras, 1/seg, no console
    python src/emulador_esp32.py --n 30 --intervalo 0.5
    python src/emulador_esp32.py --enviar http://localhost:8000/leituras

--------------------------------------------------------------------------------
Equivalente no firmware real (Arduino/ESP32, apenas ilustrativo):

    void loop() {
        float solo = lerSensorCapacitivo(PINO_SOLO);   // %
        float temp = dht.readTemperature();            // °C
        float umid = dht.readHumidity();               // %
        String alerta = solo < 28 ? "ALTO" : solo < 58 ? "MEDIO" : "BAIXO";
        String json = montarJson(solo, temp, umid, alerta);
        enviarHTTP(SERVIDOR, json);                    // WiFi -> pipeline
        delay(INTERVALO_MS);
    }
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

# Limiares de alerta na borda (espelham os limiares do rótulo de risco).
LIMIAR_ALTO = 28.0
LIMIAR_MEDIO = 58.0

ID_ESTACAO = "ESP32-MOGIANA-01"
LAT, LON = -20.78, -47.30


def nivel_alerta(umidade_solo: float) -> str:
    if umidade_solo < LIMIAR_ALTO:
        return "ALTO"
    if umidade_solo < LIMIAR_MEDIO:
        return "MEDIO"
    return "BAIXO"


def gera_leitura(passo: int, estado: dict, rng) -> dict:
    """Simula um ciclo de leitura física plausível.

    O solo tem inércia (muda devagar) e segue uma leve tendência de seca; o ar
    oscila com um ciclo diário (mais quente/seco ao 'meio-dia' da simulação).
    """
    # Solo: passeio aleatório suave em torno do estado, com leve secagem.
    estado["solo"] = max(0.0, min(100.0,
        estado["solo"] - 0.4 + rng.normal(0, 1.2)))
    solo = round(estado["solo"], 1)

    # Ciclo diário do ar (seno) + ruído. 24 passos = 1 "dia".
    fase = 2 * math.pi * (passo % 24) / 24.0
    temp = round(24.0 + 5.0 * math.sin(fase - math.pi / 2) + rng.normal(0, 0.5), 1)
    # Umidade do ar anda ao contrário da temperatura e do solo seco.
    umid_ar = round(max(20.0, min(100.0,
        80.0 - 0.8 * (temp - 24) - 0.2 * (100 - solo) + rng.normal(0, 3))), 1)

    return {
        "id_estacao": ID_ESTACAO,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lat": LAT,
        "lon": LON,
        "umidade_solo": solo,
        "temp_ar": temp,
        "umidade_ar": umid_ar,
        "alerta_local": nivel_alerta(solo),
    }


def transmite(leitura: dict, url: str | None) -> None:
    """Imprime a leitura (como log serial) e, se houver URL, envia via HTTP."""
    emoji = {"ALTO": "🔴", "MEDIO": "🟡", "BAIXO": "🟢"}[leitura["alerta_local"]]
    print(f"{emoji} {leitura['timestamp']} | solo={leitura['umidade_solo']:5.1f}% "
          f"| ar={leitura['temp_ar']:4.1f}°C/{leitura['umidade_ar']:4.1f}% "
          f"| alerta={leitura['alerta_local']}")
    print("   " + json.dumps(leitura, ensure_ascii=False))

    if url:
        try:
            import urllib.request
            req = urllib.request.Request(
                url, data=json.dumps(leitura).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            urllib.request.urlopen(req, timeout=3)
            print(f"   -> enviado para {url}")
        except Exception as e:  # POC: não derrubar a estação por falha de rede
            print(f"   -> falha ao enviar ({e.__class__.__name__}): seguindo offline")


def main() -> None:
    ap = argparse.ArgumentParser(description="Emulador da estação ESP32 (OrbiTerra)")
    ap.add_argument("--n", type=int, default=12, help="número de leituras")
    ap.add_argument("--intervalo", type=float, default=1.0, help="segundos entre leituras")
    ap.add_argument("--enviar", type=str, default=None, help="URL HTTP de destino (opcional)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(args.seed)
    estado = {"solo": 35.0}                 # começa em situação de atenção

    print(f"== Estação {ID_ESTACAO} on-line ({LAT}, {LON}) ==")
    print(f"   {args.n} leituras a cada {args.intervalo}s"
          + (f" -> {args.enviar}" if args.enviar else " (somente console)"))
    print("-" * 72)

    for passo in range(args.n):
        leitura = gera_leitura(passo, estado, rng)
        transmite(leitura, args.enviar)
        if passo < args.n - 1:
            time.sleep(args.intervalo)

    print("-" * 72)
    print("== Estação encerrada ==")


if __name__ == "__main__":
    main()
