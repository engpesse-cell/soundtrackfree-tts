# soundtrackfree-tts

Worker de Runpod Serverless que transforma um texto curto em **narração + trilha abaixada**
(MP3, −16 LUFS), usando VoxCPM2 (Apache-2.0) em 48 kHz.

- `handler.py` — recebe `{text, gender, age, mood, accent, music, cue}` e devolve MP3 em base64
- `Dockerfile` — base `runpod/pytorch` com torch 2.8 / cu128 e os pesos já embutidos
- limite de 45 palavras por amostra (regra de produto)

Deploy: Runpod Console → New Endpoint → **Import Git Repository** → este repo →
GPU (A4000/L4/4090) → min workers **0**, max 1 → container disk 20 GB.
