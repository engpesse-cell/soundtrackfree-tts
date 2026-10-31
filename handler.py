"""Handler do worker: texto -> voz (+ trilha abaixada), devolve MP3 em base64.

Mesma lógica que já roda na placa, empacotada para Runpod Serverless.
"""
from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf

try:
    import runpod
except ImportError:      # permite testar o handler sem o SDK do worker
    runpod = None

from voxcpm import VoxCPM

SR = 48000
MAX_PALAVRAS = 45
MODELO = None

# trilhas de apoio embutidas na imagem (mesmos cues do site)
CUES = {
    "boss": "/cues/game_boss.mp3",
    "trailer": "/cues/cinema_trailer.mp3",
    "vlog": "/cues/yt_vlog.mp3",
    "doc": "/cues/doc_underscore.mp3",
    "podcast": "/cues/podcast_bed.mp3",
    "promo": "/cues/ad_promo.mp3",
}


def carrega_modelo():
    global MODELO
    if MODELO is None:
        t0 = time.time()
        try:
            MODELO = VoxCPM.from_pretrained("openbmb/VoxCPM2", device="cuda", load_denoiser=False)
        except TypeError:
            MODELO = VoxCPM.from_pretrained("openbmb/VoxCPM2", device="cuda")
        print(f"modelo carregado em {time.time()-t0:.0f}s", flush=True)
    return MODELO


def descreve(p: dict) -> str:
    gen = {"male": "man", "female": "woman", "neutral": "person"}.get(p.get("gender", "male"), "man")
    idade = {"young": "in his late twenties" if gen == "man" else "in her late twenties",
             "adult": "in his forties" if gen == "man" else "in her forties",
             "mature": "in his sixties" if gen == "man" else "in her sixties"}.get(p.get("age", "adult"), "")
    clima = {"calm": "calm, measured, intimate",
             "warm": "warm, friendly, close to the microphone",
             "excited": "energetic and excited",
             "serious": "serious, grave, documentary style",
             "mysterious": "low, mysterious, suspenseful",
             "friendly": "light, conversational, smiling"}.get(p.get("mood", "calm"), "calm")
    return (f"(narrator, {p.get('accent', 'American English')} speaker, {gen} {idade}, {clima}, "
            f"studio quality, natural pacing)")


def cue_por_clima(mood: str) -> str:
    return {"calm": "doc", "warm": "vlog", "excited": "promo", "serious": "trailer",
            "mysterious": "doc", "friendly": "podcast"}.get(mood, "doc")


def handler(event):
    inp = event.get("input") or {}
    texto = re.sub(r"\s+", " ", str(inp.get("text", ""))).strip()
    palavras = len(texto.split())
    if palavras == 0:
        return {"error": "texto vazio"}
    if palavras > MAX_PALAVRAS:
        return {"error": f"limite de {MAX_PALAVRAS} palavras na amostra"}

    t0 = time.time()
    modelo = carrega_modelo()
    audio = modelo.generate(text=f"{descreve(inp)} {texto}")
    a = np.asarray(audio, dtype=np.float32).ravel()
    if a.ndim > 1:
        a = a.mean(axis=-1)
    lim = np.argwhere(np.abs(a) > 0.004)
    if len(lim):
        a = a[: lim[-1][0] + int(SR * 0.06)]
    voz = a / max(np.max(np.abs(a)), 1e-9) * 0.9
    duracao = len(voz) / SR

    with tempfile.TemporaryDirectory() as tmp:
        vp = os.path.join(tmp, "voz.wav")
        sf.write(vp, voz.astype(np.float32), SR)
        outp = os.path.join(tmp, "out.mp3")
        musica = CUES.get(inp.get("cue") or cue_por_clima(inp.get("mood", "calm")))
        if inp.get("music", True) and musica and os.path.exists(musica):
            fc = ("[1:a]loudnorm=I=-16:TP=-2:LRA=8,aresample=48000,asplit=2[v1][v2];"
                  "[0:a]volume=0.5,aresample=48000,atrim=0:60[mus];"
                  "[mus][v1]sidechaincompress=threshold=0.015:ratio=14:attack=12:release=380[dk];"
                  "[dk][v2]amix=inputs=2:duration=first:weights=1 1.5,"
                  "apad=pad_dur=1.0,loudnorm=I=-16:TP=-1.5:LRA=9[out]")
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", musica, "-i", vp, "-filter_complex", fc,
                   "-map", "[out]", "-ac", "2", "-ar", "48000", "-c:a", "libmp3lame",
                   "-b:a", "192k", outp]
        else:
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", vp, "-af", "loudnorm=I=-16:TP=-1.5:LRA=9",
                   "-ac", "2", "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "192k", outp]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            return {"error": "falha na mixagem: " + r.stderr.decode()[:200]}
        data = open(outp, "rb").read()

    return {"audio_b64": base64.b64encode(data).decode(),
            "voice_seconds": round(duracao, 2),
            "process_seconds": round(time.time() - t0, 1),
            "palavras": palavras}


if __name__ == "__main__":
    if runpod is None:
        raise SystemExit("runpod SDK ausente: instale 'runpod' ou rode handler('input') direto")
    runpod.serverless.start({"handler": handler})
