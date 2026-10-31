# soundtrackfree — worker de amostra de narração (VoxCPM2, 48 kHz)
# Roda em Runpod Serverless (scale-to-zero): GPU só existe durante a requisição.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN pip install --no-cache-dir --ignore-installed voxcpm==2.0.3 soundfile librosa runpod

# pesos dentro da imagem: o cold start não depende de download
RUN python -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2', load_denoiser=False, device='cpu')" || \
    python -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2', device='cpu')"

COPY cues /cues
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
