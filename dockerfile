FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

# Munka könyvtár
WORKDIR /workspace

RUN apt-get update

RUN pip install --no-cache-dir numpy pandas

RUN pip install --no-cache-dir \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip install torch_geometric
RUN pip install mednist

COPY src /workspace/src

# Bash legyen az alapértelmezett
CMD ["python","src/main.py"]