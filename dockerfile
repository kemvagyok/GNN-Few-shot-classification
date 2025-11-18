FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

# Munka könyvtár
WORKDIR /workspace

RUN apt-get update

RUN pip install --no-cache-dir \
    numpy pandas \
    torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

RUN pip install pytorch_geometric

RUN pip install --no-cache-dir pyg_lib torch_scatter torch_sparse \
    torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

COPY src /workspace/

# Bash legyen az alapértelmezett
CMD ["python","src/main.py"]