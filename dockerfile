FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Munka könyvtár
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pandas

RUN pip install --no-cache-dir \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir \
      pyg_lib \
      torch_scatter \
      torch_sparse \
      -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
	  
	  
RUN pip install torch_geometric
RUN pip install medmnist
RUN pip install faiss-cpu
RUN pip install pyyaml
COPY src /workspace/src

# Bash legyen az alapértelmezett
CMD ["python","src/training-evaluation.py"]