FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --upgrade pip &&     pip install -r requirements.txt

CMD ["/bin/bash"]
