FROM python:3

RUN pip install -U pip \
    && pip install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl \
    && rm -rf ~/.cache/pip

COPY main.py .

ENTRYPOINT [ "python", "main.py" ]
