FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["bash", "-c"]

ENV PATH /opt/conda/envs/ai-parts-recommendation/bin:$PATH

COPY . .

ENV PYTHONPATH=/app:/app/src

EXPOSE 4182

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "4182"]
