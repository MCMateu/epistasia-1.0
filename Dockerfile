FROM python:3.11-slim

LABEL maintainer="j.manuelcm6@gmail.com"
LABEL description="Epistasia: Python toolkit to analyze binary landscapes"
LABEL version="1.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy==1.13.1" \
    "pandas==2.2.2" \
    "matplotlib==3.9.0" \
    "jupyterlab==4.2.4" \
    "cmdstanpy" \
    "networkx"

RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

COPY pyproject.toml README.md ./
COPY epistasia/ ./epistasia/
RUN pip install --no-cache-dir -e .

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/app"]
