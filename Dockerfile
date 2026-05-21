FROM python:3.11-slim

LABEL maintainer="j.manuelcm6@gmail.com"
LABEL description="Epistasia: Python toolkit to analyze binary landscapes"
LABEL version="0.2.0"

WORKDIR /app

# Install system dependencies needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with pinned versions for reproducibility
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy==1.13.1" \
    "pandas==2.2.2" \
    "matplotlib==3.9.0" \
    "jupyterlab==4.2.4" \
    "cmdstanpy" \
    "networkx"

# Install CmdStan (required by cmdstanpy)
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# Copy and install the epistasia package
COPY pyproject.toml README.md ./
COPY epistasia/ ./epistasia/
RUN pip install --no-cache-dir -e .

# Copy tutorial notebooks and style
COPY epistasia_web/docs/notebooks/ ./notebooks/

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--notebook-dir=/app/notebooks"]
