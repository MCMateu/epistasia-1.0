# Epistasia

**Epistasia** is a Python library to analyze **binary landscapes**
(genotype–phenotype or community–function maps) and quantify epistatic
interactions across orders.

📘 Documentation: https://MCMateu.github.io/epistasia-web/  
💻 Source code: https://github.com/MCMateu/epistasia-1.0

---

## Installation

### From GitHub

Recommended while the API is still evolving:

```bash
pip install "epistasia @ git+https://github.com/MCMateu/epistasia-1.0.git"
```

### Editable install for development

```bash
git clone https://github.com/MCMateu/epistasia-1.0.git
cd epistasia-1.0
pip install -e .
```

---

## Using Epistasia with Docker

Docker provides a ready-to-use environment with Epistasia and JupyterLab
installed. This is the recommended option if you want to run analyses without
manually configuring Python environments.

### Requirements

Install Docker Desktop:

- Mac / Windows: https://www.docker.com/products/docker-desktop
- Linux: https://docs.docker.com/engine/install/

### Setup

Create a working folder for your analyses:

```bash
mkdir -p my-analysis/data my-analysis/notebooks
cd my-analysis
```

Use the following structure:

```text
my-analysis/
├── data/        # place your CSV files here
└── notebooks/   # your Jupyter notebooks will be saved here
```

Download the Docker Compose configuration file:

```bash
curl -L https://raw.githubusercontent.com/MCMateu/epistasia-1.0/main/docker-compose.user.yml -o docker-compose.user.yml
```

This will use the image [`sanchezlab/epistasia-1.0`](https://hub.docker.com/r/sanchezlab/epistasia-1.0) from Docker Hub.

Start Epistasia:

```bash
docker compose -f docker-compose.user.yml up
```

The first run will automatically download the Docker image. Later runs should
start much faster.

Open JupyterLab in your browser:

```text
http://localhost:8888
```

To stop the program, press `Ctrl+C` in the terminal.

---

## Input data format

Epistasia expects a CSV file representing a complete binary landscape.

The CSV file should contain:

- First columns: binary state variables, with values `0` or `1`.
- Last columns: numerical measurements or replicates.

Example:

```csv
sp_1,sp_2,sp_3,sp_4,sp_5,rep_1,rep_2,rep_3
0,0,0,0,0,0.000,0.000,0.000
1,0,0,0,0,0.087,0.088,0.093
0,1,0,0,0,0.140,0.132,0.127
1,1,0,0,0,0.208,0.197,0.147
...
```

Rules:

- The number of rows must be exactly `2^N`, where `N` is the number of binary variables.
  For example, 5 variables require 32 rows, 6 variables require 64 rows, and so on.
- Binary columns must appear before measurement columns.
- Multiple replicates are supported.
- A single measurement column is also supported.
- Missing values such as `NaN` are allowed.

---

## Quick start

### Basic Python usage

```python
import epistasia as ep

# Expected columns: binary variables first, measurement columns afterwards
L = ep.landscape_from_file("my_landscape.csv")

print(L)
```

Run a Walsh/epistasis analysis:

```python
out = ep.walsh_analysis(L)

print(out)
```

Plot the variance and amplitude spectrum by interaction order:

```python
ep.plot_variance_and_amplitude(L)
```

### Quick start with Docker

Place your CSV file inside the `data/` folder:

```text
my-analysis/
├── data/
│   └── my_landscape.csv
└── notebooks/
```

Then start Docker:

```bash
docker compose -f docker-compose.user.yml up
```

Open JupyterLab at:

```text
http://localhost:8888
```

JupyterLab may ask for a token the first time you open it. The token is printed in the terminal where Docker is running. Look for a line similar to:

```text
http://127.0.0.1:8888/lab?token=...
```

Copy the value after `token=` and paste it into the JupyterLab login page. 

Create a new notebook and run:

```python
import epistasia as ep

# Files inside my-analysis/data are available inside Docker under /app/data
L = ep.landscape_from_file("/app/data/my_landscape.csv")

print(L)

ep.plot_variance_and_amplitude(L)
```

---

## Status

Epistasia is under active development.
