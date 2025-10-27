# sniff
[![Build](https://github.com/iqsnider/sniff/actions/workflows/pytest.yml/badge.svg)](https://github.com/iqsnider/sniff/actions/workflows/pytest.yml)

Consensus protocol for a random graph of agents perturbed with communication noise.

---

## Overview

## Installation (with uv)

`sniff` uses [uv](https://github.com/astral-sh/uv), a fast Python package manager and environment builder.  
You **don’t** need to manually activate virtual environments.

### 1. Clone the repository
```bash
git clone git@github.com:iqsnider/sniff.git
cd sniff
````

### 2. Install dependencies and sync environment

```bash
uv sync
```

---

## Usage

Run simulations directly with `uv run`. All CLI commands have default values.

### Basic example

```bash
uv run sniff protocol-1d
```

### Running simple 2D setpoint tracking
```bash
uv run sniff protocol-2d --p-track 0.0 0.0
```

### Running grid formation consensus
```bash
uv run sniff formation --n 20 --spacing 0.1
