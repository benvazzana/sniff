# sniff
Consensus protocol for a random graph of agents perturbed with communication noise.

## Overview

---

---

## Installation (with uv)

`sniff` uses [uv](https://github.com/astral-sh/uv) — a modern, fast Python package manager and environment builder.  
You **don’t** need to manually activate virtual environments.

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/sniff.git
cd sniff
````

### 2. Install dependencies and sync environment

```bash
uv sync
```

---

## Usage

Run simulations directly with `uv run`.

### Basic example

```bash
uv run scripts/run_protocol.py protocol 10 0.5
```
