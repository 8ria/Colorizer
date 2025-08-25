# Colorizer

**Colorizer** is a web service that maps natural language text to colors using sentence embeddings. It leverages a pre-trained **ONNX model** (`distiluse-base-multilingual-cased-v2`) to generate embeddings, compares them to a set of reference embeddings, and returns the closest color as JSON.

This project demonstrates how to combine **ONNX models**, **Rust**, and **Actix Web** for a lightweight AI-powered API.

---

## Features

* Input any text in multiple languages (thanks to `distiluse-base-multilingual-cased-v2`)
* Returns the closest reference color as RGB (`r`, `g`, `b`)
* Pre-computed reference embeddings for fast lookup
* Rate-limited API with `actix-governor`
* Serves static frontend files under `/static`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/8ria/Colorizer.git
cd Colorizer
```

### 2. Install Rust

Make sure you have Rust 1.70+ installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 3. Download models

> **Note:** The `models/` folder is **not included** in this repository due to size and licensing. You must download the following files and place them in `models/`:

* `tokenizer.json` — tokenizer for `distiluse-base-multilingual-cased-v2`
* `model.onnx` — ONNX model converted from `distiluse-base-multilingual-cased-v2`

---

## Usage

### 1. Generate reference embeddings

You can generate embeddings for predefined words (or customize your own):

```bash
cargo run --bin generate_ref_embeddings
```

This will create `custom/ref_embeddings.json`.

### 2. Run the server

```bash
cargo run
```

The server will start at `http://localhost:8090/`.

---

## API

### POST `/color`

**Request:**

```json
{
  "text": "sun"
}
```

**Response:**

```json
{
  "r": 255,
  "g": 255,
  "b": 0
}
```

---

### GET `/`

Serves `static/index.html` if available. Useful for a simple frontend.

---

## Project Structure

```
Colorizer/
├─ src/
│  ├─ main.rs               # Actix server
│  ├─ bin/
│  │  └─ generate_ref_embeddings.rs  # Embedding generator
├─ models/                  # ONNX model + tokenizer
├─ custom/                  # Generated reference embeddings
├─ static/                  # Optional static frontend
├─ Cargo.toml
└─ README.md
```

---

## Dependencies

* `actix-web` — web framework
* `actix-files` — static file serving
* `actix-governor` — rate limiting
* `ort` — ONNX Runtime for Rust
* `ndarray` — numerical arrays
* `serde` + `serde_json` — JSON serialization
* `tokenizers` — HuggingFace tokenizers

---

## Notes

* The system uses **cosine similarity** to match input embeddings to reference colors.
* You can easily extend `custom/ref_embeddings.json` with more words/colors.
* For production deployment, consider HTTPS, caching, and scaling options.

---

## License

MIT © AndriaK
