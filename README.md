# sps_genai

FastAPI project for Applied Generative AI (Columbia SPS).

- **Module 3 class activity:** bigram text generation (`/generate`)
- **Assignment 1:** word embeddings with spaCy `en_core_web_lg` (`/embedding`, `/similarity`)

## Project structure

```
sps_genai/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app and endpoints
│   ├── bigram_model.py      # Bigram model (from Module 2 Practical 2)
│   └── embedding_model.py   # Word embeddings (from Module 2 Practical 3)
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

The first build downloads the spaCy `en_core_web_lg` model (about 400 MB), so it can take a few minutes.

Open the interactive API docs at http://127.0.0.1:8000/docs

## Run locally with uv (without Docker)

```bash
uv sync
uv run fastapi dev app/main.py
```

## API endpoints

### `GET /`

Health check.

```bash
curl http://127.0.0.1:8000/
```

```json
{"Hello": "World"}
```

### `POST /generate`

Generates text from the bigram model, starting from `start_word`.

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 10}'
```

```json
{"generated_text": "the story of monte cristo is falsely imprisoned and later"}
```

Output is random, so it changes on each call.

### `POST /embedding`

Returns the 300-dimensional word embedding for the query word.

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'
```

```json
{"word": "apple", "embedding": [-0.36391, 0.43771, -0.20447, "... 300 values in total"]}
```

### `POST /similarity`

Returns the cosine similarity between two words (or two sentences).

```bash
curl -X POST http://127.0.0.1:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"word1": "apple", "word2": "car"}'
```

```json
{"word1": "apple", "word2": "car", "similarity": 0.2174709439277649}
```
