# sps_genai

FastAPI project for Applied Generative AI (Columbia SPS).

- **Module 3 class activity:** bigram text generation (`/generate`)
- **Assignment 1:** word embeddings with spaCy `en_core_web_lg` (`/embedding`, `/similarity`)
- **Assignment 2:** CIFAR-10 image classifier using the CNN64 model from `helper_lib` (`/classify`)
- **Assignment 3:** GAN that generates hand-written MNIST digits, using the MNIST_GAN model from `helper_lib` (`/generate_digits`)

## Project structure

```
sps_genai/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app and endpoints
│   ├── bigram_model.py      # Bigram model (from Module 2 Practical 2)
│   ├── embedding_model.py   # Word embeddings (from Module 2 Practical 3)
│   ├── classifier.py        # CIFAR-10 classifier (Assignment 2)
│   ├── digit_generator.py   # MNIST digit generator (Assignment 3)
│   └── models/
│       ├── cnn64_cifar10.pth  # Trained CNN64 weights
│       └── gan_mnist.pth      # Trained MNIST GAN weights
├── helper_lib/              # Reusable neural network code (Modules 4, 5, 6 and 8: CNNs, VAE, GAN, diffusion)
├── gan_example.py           # Trains the WGAN on CelebA faces (Module 6 activity)
├── diffusion_example.py     # Trains the diffusion model on Oxford Flowers 102 (Module 8 activity)
├── train_cifar10_cnn64.py   # Trains CNN64 on CIFAR-10 (Assignment 2)
├── train_mnist_gan.py       # Trains the MNIST GAN (Assignment 3)
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

### `POST /classify`

Classifies an uploaded image into one of the 10 CIFAR-10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Any image size works: it is resized to 64x64 before going into the CNN.

```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@my_image.jpg"
```

Example response (the probabilities depend on the image):

```json
{
  "filename": "my_image.jpg",
  "predicted_class": "cat",
  "top_predictions": [
    {"class": "cat", "probability": 0.71},
    {"class": "dog", "probability": 0.18},
    {"class": "frog", "probability": 0.04}
  ]
}
```

In the `/docs` page, open `POST /classify`, click **Try it out**, and choose an image file.

### `GET /generate_digits`

Generates new hand-written digits with the MNIST GAN (Assignment 3) and returns them as a PNG grid.

| Query parameter | Default | Meaning |
|---|---|---|
| `num_images` | 16 | How many digits to generate (1 to 64) |
| `seed` | none | Optional random seed; the same seed always gives the same digits |

```bash
curl "http://127.0.0.1:8000/generate_digits?num_images=16&seed=7" --output digits.png
```

You can also open http://127.0.0.1:8000/generate_digits?num_images=16 directly in a browser, or use **Try it out** in `/docs`.

## Assignment 2: CNN64 model and training

`helper_lib/model.py` defines `CNN64`, which matches the required architecture:

| Layer | Output shape |
|---|---|
| Input (RGB) | 3 x 64 x 64 |
| Conv2D 16 filters, 3x3, stride 1, padding 1 + ReLU | 16 x 64 x 64 |
| MaxPool 2x2, stride 2 | 16 x 32 x 32 |
| Conv2D 32 filters, 3x3, stride 1, padding 1 + ReLU | 32 x 32 x 32 |
| MaxPool 2x2, stride 2 | 32 x 16 x 16 |
| Flatten | 8192 |
| Fully connected 100 units + ReLU | 100 |
| Fully connected 10 units | 10 |

CIFAR-10 images are 32x32, so they are resized to 64x64 to match this architecture. To retrain the model (the trained weights are already included in `app/models/`):

```bash
uv run python train_cifar10_cnn64.py
```

## Assignment 3: MNIST GAN and training

`helper_lib/model.py` defines `MNISTGenerator` and `MNISTDiscriminator` (wrapped together as `MNISTGAN`, available through `get_model("MNIST_GAN")`), matching the required architecture:

**Generator**

| Layer | Output shape |
|---|---|
| Input: noise vector | 100 |
| Fully connected to 7×7×128, then reshape | 128 × 7 × 7 |
| ConvTranspose2D 128 → 64, kernel 4, stride 2, padding 1 + BatchNorm2D + ReLU | 64 × 14 × 14 |
| ConvTranspose2D 64 → 1, kernel 4, stride 2, padding 1 + Tanh | 1 × 28 × 28 |

**Discriminator**

| Layer | Output shape |
|---|---|
| Input: image | 1 × 28 × 28 |
| Conv2D 1 → 64, kernel 4, stride 2, padding 1 + LeakyReLU(0.2) | 64 × 14 × 14 |
| Conv2D 64 → 128, kernel 4, stride 2, padding 1 + BatchNorm2D + LeakyReLU(0.2) | 128 × 7 × 7 |
| Flatten + Linear | 1 (real/fake score; sigmoid gives the probability) |

The GAN is trained on MNIST with binary cross entropy and Adam (`helper_lib/trainer.py`, `train_mnist_gan`). To retrain it (the trained weights are already included in `app/models/`):

```bash
uv run python train_mnist_gan.py
```

## Module 8 activity: diffusion model in `helper_lib`

From the Module 8 Practical 2 (Diffusion Methods) notebook:

- `helper_lib/model.py`: `get_model("Diffusion")` returns a `DiffusionModel` that wraps a UNet (residual, down and up blocks with a sinusoidal noise embedding) for 64x64 color images, using the offset cosine diffusion schedule and an EMA copy of the UNet for generation.
- `helper_lib/trainer.py`: `train_diffusion(model, data_loader, criterion, optimizer, device, epochs)` trains the UNet to predict the noise added to each image and returns the model.
- `helper_lib/generator.py`: `generate_samples(model, device, num_samples=10, diffusion_steps=100)` starts from standard normal noise, runs the reverse diffusion, and plots the images on a grid.
- `helper_lib/data_loader.py`: `dataset="Flowers102"` loads Oxford Flowers 102 at 64x64.

```bash
uv run python diffusion_example.py
```

Sample grids are saved to `checkpoints/diffusion_samples/` after every epoch.
