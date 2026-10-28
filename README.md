# Assignment 1: FastAPI and Word Embeddings

This project extends the Module 3 bigram text generation API with a spaCy word embedding endpoint. It uses the `en_core_web_lg` model.

## Run with Docker

From the project root, with Docker Desktop running:

```bash
docker build -t sps-genai .
docker run --rm -p 8000:80 sps-genai
```

The first command builds the image and installs the Python dependencies and spaCy model. The second command starts the API. The first build may take several minutes.

Open http://127.0.0.1:8000/docs to test the endpoints.

## API endpoints

- `GET /embedding?word=apple`: returns the queried word, its 300-dimensional embedding vector, and the vector dimensions.
- `POST /generate`: generates text from the bigram model. Example request body:

```json
{"start_word": "the", "length": 10}
```

## Run locally without Docker

With Python and uv installed, run these commands from the project root:

```bash
uv sync
uv run fastapi dev app/main.py
```

Then open http://127.0.0.1:8000/docs.



## Assignment 2: CNN Image Classification

This assignment adds a trained CIFAR-10 image classifier to the existing FastAPI application.

### Model architecture

- Input: 64×64 RGB image.
- Conv2d: 3→16 channels, kernel 3×3, stride 1, padding 1.
- ReLU, followed by max pooling with kernel 2×2 and stride 2.
- Conv2d: 16→32 channels, kernel 3×3, stride 1, padding 1.
- ReLU, followed by max pooling with kernel 2×2 and stride 2.
- Flatten: 8192 features.
- Fully connected layer: 8192→100, followed by ReLU.
- Fully connected layer: 100→10 output logits.

### Training and evaluation

CIFAR-10 images are resized to 64×64 and converted to tensors.
The same preprocessing is used for inference.

- Training images: 50,000.
- Test images: 10,000.
- Epochs: 10.
- Batch size: 64.
- Optimizer: Adam, learning rate 0.001.
- Loss: CrossEntropyLoss.
- Final training accuracy: 81.19%.
- Test accuracy: 65.95%.

To retrain, place the extracted CIFAR-10 files in
`data/cifar-10-batches-py/`, then run:

```bash
uv sync --frozen
uv run python train_cnn.py
```

Trained weights are included at `models/cnn_cifar10.pth`.
Metrics are stored at `models/metrics.json`.
Retraining is not required to run the API.

### Docker deployment

With Docker Desktop running, execute from the project root:

```bash
docker build -t sps-genai .
docker run --rm --name assignment2-cnn -p 8000:80 sps-genai
```

Open http://127.0.0.1:8000/docs.

The first build downloads dependencies and may take several minutes.

### Image classification API

Endpoint: `POST /classify-image`

Upload a PNG or JPEG using the multipart field `file`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/classify-image \
  -F "file=@sample_images/test_00_cat.png"
```

Example response:

```json
{
  "class_id": 3,
  "class_name": "cat",
  "confidence": 0.7646
}
```

The confidence field is the predicted class's softmax score,
not a guarantee that the prediction is correct.

Supported classes: airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, and truck.

The model always predicts one of these ten classes; it does
not detect unknown categories.

The existing `/generate` and `/embedding` endpoints are retained.




## Assignment 3: MNIST GAN Image Generation

This assignment implements the required GAN architecture in PyTorch,
trains it on MNIST, and adds image generation to the existing FastAPI
application.

The `POST /generate-digits` endpoint and its request handler are defined
directly in `app/main.py`, using the existing FastAPI `app`.

### Project files

- `helper_lib/data_loader.py`: MNIST loading and preprocessing.
- `helper_lib/model.py`: Generator, Discriminator, GAN, and `get_model()`.
- `helper_lib/trainer.py`: GAN training with separate optimizers.
- `helper_lib/generator.py`: sample generation and image grids.
- `helper_lib/utils.py`: model weight saving.
- `train_gan.py`: training entry point and checkpoint saving.
- `app/main.py`: GAN inference endpoint.
- `models/gan_generator.pth`: final trained Generator weights.
- `sample_images/gan/`: sample grids saved during training.

### Required architecture

Generator:

1. Input noise: `[B, 100]`.
2. Linear: 100 -> 128 * 7 * 7, then reshape to `[B, 128, 7, 7]`.
3. ConvTranspose2d: 128 -> 64, kernel 4, stride 2, padding 1.
4. BatchNorm2d(64), followed by ReLU.
5. ConvTranspose2d: 64 -> 1, kernel 4, stride 2, padding 1.
6. Tanh, producing images of shape `[B, 1, 28, 28]`.

Discriminator:

1. Input image: `[B, 1, 28, 28]`.
2. Conv2d: 1 -> 64, kernel 4, stride 2, padding 1.
3. LeakyReLU(0.2).
4. Conv2d: 64 -> 128, kernel 4, stride 2, padding 1.
5. BatchNorm2d(128), followed by LeakyReLU(0.2).
6. Flatten and Linear: 6272 -> 1.
7. Sigmoid to obtain a real/fake probability.

### Dataset and preprocessing

The training data comes explicitly from `torchvision.datasets.MNIST`,
called inside `get_mnist_data_loader()` in
`helper_lib/data_loader.py`.

- Training split: 60,000 images.
- Image format: 28 x 28 grayscale.
- `ToTensor()` converts pixel values to [0, 1].
- `Normalize((0.5,), (0.5,))` converts them to [-1, 1],
  matching the Generator's Tanh output.
- MNIST is downloaded automatically to `data/` if missing.

This is an unconditional GAN. Training uses the images without their
digit labels; the API cannot request a specific digit.

### Training

The final model was trained using:

- Epochs: 5.
- Batch size: 64.
- Noise dimension: 100.
- Loss: BCELoss.
- Separate Adam optimizers for Generator and Discriminator.
- Learning rate: 0.0002 for both optimizers.
- Adam betas: (0.5, 0.999).
- Random seed: 42.
- Training device: CPU.

For each batch, the Discriminator learns from real MNIST images and
detached generated images. The Generator then learns to make generated
images receive a real label from the Discriminator.

The Discriminator remains in training mode during both updates so its
BatchNorm behavior is consistent. Its parameters are frozen during the
Generator update. Evaluation mode is used when generating saved samples
and serving API inference.

To run a small training check:

```bash
uv sync --frozen
uv run python -u train_gan.py --smoke-test --epochs 1
```

To reproduce the final training configuration:

```bash
uv run python -u train_gan.py --epochs 5
```

Each invocation starts training from scratch. A full training run
overwrites the default GAN weights, checkpoint, and sample grids.
Exact results may vary across platforms and dependency versions.

The final inference weights are included at:

```text
models/gan_generator.pth
```

Retraining and downloading MNIST are not required to run inference.

### Local API

From the project root:

```bash
uv sync --frozen
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs.

The existing `/generate`, `/embedding`, and `/classify-image`
endpoints are retained.

### GAN image-generation endpoint

**Endpoint:** `POST /generate-digits`

Example JSON request:

```json
{
  "num_samples": 40,
  "seed": 42
}
```

Parameters:

- `num_samples`: integer from 1 to 100; default is 10.
- `seed`: optional integer from 0 to 4294967295. Supplying the same
  seed reproduces the noise input.

The endpoint loads `models/gan_generator.pth`, runs the Generator in
evaluation mode on CPU, converts output pixels from [-1, 1] to [0, 1],
and returns a PNG grid.

Example call:

```bash
curl --fail-with-body -X POST \
  "http://127.0.0.1:8000/generate-digits" \
  -H "Content-Type: application/json" \
  -d '{"num_samples":40,"seed":42}' \
  --output generated_digits.png
```

Responses:

- `200`: generated PNG image, with content type `image/png`.
- `422`: invalid request, such as `num_samples` equal to 0.
- `503`: Generator weights are missing.

Some generated digits remain ambiguous or distorted.

### Docker deployment

With Docker Desktop running, execute from the project root:

```bash
docker build -t sps-genai-a3 .
docker run --rm --name assignment3-gan -p 8000:80 sps-genai-a3
```

Open http://127.0.0.1:8000/docs.

The host's port 8000 maps to port 80 inside the container.
The image includes the application code, helper library, dependencies,
and model weights.

To save an image returned by the container:

```bash
curl --fail-with-body -X POST \
  "http://127.0.0.1:8000/generate-digits" \
  -H "Content-Type: application/json" \
  -d '{"num_samples":40,"seed":42}' \
  --output gan_docker_result.png
```

### Validation performed

- Generator forward pass returned shape `[2, 1, 28, 28]`.
- Discriminator forward pass returned shape `[2, 1]`.
- MNIST loading returned 60,000 training images.
- A training batch had shape `[64, 1, 28, 28]` and pixel range [-1, 1].
- A small training run completed and saved weights and samples.
- The final 5-epoch run generated recognizable, varied digit shapes.
- The local API returned HTTP 200 and a PNG grid for 40 samples.
- A request with `num_samples: 0` returned HTTP 422.
- The Docker image built successfully and the container started.
- The container API returned HTTP 200 and a PNG digit grid.