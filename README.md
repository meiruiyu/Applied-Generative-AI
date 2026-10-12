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