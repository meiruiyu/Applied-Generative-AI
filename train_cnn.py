import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from helper_lib.model import AssignmentCNN
from helper_lib.data_loader import get_data_loader


def main():
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device, flush=True)

    train_loader = get_data_loader(train=True)
    test_loader = get_data_loader(train=False)

    model = AssignmentCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    history = []
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_number, (images, labels) in enumerate(
            train_loader, start=1
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if batch_number % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batch_number}/{len(train_loader)} | "
                    f"Loss: {total_loss / total:.4f} | "
                    f"Accuracy: {correct / total:.2%}",
                    flush=True,
                )

        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / total,
            "train_accuracy": correct / total,
            "seconds": time.perf_counter() - start,
        }
        history.append(record)

        # Save trained weights after each completed epoch.
        weights = {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        }
        torch.save(weights, output_dir / "cnn_cifar10.pth")

        print(
            f"Epoch {epoch + 1} completed in "
            f"{record['seconds']:.1f}s | "
            f"Train accuracy: {record['train_accuracy']:.2%}",
            flush=True,
        )

    # Evaluate once on the held-out test set.
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    metrics = {
        "epochs": epochs,
        "batch_size": train_loader.batch_size,
        "learning_rate": 0.001,
        "test_images": total,
        "test_loss": total_loss / total,
        "test_accuracy": correct / total,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(f"\nTest images: {total}", flush=True)
    print(f"Test accuracy: {correct / total:.2%}", flush=True)
    print("Saved: models/cnn_cifar10.pth", flush=True)
    print("Saved: models/metrics.json", flush=True)


if __name__ == "__main__":
    main()