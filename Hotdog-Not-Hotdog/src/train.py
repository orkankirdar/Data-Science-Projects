from data_loader import load_data
from model import build_model

def train(data_dir, epochs=15):
    train_ds, val_ds = load_data(data_dir)

    model = build_model()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save("models/hotdog_model.keras")

if __name__ == "__main__":
    train("data/raw/train")
