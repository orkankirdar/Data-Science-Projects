from src.data_loader import load_data
from src.model import build_model

def train(data_dir, epochs=5):
    train_ds = load_data(data_dir)
    val_ds = load_data(data_dir.replace("train", "val"))
    
    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("models/hotdog_model.keras")

if __name__ == "__main__":
    train("data/train")

