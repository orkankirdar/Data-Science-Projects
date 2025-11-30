import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import load_test_data

def evaluate_model():
    test_dir = "data/raw/test"
    test_ds = load_test_data(test_dir)

    model = keras.models.load_model("models/hotdog_model.keras")

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

    y_pred_scores = model.predict(test_ds) 
    
    y_pred = (y_pred_scores > 0.5).astype("int32").reshape(-1)

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix (rows = gerçek, cols = tahmin):")
    print(cm)

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["hotdog", "not_hotdog"] 
    ))


if __name__ == "__main__":
    evaluate_model()