# Hotdog or Not Hotdog 🌭🍔

This project solves a **binary image classification** problem by determining whether an image contains a **hotdog** or **not hotdog** using deep learning and transfer learning techniques.

## 📌 Project Overview
- Binary image classification (Hotdog vs Not Hotdog)
- Built with **Convolutional Neural Networks (CNN)**
- **Transfer Learning** used to improve performance and reduce training time
- Modular project structure (training, evaluation, inference separated)

## 🗂️ Project Structure
Hotdog-Not-Hotdog/
│
├── data/
│ ├── raw/ # Raw image dataset
│ └── processed/ # Preprocessed / resized images
│
├── models/
│ └── hotdog_model.keras # Trained model
│
├── notebooks/
│ └── Hotdog_vs_Not_Hotdog_Transfer_Learning.ipynb
│
├── src/
│ ├── data_loader.py # Data loading & preprocessing
│ ├── model.py # Model architecture
│ ├── train.py # Training pipeline
│ ├── evaluate.py # Model evaluation
│ └── predict.py # Inference on new images


## 🧠 Model & Approach
- Pretrained CNN backbone (Transfer Learning)
- Fine-tuned on a custom hotdog / not hotdog dataset
- Binary Cross-Entropy loss
- Adam optimizer
- Model saved and reused for inference

## 📊 Evaluation
- Model performance evaluated on validation data
- Trained model stored in the `models/` directory
- Evaluation logic separated from training for clarity

## 🚀 Inference
You can run predictions on new images using the inference script:

```bash
python src/predict.py --image path_to_image

