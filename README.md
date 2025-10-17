# Handwritten Character Recognition using CNN

This repository contains a deep learning project for **handwritten character recognition** using a **Convolutional Neural Network (CNN)**. The project utilizes the [OCR-DNN Ensemble Dataset](https://www.kaggle.com/datasets/viratsrivastava/ocr-dnn-ensemble-dataset/versions/2) from Kaggle, which comprises a diverse set of handwritten characters suitable for training robust OCR models.

## 📝 Project Overview

Handwritten character recognition is a pivotal task in computer vision, enabling machines to interpret handwritten text. This project employs a CNN to classify images of handwritten characters into their respective labels. The model is trained on the OCR-DNN Ensemble Dataset, which includes a variety of handwritten characters to enhance the model's generalization capabilities.

Key features include:

* Image preprocessing and normalization.
* CNN architecture with multiple convolutional and pooling layers.
* Model training, validation, and evaluation.
* Prediction of unseen handwritten characters.

## 🛠 Technologies Used

* **Python 3.x**
* **TensorFlow / Keras**
* **NumPy, Pandas**
* **Matplotlib / Seaborn** for visualization

## 📂 Project Structure

```
Handwritten-Character-Recognition/
│
├── data/                  # Dataset files (training/testing)
├── notebooks/             # Jupyter notebooks for experiments
├── src/                   # Source code (CNN model, preprocessing scripts)
├── models/                # Trained models
├── requirements.txt       # Required Python packages
└── README.md              # Project documentation
```

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/handwritten-character-recognition.git
cd handwritten-character-recognition
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the OCR-DNN Ensemble Dataset from Kaggle and place it in the `data/` directory.

4. Run the training script:

```bash
python src/train_cnn.py
```

5. Predict characters using the trained model:

```bash
python src/predict.py --image path_to_image
```

## 📈 Model Performance

* High accuracy on the OCR-DNN Ensemble Dataset.
* CNN architecture optimized for feature extraction and generalization.
* Can be extended to recognize digits, uppercase and lowercase letters.

## 🔗 References

* [OCR-DNN Ensemble Dataset on Kaggle](https://www.kaggle.com/datasets/viratsrivastava/ocr-dnn-ensemble-dataset/versions/2)
* [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn)

## 🧑‍💻 Author

**Nada Ahmed** – AI Developer | Machine Learning Enthusiast


