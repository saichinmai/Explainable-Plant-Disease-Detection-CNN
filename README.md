# 🌿 Explainable Plant Disease Detection Using Convolutional Neural Networks (CNNs)

## 1. Project Overview

Plant diseases can reduce agricultural productivity, making early identification important. This project develops an explainable plant disease detection system using Convolutional Neural Networks (CNNs) to classify plant leaf images into healthy and diseased categories.

Grad-CAM (Gradient-weighted Class Activation Mapping) is integrated to generate visual heatmaps highlighting the image regions that influence the model's predictions. A Streamlit-based web application allows users to upload leaf images and view the predicted category, confidence score and visual explanation.

The project demonstrates the practical application of Deep Learning and Explainable Artificial Intelligence (XAI) in agriculture.

## 2. Objectives

- Develop a CNN-based model for plant disease classification.
- Classify plant leaf images into healthy and diseased categories.
- Apply Grad-CAM to explain the model's predictions.
- Develop an interactive web application using Streamlit.
- Evaluate model performance using accuracy, precision, recall, F1-score and a confusion matrix.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow / Keras | CNN model development and training |
| CNN | Plant disease image classification |
| NumPy | Numerical operations |
| OpenCV | Image processing and heatmap generation |
| Matplotlib | Graphs and visualizations |
| Scikit-learn | Model evaluation and classification metrics |
| Streamlit | Web application development |
| Google Colab | Model training and experimentation |
| Visual Studio Code | Application development |
| Git and GitHub | Version control and project hosting |

## 4. Dataset Details

The project uses the PlantVillage plant disease image dataset.

The implementation uses 10 selected classes representing three plant types:

### Pepper
1. Pepper bell Bacterial spot
2. Pepper bell healthy

### Potato
3. Potato Early blight
4. Potato healthy

### Tomato
5. Tomato Early blight
6. Tomato Late blight
7. Tomato Leaf Mold
8. Tomato Yellow Leaf Curl Virus
9. Tomato mosaic virus
10. Tomato healthy

### Dataset Split

The dataset was divided into three subsets:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The notebook performs dataset preparation, image preprocessing and splitting.

## 5. Methodology

The project follows these major steps:

1. Dataset collection – Selected plant leaf images from PlantVillage.
2. Data preprocessing – Resized images to 128 × 128 pixels and normalized pixel values.
3. Dataset splitting – Divided the images into training, validation and testing sets.
4. CNN model development – Built a CNN architecture using convolutional and pooling layers.
5. Model training – Trained the model using the training dataset and monitored validation performance.
6. Model evaluation – Evaluated the trained model using the test dataset.
7. Grad-CAM implementation – Generated heatmaps to visualize important image regions.
8. Web application development – Integrated the trained model into a Streamlit application.

## 6. CNN Architecture

The CNN model contains the following layers:

- Input layer: 128 × 128 × 3
- Random horizontal flipping
- Random rotation
- Convolutional layer: 32 filters
- Max pooling layer
- Convolutional layer: 64 filters
- Max pooling layer
- Convolutional layer: 128 filters
- Max pooling layer
- Flatten layer
- Dense layer: 128 neurons
- Dropout layer: 0.5
- Output layer: 10 classes with Softmax activation

**Model configuration:**
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Evaluation metric: Accuracy
- Training epochs: Up to 20
- Batch size: 32
- Image size: 128 × 128

## 7. Model Evaluation Results

The trained model was evaluated using the test dataset.

### Overall Results

| Metric | Result |
|---|---:|
| Test Accuracy | 95.64% |
| Test Loss | 0.1231 |
| Macro Precision | 0.94 |
| Macro Recall | 0.96 |
| Macro F1-score | 0.95 |
| Weighted F1-score | 0.96 |

### Evaluation Techniques

- Accuracy measures the overall proportion of correct predictions.
- Precision measures how many predicted class labels were correct.
- Recall measures how many actual class samples were correctly identified.
- F1-score combines precision and recall.
- Confusion matrix provides a class-wise comparison of actual and predicted labels.

## 8. Explainable AI Using Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to explain the CNN model's predictions.

It generates a heatmap showing the image regions that influenced the predicted class.

### Grad-CAM Visualization

The application displays:

- Original uploaded leaf image.
- Grad-CAM heatmap.
- Heatmap overlay on the original image.

The highlighted regions help users understand which parts of the image influenced the model's prediction.

**Note:** Grad-CAM provides a visual explanation of model behaviour. It does not prove the biological cause of a disease or guarantee a correct prediction.

## 9. Streamlit Web Application

The project includes an interactive web application developed using Streamlit.

### Main Features

- Upload plant leaf images in JPG, JPEG or PNG format.
- Predict the plant disease category.
- Display prediction confidence.
- Display probabilities for all 10 classes.
- Generate Grad-CAM heatmaps.
- Display the Grad-CAM overlay.

## 10. Project Folder Structure

```text
Explainable-Plant-Disease-Detection-CNN/
│
├── EDL_PROJECT.ipynb
├── app.py
├── plant_disease_model.keras
├── class_names.json
├── requirements.txt
└── README.md
```

## 11. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/saichinmai/Explainable-Plant-Disease-Detection-CNN.git
```

### Step 2: Open the Project Folder

```bash
cd Explainable-Plant-Disease-Detection-CNN
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

### Step 5: Open the Application

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

Upload a plant leaf image and click **Analyze Plant Image** to view the prediction and Grad-CAM explanation.

## 12. Limitations

- The model is trained on selected PlantVillage dataset classes and may not generalize equally well to all real-world field conditions.
- Dataset class imbalance may affect individual class performance.
- Some classes have fewer test images, so their individual evaluation results should be interpreted carefully.
- Image quality, lighting and background differences may influence predictions.
- Grad-CAM is an interpretation aid, not a definitive plant disease diagnosis.

## 13. Future Enhancements

- Add more plant species and disease categories.
- Improve performance on real-world field images.
- Develop a mobile-friendly application.
- Explore transfer learning with pretrained CNN architectures.
- Add disease prevention and treatment information from verified agricultural sources.

## 14. Conclusion

This project demonstrates the use of Convolutional Neural Networks for plant disease classification and Grad-CAM for explainable predictions. The trained model achieved 95.64% test accuracy on the selected test dataset. The Streamlit application provides an interactive interface for image upload, disease prediction and visual explanation.

The project combines Deep Learning, Explainable AI and web application development to demonstrate a practical approach to plant disease detection.

## 15. Repository

**GitHub:** https://github.com/saichinmai/Explainable-Plant-Disease-Detection-CNN

---

**Project Domain:** Deep Learning and Explainable Artificial Intelligence

**Project Title:** Explainable Plant Disease Detection Using Convolutional Neural Networks (CNNs)