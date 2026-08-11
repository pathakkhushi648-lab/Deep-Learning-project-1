# Breast Cancer Wisconsin(  Deep Learning)

## Project Overview

This project implements an end-to-end **Deep Learning classification pipeline** for predicting whether a breast tumor is **Malignant** or **Benign** using the **Breast Cancer Wisconsin (Diagnostic) dataset**.

The project focuses on building, evaluating, and comparing multiple neural-network architectures and training strategies. It demonstrates how preprocessing, activation functions, Early Stopping, Dropout, and regularization techniques can influence model performance and generalization.

The complete workflow includes data exploration, preprocessing, neural-network development, model evaluation, performance comparison, classification threshold analysis, and model interpretation.

---

## Objectives

The main objectives of this project are:

* Perform Exploratory Data Analysis (EDA) on the breast cancer dataset.
* Identify missing values, duplicate records, and class distribution.
* Analyze relationships between numerical features.
* Prepare the dataset for Deep Learning.
* Apply feature scaling using `StandardScaler`.
* Develop a Single-Layer Perceptron (SLP) as a baseline model.
* Develop Multi-Layer Perceptron (MLP) models.
* Compare ReLU, Tanh, and Sigmoid activation functions.
* Investigate the effect of Early Stopping.
* Compare different Dropout rates.
* Evaluate L1, L2, and combined L1-L2 regularization.
* Develop a final combined neural-network model.
* Evaluate models using Accuracy, Precision, Recall, and F1-Score.
* Analyze classification thresholds.
* Compare model performance and generalization.

---

## Dataset

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

The dataset contains diagnostic measurements computed from digitized images of breast mass samples.

The project uses **30 numerical features** related to characteristics of cell nuclei.

The target variable represents:

* `0` → Malignant
* `1` → Benign

The notebook performs the following dataset preparation steps:

1. Load the CSV dataset.
2. Inspect dataset shape and structure.
3. Check duplicate records.
4. Check missing values.
5. Generate descriptive statistics.
6. Rename the diagnosis column.
7. Encode the target variable.
8. Separate features and target.
9. Perform correlation analysis.

---

## Features

The model uses 30 numerical diagnostic features, including measurements related to:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

These measurements are available in three forms:

* Mean
* Standard Error
* Worst

---

## Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Missing Value & Duplicate Check
   ↓
Exploratory Data Analysis
   ↓
Target Encoding
   ↓
Feature Selection
   ↓
Correlation Analysis
   ↓
Stratified Train-Test Split
   ↓
StandardScaler
   ↓
Single-Layer Perceptron
   ↓
Multi-Layer Perceptron
   ↓
Activation Function Comparison
   ↓
Early Stopping
   ↓
Dropout Experiments
   ↓
L1 / L2 / L1-L2 Regularization
   ↓
Final Combined Model
   ↓
Model Evaluation
   ↓
Threshold Analysis
   ↓
Model Comparison
   ↓
Conclusion
```

---

## Exploratory Data Analysis

The project performs several EDA steps to understand the dataset before model training.

### Dataset Inspection

* Dataset shape
* Data types
* Statistical summary
* Duplicate records
* Missing values

### Visualizations

The notebook includes visualizations such as:

* Missing Values Heatmap
* Target Class Distribution
* Feature Correlation Heatmap
* Training vs Validation Loss
* Training vs Validation Accuracy
* Activation Function Comparison
* Confusion Matrices
* Model Accuracy Comparison
* Classification Threshold Analysis

---

## Data Preprocessing

### Target Encoding

The original diagnosis labels are converted into binary numerical values:

```text
Malignant → 0
Benign    → 1
```

### Train-Test Split

The dataset is divided into:

```text
Training Data → 80%
Testing Data  → 20%
```

A **stratified split** is used to maintain the class distribution in both training and testing datasets.

### Feature Scaling

`StandardScaler` is applied to the numerical features.

The scaler is fitted only on the training data and then applied to the testing data to prevent data leakage.

---

# Deep Learning Models

## 1. Single-Layer Perceptron

The SLP is used as the baseline neural-network model.

Architecture:

```text
30 Input Features
       ↓
Sigmoid Output
       ↓
Binary Classification
```

The model uses:

* Adam optimizer
* Binary Cross-Entropy loss
* Sigmoid activation
* 50 training epochs
* Batch size of 32

The SLP provides a baseline for comparing more complex neural-network architectures.

---

## 2. Multi-Layer Perceptron

A deeper MLP architecture is developed to learn nonlinear relationships.

Architecture:

```text
30
 ↓
64 Neurons
 ↓
32 Neurons
 ↓
1 Output
```

Hidden layers use nonlinear activation functions, while the output layer uses Sigmoid activation for binary classification.

---

## 3. Activation Function Comparison

Three activation functions are evaluated:

* ReLU
* Tanh
* Sigmoid

The models are compared using validation performance.

This experiment demonstrates how the choice of activation function can influence neural-network learning and generalization.

---

## 4. Early Stopping

Early Stopping is introduced to reduce overfitting.

The model monitors validation loss and stops training when the validation performance no longer improves.

The best model weights are restored after training.

---

## 5. Dropout Experiment

Different Dropout rates are evaluated:

```text
0.1
0.3
0.5
```

Dropout randomly disables a proportion of neurons during training, helping reduce overfitting and improve generalization.

---

## 6. Regularization

The project evaluates different regularization strategies:

### L1 Regularization

Encourages sparse model parameters.

### L2 Regularization

Penalizes large model weights and helps control model complexity.

### Combined L1-L2 Regularization

Combines both regularization approaches.

---

# Final Model

The final model combines multiple techniques designed to improve generalization.

Architecture:

```text
Input: 30 Features
        ↓
Dense: 128 Neurons
ReLU
L2 Regularization
        ↓
Dropout: 0.3
        ↓
Dense: 64 Neurons
ReLU
L2 Regularization
        ↓
Dropout: 0.3
        ↓
Dense: 1 Neuron
Sigmoid
        ↓
Prediction
```

### Training Configuration

* Optimizer: Adam
* Loss: Binary Cross-Entropy
* Batch Size: 32
* Maximum Epochs: 300
* Validation Split: 10%
* Early Stopping: Enabled
* L2 Regularization: `0.001`
* Dropout: `0.3`

---

# Model Evaluation

The models are evaluated using multiple classification metrics.

### Accuracy

Measures the overall proportion of correctly classified observations.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual positive cases are correctly identified.

### F1-Score

Provides a balance between Precision and Recall.

For this project, Recall is especially important because incorrectly classifying a malignant case as benign can be more serious than other classification errors.

---

## Model Comparison

The notebook creates a comparison table containing:

| Model                | Architecture      | Regularization | Dropout       | Early Stopping | Accuracy              | Precision             | Recall                | F1-Score              |
| -------------------- | ----------------- | -------------- | ------------- | -------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| SLP                  | 30 → 1            | None           | 0             | No             | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook |
| Best MLP             | 30 → 64 → 32 → 1  | None           | 0             | No             | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook |
| MLP + Early Stopping | 30 → 128 → 64 → 1 | None           | 0             | Yes            | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook |
| MLP + Dropout        | 30 → 128 → 64 → 1 | None           | Selected Rate | Yes            | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook |
| MLP + L2             | 30 → 128 → 64 → 1 | L2 (0.001)     | 0             | Yes            | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook |
| Final Combined Model | 30 → 128 → 64 → 1 | L2 (0.001)     | 0.3           | Yes            | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook | Evaluated in Notebook |

The exact numerical results are generated when the notebook is executed.

---

# Threshold Analysis

The final model is also evaluated using different classification thresholds:

```text
0.30
0.35
0.40
0.45
0.50
0.55
0.60
```

For each threshold, the following metrics are calculated:

* Accuracy
* Precision
* Recall
* F1-Score

This analysis demonstrates how changing the classification threshold affects the balance between Precision and Recall.

---

# Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning Utilities

* Scikit-learn

### Model Persistence

* Joblib

### Development Environment

* Google Colab
* Jupyter Notebook

---

# Python Libraries

```text
numpy
pandas
matplotlib
seaborn
tensorflow
scikit-learn
joblib
```

---

# Project Structure

```text
Breast-Cancer-Classification/
│
├── Breast_Cancer_Wisconsin_(Diagnostic).ipynb
│
├── Breast Cancer Wisconsin (Diagnostic).data
│
├── scaler.pkl
│
├── plots/
│   └── class_distribution.png
│
├── README.md
│
└── requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd Breast-Cancer-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

The project can be executed using **Google Colab** or **Jupyter Notebook**.

### Google Colab

1. Open the `.ipynb` file in Google Colab.
2. Upload the dataset.
3. Update the dataset path if required.
4. Run the notebook cells sequentially.
5. Review the visualizations and model results.

### Jupyter Notebook

Run:

```bash
jupyter notebook
```

Open the notebook and execute the cells sequentially.

---

# Output and Results

The project generates:

* Dataset inspection results
* Missing-value analysis
* Class-distribution visualization
* Feature correlation heatmap
* Training and validation curves
* Activation-function comparison
* Confusion matrices
* Classification reports
* Model performance comparison table
* Model accuracy comparison chart
* Classification threshold analysis
* Saved scaler file

---

# Model Persistence

The trained preprocessing scaler is saved using Joblib:

```text
scaler.pkl
```

This allows the same feature-scaling transformation to be reused when making predictions on new data.

---

# Key Learnings

This project demonstrates practical understanding of:

* Exploratory Data Analysis
* Data preprocessing
* Feature scaling
* Binary classification
* Neural-network architecture design
* Activation functions
* Model training
* Validation
* Early Stopping
* Dropout
* L1 regularization
* L2 regularization
* Model evaluation
* Confusion matrix interpretation
* Precision and Recall trade-offs
* F1-Score
* Classification threshold optimization
* Model comparison
* Prevention of data leakage
* Model persistence

---

# Conclusion

This project successfully implements a complete Deep Learning workflow for binary breast cancer classification using the Breast Cancer Wisconsin (Diagnostic) dataset.

The project begins with exploratory data analysis and preprocessing, followed by the development of a Single-Layer Perceptron and multiple Multi-Layer Perceptron architectures. Different activation functions, Early Stopping strategies, Dropout rates, and regularization techniques are systematically investigated.

The final combined model incorporates **ReLU activation, L2 regularization, Dropout, and Early Stopping** to improve model generalization.

Model performance is evaluated using **Accuracy, Precision, Recall, and F1-Score**, rather than relying only on accuracy. Recall is particularly important in this application because false-negative predictions may be more consequential when identifying malignant cases.

Overall, the project demonstrates how a structured Deep Learning pipeline can be developed, optimized, evaluated, and compared for binary classification tasks while emphasizing preprocessing, generalization, and appropriate evaluation metrics.

---

# Disclaimer

This project is intended for **educational and research purposes only**. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

# Author

**Swarna Pathak**

Data Analyst | Python | SQL | Power BI | Machine Learning | Deep Learning

GitHub: Add your GitHub profile URL

LinkedIn: Add your LinkedIn profile URL
