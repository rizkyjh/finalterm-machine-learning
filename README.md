# finalterm-machine-learning
# Machine Learning Portfolio: Hands-On End-to-End Models
### Final Term Submission

**Student Identification**
* **Name:** Rizky Januar Hardi
* **Class:** Machine Learning
* **NIM:** 1103220166

---

## Repository Purpose
This repository serves as the comprehensive submission for the Machine Learning course. It contains three distinct end-to-end pipelines demonstrating mastery over different machine learning domains:
1.  **Binary Classification:** Fraud Detection (Handling Imbalanced Data).
2.  **Regression:** Song Year Prediction (Audio Feature Analysis).
3.  **Computer Vision:** Fish Species Classification (Deep Learning with CNNs).

---

## Project 1: End-to-End Fraud Detection Pipeline
### (Midterm Task 1)

**1. Project Purpose**
The objective of this project is to design and implement a robust Machine Learning pipeline to detect fraudulent online transactions. This task focuses on binary classification using real-world data with significant class imbalance.

**2. Dataset & Overview**
* **Dataset:** IEEE-CIS Fraud Detection (`train_transaction.csv`).
* **Target Variable:** `isFraud` (0 = Legitimate, 1 = Fraud).
* **Key Challenge:** The dataset is highly imbalanced (fraud cases are very rare compared to normal transactions).
* **Preprocessing Steps:**
    * **Data Cleaning:** Removed columns with >70% missing values to optimize memory usage.
    * **Imputation:** Filled missing numerical values with 0 and categorical values with "Unknown".
    * **Encoding:** Applied Label Encoding to convert categorical features into numeric format.
    * **Scaling:** Used StandardScaler to normalize features for the Neural Network.

**3. Models & Results**
We implemented and compared two models, focusing on handling the class imbalance using `class_weight='balanced'`.

| Model | Type | Key Configuration | Metric (ROC-AUC) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear | `class_weight='balanced'` | Baseline Score |
| **Neural Network (MLP)** | Deep Learning | Layers: (64, 32), ReLU | Deep Learning Score |

**Conclusion:** The Logistic Regression model provided a stable baseline, while the Neural Network captured more complex non-linear relationships. The final predictions for the test set are saved in `submission.csv`.

---

## Project 2: Song Year Prediction Pipeline
### (Midterm Task 2)

**1. Project Purpose**
To build an end-to-end regression system that predicts the release year of a song based on its audio characteristics (timbre, texture, etc.). This demonstrates the ability to model continuous target variables using both traditional ML and Deep Learning.

**2. Dataset Details**
* **Source:** `midterm-regresi-dataset.csv`
* **Structure:** The first column is the Target (Year), followed by 90 continuous numerical audio features.
* **Preprocessing:**
    * Renamed the first column to `Year`.
    * Applied StandardScaler to all feature columns to ensure Neural Network convergence.
    * Split data into 80% Training and 20% Testing sets.

**3. Models Implemented**
We compared a robust ensemble method against a Deep Learning approach:
* **Random Forest Regressor:**
    * A non-linear model that builds multiple decision trees and averages their outputs.
    * Configured with `n_estimators=10` (Speed Mode) to handle the large dataset size (~500k rows) efficiently.
* **MLP Regressor (Deep Learning):**
    * A Feed-Forward Neural Network.
    * Architecture: Hidden layers of (50,) neurons with ReLU activation.

**4. Performance Evaluation**
We used **Root Mean Squared Error (RMSE)** to measure the average error in years.

| Model | RMSE (Lower is Better) | Performance Analysis |
| :--- | :--- | :--- |
| **Random Forest** | *[Insert RMSE]* | Typically handles tabular audio data better. |
| **Neural Network** | *[Insert RMSE]* | Requires more tuning but captures abstract patterns. |

---

## Project 3: Fish Image Classification (CNN)
### (Final Term Task)

**1. Project Purpose**
To develop a complete Computer Vision pipeline that classifies fish species from raw images using a Convolutional Neural Network (CNN). This project demonstrates data ingestion optimization, image augmentation, and custom Deep Learning architecture design.

**2. Dataset**
* **Source:** Fish Image Dataset
* **Structure:** 31 distinct fish species organized into Train, Validation, and Test directories.
* **Preprocessing:**
    * **Vectorization:** Implemented `tf.data.Dataset` pipelines with `prefetch` and `cache` for high-speed GPU training.
    * **Augmentation:** Applied random horizontal flips, rotations, and zooms to prevent overfitting.
    * **Normalization:** Rescaled pixel values from [0, 255] to [0, 1].

**3. CNN Architecture**
We built a sequential CNN from scratch with the following structure:
* **Convolutional Layers:** Three blocks of Conv2D (32, 64, 128 filters) + MaxPooling to extract hierarchical features.
* **Fully Connected Layers:** Dense layer (128 units) with ReLU activation.
* **Regularization:** Dropout (0.5) layer to reduce overfitting.
* **Output:** Softmax layer with 31 units for multi-class probability.

**4. Evaluation**
* **Training vs Validation:** Monitored accuracy curves to detect overfitting.
* **Confusion Matrix:** Generated heatmaps to identify which fish species are commonly confused.
* **Real-world Test:** The model includes a script to predict species from single raw image files with confidence scores.

---

## How to Navigate This Repository

1.  **Fraud Detection:**
    * `midterm_fraud_detection.ipynb`: Main code for Task 1.
    * `submission.csv`: Probability scores for fraud detection.

2.  **Song Prediction:**
    * `midterm_regression.ipynb`: Main code for Task 2.
    * `midterm-regresi-dataset.csv`: Audio feature dataset.

3.  **Fish Classification:**
    * `finalterm-CNN.ipynb`: Main code for Task 3 (CNN).
    * `fish_classifier_final.keras`: The saved trained model file.

## Dependencies
To run all notebooks in this repository, the following libraries are required:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow opencv-python
