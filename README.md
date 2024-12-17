# Facial Recognition and Identification System

## Project Overview
This project focuses on incorporating **facial recognition and identification** into a security system for a law enforcement company, **ABC**. The **UMIST_Cropped.mat** dataset is used to train and validate the system. The project involves data preprocessing, clustering, training a neural network, and evaluating the results.

---

## Table of Contents
1. [Dataset](#dataset)
2. [Data Splitting](#data-splitting)
3. [Data Preprocessing](#data-preprocessing)
4. [Clustering Technique](#clustering-technique)
5. [Neural Network Architecture](#neural-network-architecture)
6. [Results and Analysis](#results-and-analysis)
7. [Challenges and Decisions](#challenges-and-decisions)
8. [Running the Project](#running-the-project)

---

## Dataset
The **UMIST_Cropped.mat** dataset contains facial images from multiple individuals. It is provided as input to train, validate, and test the facial recognition system.

---

## Data Splitting
We split the dataset into **training**, **validation**, and **test sets** using **stratified sampling** to ensure an equal number of images per person in each set. 

- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

**Rationale**: Stratified sampling ensures fair representation of each class (person), avoiding bias and improving generalization.

---

## Data Preprocessing
To improve the model's performance, the data undergoes the following preprocessing steps:

1. **Normalization**: Images are scaled by dividing pixel values by 255 to ensure consistent input.
   - Formula: \[ X_{norm} = \frac{X}{255} \]

2. **Resizing**: Images are resized to a standard dimension for uniformity.

3. **Dimensionality Reduction**:
   - **t-SNE and PCA** were both used for **K-Means++ clustering** to enhance feature extraction and improve clustering accuracy.
   - For all other clustering methods, **PCA alone** was used.
   - **Rationale**: t-SNE and PCA together enhance visual representation and clustering performance, while PCA reduces computational complexity for other clustering methods.

4. **Data Augmentation**: To balance the dataset, we generated augmented images for each category until all categories had the same number of images.

---

## Clustering Technique
We applied the following clustering techniques:

1. **DBSCAN**
2. **Gaussian Mixture Models (GMM)**
3. **K-Means**
4. **K-Means++ with t-SNE and PCA**

**Evaluation**: Each clustering technique was evaluated using the **Adjusted Rand Index (ARI)** score. The best score was achieved using **K-Means++ with t-SNE and PCA**.

---

## Neural Network Architecture
The following neural network was designed and trained using the best clustering technique (**K-Means++ with t-SNE and PCA**):

```python
from tensorflow import keras
from tensorflow.keras import layers

# Concatenate cluster labels with flattened image data
X_train_combined = np.concatenate((X_train_pca, X_train_reduced), axis=1)
X_val_combined = np.concatenate((X_val_pca, X_val_reduced), axis=1)
X_test_combined = np.concatenate((X_test_pca, X_test_reduced), axis=1)

model = keras.Sequential([
    layers.Input(shape=(X_train_combined.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(best_score, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

**Key Components**:
- **Activation Function**: ReLU for hidden layers and Softmax for the output layer.
- **Loss Function**: Sparse Categorical Cross-Entropy, suitable for multi-class classification.
- **Optimization**: Adam optimizer for efficient training.
- **Input Features**: Concatenated cluster labels and flattened image data.

---

## Results and Analysis
The system's performance was evaluated using the test set. Metrics such as **accuracy**, **precision**, and **recall** were calculated.

- **Best Clustering Technique**: K-Means++ with t-SNE and PCA.
- **Accuracy Achieved**: XX% (example placeholder)
- Visualizations of loss and accuracy trends during training are provided in the report.

---

## Challenges and Decisions
### Challenges Faced
1. Balancing the dataset while maintaining fairness.
2. Optimizing hyperparameters for better accuracy.
3. Selecting the best clustering technique using ARI scores.

### Decisions Made
- Use of **t-SNE and PCA** for K-Means++ clustering to improve accuracy.
- Application of **DBSCAN**, **GMM**, and **K-Means** for evaluation.
- Balancing the dataset using **data augmentation**.
- Fine-tuning the neural network architecture based on validation results.

---

## Running the Project
Follow these steps to run the project:

1. **Install Dependencies**:
   ```bash
   pip install numpy scipy sklearn tensorflow matplotlib
   ```
2. **Load the Dataset**: Ensure `UMIST_Cropped.mat` is placed in the project directory.

3. **Run the Main Script**:
   ```bash
   python main.py
   ```

4. **View Results**: Output results and visualizations will be saved in the `results/` directory.

---


