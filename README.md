# SMS Spam Classification (CS229 Problem Set 2)

This project implements machine learning algorithms to classify SMS messages as either "Spam" or "Ham" (non-spam). [cite_start]It is the solution to **Problem 6** of the CS229 Supervised Learning II Problem Set[cite: 163].

The project explores two primary classification methods: **Multinomial Naive Bayes** and **Support Vector Machines (SVM)**, building the implementation from scratch to handle text processing, feature extraction, and prediction.

## 📂 Project Overview

* [cite_start]**Goal**: Distinguish between real and spam SMS messages using text analysis[cite: 167, 172].
* [cite_start]**Dataset**: SMS Spam Collection v.1[cite: 169].
* **Models Implemented**:
    1.  [cite_start]**Naive Bayes**: Multinomial event model with Laplace smoothing[cite: 177].
    2.  [cite_start]**SVM**: Radial Basis Function (RBF) kernel with hyperparameter tuning[cite: 195].

## 📊 Dataset

[cite_start]The project uses the **SMS Spam Collection** developed by Tiago A. Almeida and José María Gómez Hidalgo[cite: 169]. The data contains raw text SMS messages labeled as spam or non-spam.

* [cite_start]**Training Data**: `data/ds6_spam_train.tsv` [cite: 170]
* [cite_start]**Testing Data**: `data/ds6_spam_test.tsv` [cite: 170]

## 🛠️ Methodology

### 1. Data Preprocessing
Raw SMS messages are converted into numerical feature vectors using the **Bag-of-Words** model:
* **Tokenization**: Messages are split into words and converted to lowercase.
* **Dictionary Creation**: A vocabulary is built using only words that appear in at least **5 messages** to reduce noise.
* **Vectorization**: Text is transformed into a matrix where $x_j^{(i)}$ represents the frequency of word $j$ in message $i$.

### 2. Naive Bayes Classifier


[Image of Naive Bayes classifier diagram]

* **Smoothing**: Implements Laplace smoothing to handle zero-frequency words.
* [cite_start]**Log-Domain Prediction**: To prevent numerical underflow (caused by multiplying many small probabilities), predictions are calculated in the log domain[cite: 183]:
    $$\sum \log p(x_k|y) + \log p(y)$$
* [cite_start]**Interpretability**: The model identifies "indicative words" by comparing the log-probability ratio of a word appearing in spam vs. ham[cite: 187].

### 3. Support Vector Machine (SVM)

* [cite_start]**Kernel**: Uses a Radial Basis Function (RBF) kernel[cite: 195].
* [cite_start]**Hyperparameter Tuning**: A Grid Search is performed to find the optimal **radius** ($\gamma$) for the RBF kernel by maximizing accuracy on a validation set[cite: 197].

## 🚀 Results

The models were evaluated on the held-out testing set.

| Model | Accuracy | Notes |
| :--- | :--- | :--- |
| **Naive Bayes** | **97.85%** | Highly effective for text data. |
| **SVM (RBF)** | 96.77% | Optimal radius found: `0.1`. |

### Top 5 Spam Indicators
The Naive Bayes analysis identified the following words as the strongest indicators of spam:
1.  `claim`
2.  `won`
3.  `prize`
4.  `tone`
5.  `urgent!`

## 💻 Usage

### Prerequisites
* Python 3.x
* NumPy

### Running the Project
1.  Ensure the dataset is located in the `data/` directory.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Spam classification .ipynb"
    ```
3.  Run all cells to execute the training, prediction, and evaluation pipeline.

## 📁 File Structure

* `Spam classification .ipynb`: Main execution notebook containing logic and results.
* `src/`: Source code directory.
    * `p06_spam.py`: (Implied) Contains function definitions for `get_words`, `transform_text`, etc.
    * [cite_start]`svm.py`: (Implied) Provided SVM implementation[cite: 195].
    * `util.py`: (Implied) Helper functions for loading data.
* [cite_start]`output/`: Directory where model predictions and dictionaries are saved[cite: 176].

## 📜 References
* [cite_start]CS229: Machine Learning - Problem Set 2[cite: 1, 3].
* [cite_start]Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering: New Collection and Results* (2011)[cite: 190].
