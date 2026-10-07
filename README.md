# 🍷 Wine Quality Prediction — Random Forest vs SGD vs SVC

A machine learning classification project that predicts **wine quality** from physicochemical properties such as acidity, sulphates, sulfur dioxide, density, pH, and alcohol content.

The project compares three classification algorithms — **Random Forest, Support Vector Classifier (SVC), and SGD Classifier** — and evaluates their performance using accuracy, precision, recall, F1-score, cross-validation, and confusion matrices.

---

## 📌 Project Overview

Wine quality is influenced by several physicochemical characteristics. The objective of this project is to build a machine learning model capable of predicting whether a wine belongs to the **Good** or **Bad** quality category based on its measurable chemical properties.

The project follows an end-to-end machine learning workflow:

* Dataset loading and inspection
* Data quality checking
* Exploratory Data Analysis (EDA)
* Class imbalance analysis
* Feature engineering
* Stratified train-test splitting
* Model training
* Hyperparameter tuning
* Model evaluation
* Cross-validation
* Feature importance analysis
* Model comparison
* Final model recommendation

---

## 🎯 Objective

The primary objectives of this project are:

1. Analyze the physicochemical characteristics of red wine.
2. Understand the relationship between wine properties and quality.
3. Handle the class imbalance present in the original quality labels.
4. Build classification models for wine quality prediction.
5. Compare different machine learning algorithms.
6. Identify the most important features affecting wine quality.
7. Select the best-performing model for potential deployment.

---

## 📊 Dataset

The project uses the **Wine Quality Dataset (`WineQT.csv`)**, containing physicochemical measurements of red wine samples.

### Dataset Information

| Property           |     Value |
| ------------------ | --------: |
| Samples            |     1,143 |
| Features/Columns   |        13 |
| Numerical Features |        11 |
| Target             | `quality` |
| Quality Range      |       3–8 |
| Missing Values     |         0 |
| Duplicate Rows     |   Checked |
| Dataset Type       |  Red Wine |

### Features

| Feature                | Description                            |
| ---------------------- | -------------------------------------- |
| `fixed acidity`        | Fixed/non-volatile acidity of the wine |
| `volatile acidity`     | Volatile acidity level                 |
| `citric acid`          | Citric acid concentration              |
| `residual sugar`       | Amount of remaining sugar              |
| `chlorides`            | Chloride concentration                 |
| `free sulfur dioxide`  | Free sulfur dioxide level              |
| `total sulfur dioxide` | Total sulfur dioxide level             |
| `density`              | Density of the wine                    |
| `pH`                   | Acidity/basicity level                 |
| `sulphates`            | Sulphate concentration                 |
| `alcohol`              | Alcohol percentage                     |
| `quality`              | Original wine quality score            |
| `Id`                   | Dataset identifier                     |

---

## 🧠 Machine Learning Approach

### 1. Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv("WineQT.csv")
```

The dataset is then inspected using:

* `head()`
* `info()`
* `describe()`
* Missing-value checks
* Duplicate-row checks

---

### 2. Exploratory Data Analysis

EDA was performed to understand the distribution and relationships between the variables.

The analysis includes:

* Feature distributions
* Wine quality distribution
* Correlation analysis
* Correlation heatmap
* Relationship between physicochemical properties and quality
* Class imbalance analysis

The analysis showed that several chemical properties have meaningful relationships with wine quality.

---

## ⚖️ Handling Class Imbalance

The original `quality` target contains six classes:

```text
3, 4, 5, 6, 7, 8
```

However, the dataset is highly imbalanced. Some quality classes contain very few observations.

Therefore, the project creates a binary target:

### Binary Classification

```text
Bad   → quality < 6
Good  → quality >= 6
```

This produces a more balanced target distribution of approximately:

* **Bad:** 46%
* **Good:** 54%

This approach makes the classification problem more suitable for the available dataset size.

---

## 🔬 Models Implemented

Three machine learning classification algorithms were trained and compared.

### 🌲 1. Random Forest Classifier

Random Forest is an ensemble learning algorithm based on multiple decision trees.

Advantages:

* Handles non-linear relationships
* Captures feature interactions
* Does not require feature scaling
* Provides feature importance
* Robust to many types of feature distributions

---

### 📈 2. SGD Classifier

The Stochastic Gradient Descent classifier provides a lightweight and computationally efficient linear classification approach.

Advantages:

* Fast training
* Low computational cost
* Suitable for large datasets

However, because it uses a linear decision boundary, it may not capture complex non-linear relationships between wine properties.

---

### 🎯 3. Support Vector Classifier (SVC)

The project uses SVC with an RBF kernel.

SVC is useful for capturing non-linear relationships and performs well when features are properly scaled.

Advantages:

* Effective for non-linear classification
* Works well with moderate-sized datasets
* RBF kernel can capture complex decision boundaries

---

## 🔄 Preprocessing Pipeline

For models requiring scaling, the project uses:

```text
StandardScaler
      ↓
Machine Learning Model
```

A consistent `random_state = 42` is used to make the experiments reproducible.

The train-test split uses **stratification** to preserve the target-class distribution.

---

## 📏 Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix
* 5-Fold Cross-Validation
* Macro F1 for 3-class sanity checking

---

## 🏆 Model Performance

### Binary Classification — Good vs Bad

| Model             | CV Accuracy (Train) | Test Accuracy | Weighted F1 |         5-Fold CV |
| ----------------- | ------------------: | ------------: | ----------: | ----------------: |
| **Random Forest** |           **0.781** |     **0.821** |    **0.82** | **0.786 ± 0.013** |
| SVC (RBF)         |               0.763 |         0.786 |        0.79 |     0.755 ± 0.013 |
| SGD               |               0.751 |         0.777 |        0.78 |     0.754 ± 0.009 |

### 🥇 Best Model: Random Forest

Random Forest achieved the best overall performance:

* **Test Accuracy:** ~82.1%
* **Weighted F1-score:** ~0.82
* **5-Fold CV Accuracy:** ~78.6% ± 1.3%

It consistently outperformed both SVC and SGD.

---

## 🔍 3-Class Sanity Check

A separate experiment was performed using three wine-quality categories.

The results were:

| Model             |   Accuracy |   Macro F1 | High-Class F1 |
| ----------------- | ---------: | ---------: | ------------: |
| **Random Forest** | **0.6856** | **0.6651** |    **0.6038** |
| SVC               |     0.6463 |     0.5858 |        0.4167 |
| SGD               |     0.6245 |     0.4572 |        0.0588 |

Random Forest remained the strongest model in the 3-class experiment.

The results also demonstrate that predicting the high-quality class becomes significantly more difficult when the problem is formulated as a multi-class classification task.

---

## ⭐ Feature Importance

Random Forest feature importance analysis identified the following features as particularly important:

1. **Alcohol**
2. **Sulphates**
3. **Total Sulfur Dioxide**
4. **Volatile Acidity**

These features showed meaningful relationships with wine quality in the analysis.

### Key Insight

**Alcohol** was the most important predictor in the Random Forest model, followed by sulphates, total sulfur dioxide, and volatile acidity.

---

## 💡 Key Findings

### 1. Random Forest performed best

Random Forest achieved the highest test accuracy and F1-score among the evaluated models.

### 2. Alcohol is an important predictor

Alcohol concentration was identified as the most influential feature in the Random Forest feature-importance analysis.

### 3. Binary classification is more practical for this dataset

The original six-class target was heavily imbalanced. Converting it into Good/Bad categories resulted in a more balanced classification problem.

### 4. SVC performed competitively

SVC achieved approximately **78.6% test accuracy**, making it the second-best model in the comparison.

### 5. SGD was the fastest but least accurate

SGD provides a lightweight solution but its linear decision boundary is less capable of capturing the non-linear relationships present in wine chemistry.

---

## 🏅 Final Model Recommendation

### Random Forest Classifier

Random Forest is recommended as the final model because:

* It achieved the highest test accuracy.
* It achieved the highest weighted F1-score.
* It performed consistently during cross-validation.
* It handles non-linear relationships effectively.
* It does not require feature scaling.
* It provides feature importance for interpretability.
* It can capture interactions between physicochemical properties.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook
* Google Colab

---

## 📁 Project Structure

```text
Wine-Quality-Prediction/
│
├── WineQT.csv
├── Wine_Quality_Prediction.ipynb
├── README.md
└── requirements.txt
```

> File names can be adjusted according to the actual names used in the GitHub repository.

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd Wine-Quality-Prediction
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

Open:

```text
Wine_Quality_Prediction.ipynb
```

Make sure `WineQT.csv` is located in the same directory as the notebook.

---

## 📌 Project Workflow

```text
Wine Quality Dataset
        ↓
Data Loading
        ↓
Data Inspection
        ↓
Data Quality Checks
        ↓
Exploratory Data Analysis
        ↓
Class Imbalance Analysis
        ↓
Feature Engineering
        ↓
Binary Target Creation
        ↓
Stratified Train/Test Split
        ↓
Feature Scaling
        ↓
Model Training
   ┌────┼────┐
   ↓    ↓    ↓
  RF   SGD  SVC
   └────┼────┘
        ↓
Model Evaluation
        ↓
Cross Validation
        ↓
Feature Importance
        ↓
Model Comparison
        ↓
Random Forest Selected
```

---

## ⚠️ Limitations

Although the model provides promising results, there are some limitations:

* The dataset contains only **1,143 samples**.
* The test set contains approximately **229 samples**, so small differences in model performance may be influenced by sampling variation.
* Wine quality ratings are subjective human assessments.
* The original quality classes are highly imbalanced.
* The binary classification approach simplifies the original six-class quality problem.

---

## 🔮 Future Improvements

The project can be further improved by exploring:

* SMOTE for handling class imbalance
* `class_weight` for multi-class classification
* XGBoost
* LightGBM
* Gradient Boosting
* Feature selection
* Probability calibration
* Classification threshold tuning
* More extensive hyperparameter optimization
* Repeated cross-validation
* Larger wine-quality datasets
* Deployment using Streamlit or Flask

---

## 📈 Business / Practical Application

A wine quality prediction system can potentially support:

* Wine quality assessment
* Quality-control processes
* Production monitoring
* Identification of important chemical properties
* Data-driven wine production decisions
* Automated quality classification

The model should be considered a **decision-support tool**, rather than a replacement for professional wine-quality assessment.

---

## 👨‍💻 Author

**Manoj Kumar**

Aspiring Data Analyst / Machine Learning Enthusiast

* GitHub: `github.com/manojbais6268-a11y`
* LinkedIn: `linkedin.com/in/manoj-kumar-86a560322`

---

## 📜 License

This project is intended for **educational and portfolio purposes**.

If you reuse or modify the project, please provide appropriate attribution to the original dataset and project author.

