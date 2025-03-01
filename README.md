# ML-Playground

Welcome to **ML-Playground** repository, a machine learning portfolio where I implement various machine learning algorithms **from scratch** without relying on existing libraries like `scikit-learn` and test them on **public datasets**. 

## 🚀 What You'll Find Here
- **Implementations from Scratch**: Custom-built machine learning algorithms.
- **Classification & Regression Problems**
- **Model Evaluation**: Performance analysis and visualization of trained models.


## 📂 Repository Structure

- **ml-playground/**
  - **notebooks/** - Python notebook scripts for classification and regression problems
  - **src/** - Custom ML implementations
    - `logistic_regression_from_scratch.py`
    - `linear_regression_from_scratch.py`
    - `decision_tree_classifier_from_scratch.py`
  - **data/** - Datasets (ignored in `.gitignore`)
  - `requirements.txt` - Dependencies
  - `README.md` - Project Documentation 


## 🏗️ Implemented Machine Learning Algorithms
| Algorithm               | Implementation Type |
|-------------------------|---------------------|
| Logistic Regression     | Scratch & Sklearn  |
| Linear Regression       | Scratch & Sklearn  |
| Decision Tree Classifier| Scratch & Sklearn  |


## ⚡ How to Set Up the Environment
This project uses **Conda** for dependency management. To create a virtual environment with all required packages, run the following command:

```sh
conda create --name ml-playground --file requirements.txt
```

Once the environment is created, activate it:
```sh
conda activate ml-playground
```