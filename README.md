# Age Group Prediction Challenge

## Overview

This repository contains the solution for a classification challenge focused on predicting the `age_group` (Adult or Senior) of respondents based on various health and activity-related features. The goal is to build a machine learning model that accurately classifies individuals into one of two age groups.

## Problem Statement

The challenge involves predicting whether a respondent is an "Adult" (under 65 years old) or a "Senior" (65 years old and older) based on provided demographic, health, and activity data. The target variable `age_group` needs to be mapped to `0` for Adult and `1` for Senior. The dataset contains missing values that require appropriate handling.

## Dataset

The data is provided in two main files: `train.csv` and `test.csv`, along with a `sample_submission.csv` for formatting.

* **`train.csv`**: Contains the training set observations (2,016 rows) with the target variable (`age_group`) for model training.
* **`test.csv`**: Contains the testing set observations (312 rows) without the target column. Predictions from the trained model should be made for this set.
* **`sample_submission.csv`**: A sample file demonstrating the required format for the submission. The `age_group` column should contain `0` or `1`.

### Target Variable Classes:

* `age_group`: Adult - `0`
* `age_group`: Senior - `1`

### Features:

The dataset includes the following features:

* **`SEQN`**: Sequence number (identifier).
* **`RIAGENDR`**: Respondent's Gender (1=Male, 2=Female).
* **`PAQ605`**: Physical activity questionnaire response (Engages in moderate/vigorous-intensity sports/activities in a typical week).
* **`BMXBMI`**: Body Mass Index.
* **`LBXGLU`**: Glucose level.
* **`DIQ010`**: Diabetes questionnaire response.
* **`LBXGLT`**: Glucose tolerance (Oral).
* **`LBXIN`**: Insulin level.

**Note**: The dataset contains missing values (NaNs).

## Solution Approach

The solution involves a typical machine learning pipeline for a classification task:

1.  **Data Loading**: Load `train.csv`, `test.csv`, and `sample_submission.csv` into pandas DataFrames.
2.  **Exploratory Data Analysis (EDA)**: Initial inspection of data types, missing values, and distribution of the target variable.
3.  **Missing Value Handling**:
    * For the target variable (`age_group`) in the training data, rows with missing values were dropped to ensure a clean target for training.
    * For numerical features, missing values were imputed using the **median strategy** (robust to outliers).
4.  **Feature Engineering & Preprocessing**:
    * The `SEQN` (Sequence number) column was dropped as it is an identifier and not a predictive feature.
    * Categorical features (`RIAGENDR`, `PAQ605`, `DIQ010`) were identified and converted to `object` type to ensure proper handling.
    * One-Hot Encoding was applied to the categorical features to convert them into a numerical format suitable for machine learning algorithms. `drop_first=True` was used to avoid multicollinearity.
    * Ensured consistent columns between the training and test sets after one-hot encoding to handle cases where a category might appear in one set but not the other.
5.  **Target Variable Encoding**: The `age_group` column in the training data was mapped from string labels ('Adult', 'Senior') to numerical values (0, 1) respectively.
6.  **Model Selection**: A `RandomForestClassifier` was chosen for its robustness and good performance on tabular data without extensive hyperparameter tuning.
7.  **Model Training**: The classifier was trained on the preprocessed training features and encoded target variable.
8.  **Prediction**: Predictions were made on the preprocessed test set.
9.  **Submission File Generation**: The predictions were formatted into a `submission.csv` file, matching the `sample_submission.csv` structure.

## Technologies Used

* **Python 3.x**
* **pandas**: For data manipulation and analysis.
* **scikit-learn**: For data preprocessing (imputation, encoding) and machine learning (RandomForestClassifier).

## How to Run the Code (Reproduce Results)

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Place the data files:** Ensure `train.csv`, `test.csv`, and `sample_submission.csv` are placed in the root directory of the cloned repository.
3.  **Install dependencies:**
    ```bash
    pip install pandas scikit-learn
    ```
4.  **Run the script:**
    Execute the Python script containing the provided code. The code demonstrates the full pipeline from data loading to submission file generation.
    ```bash
    python your_solution_script_name.py
    ```
    (Replace `your_solution_script_name.py` with the actual name of your Python file.)

## Files in this Repository

* `train.csv`: Training data.
* `test.csv`: Test data.
* `sample_submission.csv`: Example submission format.
* `Aiplanethackathon.ipynb': The main script/notebook containing the Python code for data loading, preprocessing, model training, and prediction.
* `submission.csv`: The generated submission file after running the code.

## Results

The `submission.csv` file will be generated in the root directory, containing the predicted `age_group` (0 or 1) for each entry in the `test.csv` file.

## Further Enhancements (Potential Future Work)

* **Hyperparameter Tuning**: Optimize `RandomForestClassifier` parameters using techniques like GridSearchCV or RandomizedSearchCV.
* **Cross-Validation**: Implement k-fold cross-validation for more robust model evaluation.
* **Other Models**: Experiment with other classification algorithms (e.g., Gradient Boosting, Support Vector Machines, Neural Networks).
* **Advanced Feature Engineering**: Explore creating new features from existing ones (e.g., interaction terms, polynomial features) if domain knowledge suggests it.
* **Outlier Detection/Handling**: Investigate and handle outliers if they significantly impact model performance.
* **Imputation Strategies**: Experiment with more sophisticated imputation methods (e.g., K-Nearest Neighbors imputation).

---

**Author:** Iffa
**Date:** June 28, 2025
