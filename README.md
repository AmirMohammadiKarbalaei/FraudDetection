# FraudDetection


## Overview


FraudDetection is a Python-based system for detecting fraudulent transactions in financial data. The system uses a combination of machine learning algorithms and data preprocessing techniques to identify potentially fraudulent transactions.

## Features


*   **Data Preprocessing**: The system includes a data preprocessing module that handles missing values, encodes categorical variables, and normalises numerical features.
*   **Machine Learning Models**: The system includes two machine learning models: a deep learning model (FraudDetectionModel) and a random forest model (FraudDetectionRFModel).
*   **Model Evaluation**: The system includes a model evaluation module that calculates accuracy, precision, recall, and F1 score for each model.
*   **Model Saving and Loading**: The system allows for saving and loading trained models for future use.

## Requirements


*   **Python**: The system is built using Python 3.10.14.
*   **Libraries**: The system requires the following libraries:
    *   `pandas` for data manipulation and analysis
    *   `numpy` for numerical computations
    *   `torch` for deep learning
    *   `sklearn` for machine learning
    *   `matplotlib` and `seaborn` for data visualisation

## Usage


### Data Preparation

1.  Download the dataset (e.g., `Fraud.csv`) and place it in the `Fraud_data` directory.
2.  Preprocess the data using the `DataPreprocessing` module.

### Model Training

1.  Train the deep learning model using the `FraudDetectionModel` class.
2.  Train the random forest model using the `FraudDetectionRFModel` class.

### Model Evaluation

1.  Evaluate the performance of each model using the `ModelEvaluation` module.

### Model Saving and Loading

1.  Save the trained models using the `ModelSaving` module.
2.  Load the saved models using the `ModelLoading` module.

## Example Use Cases


*   **Fraud Detection**: Use the system to detect fraudulent transactions in a financial dataset.
*   **Model Comparison**: Use the system to compare the performance of different machine learning models on a fraud detection task.

## Contributing


Contributions are welcome! If you would like to contribute to the FraudDetection system, please fork the repository and submit a pull request.

## License


The FraudDetection system is released under the MIT License.