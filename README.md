# Spam Mail Predictor

A machine learning project that classifies emails as spam or legitimate (ham) using Logistic Regression and TF-IDF feature extraction.

## Overview

This project builds a predictive model to automatically detect and classify spam emails. It uses natural language processing techniques combined with a Logistic Regression classifier to achieve high accuracy in spam detection.

## Features

- **Data Preprocessing**: Handles missing values and cleans email data
- **Feature Extraction**: Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Machine Learning**: Implements Logistic Regression for binary classification
- **Model Evaluation**: Provides accuracy metrics on both training and test datasets
- **Predictive System**: Makes real-time predictions on new email inputs

## Dataset

The project uses a mail classification dataset (`mail_data.csv`) containing:
- **Message**: The email content
- **Category**: Classification label (spam/ham)

The dataset includes emails labeled as either:
- `spam` (0): Unsolicited or malicious emails
- `ham` (1): Legitimate emails

## Project Structure

```
Spam Mail Predictor/
├── Spam Mail Predictor.ipynb    # Main Jupyter notebook
├── Dataset/
│   └── mail_data.csv            # Email classification dataset
└── README.md                    # Project documentation
```

## Requirements

```
numpy
pandas
scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Spam\ Mail\ Predictor
```

2. Install the required dependencies:
```bash
pip install numpy pandas scikit-learn
```

3. Ensure you have Jupyter Notebook installed:
```bash
pip install jupyter
```

## Usage

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Spam Mail Predictor.ipynb`

3. Run all cells sequentially to:
   - Load and preprocess the data
   - Train the Logistic Regression model
   - Evaluate model performance
   - Make predictions on new emails

### Making Predictions

The notebook includes a predictive system that can classify new emails:

```python
input_mail = ["Your email text here"]
input_mail_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_mail_features)

if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')
```

## Model Performance

The model is evaluated on both training and test datasets:
- **Training Data Accuracy**: High accuracy on the data used for training
- **Test Data Accuracy**: Evaluation on unseen data to assess generalization

Accuracy metrics are printed during model evaluation in the notebook.

## Methodology

### 1. Data Collection & Preprocessing
- Loads email data from CSV file
- Replaces null values with empty strings
- Analyzes data distribution and structure

### 2. Label Encoding
- Converts categorical labels to numerical values
- Spam = 0, Ham = 1

### 3. Feature Extraction
- Uses TF-IDF Vectorizer to convert text into numerical features
- Parameters:
  - `min_df=1`: Minimum document frequency
  - `stop_words='english'`: Removes common English words
  - `lowercase=True`: Converts all text to lowercase

### 4. Train-Test Split
- 80% training data, 20% test data
- Random state set to 2 for reproducibility

### 5. Model Training
- Trains Logistic Regression classifier on extracted features
- Binary classification model

### 6. Model Evaluation
- Calculates accuracy scores on training and test datasets
- Provides performance metrics for model assessment

## Key Libraries

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and metrics
  - `train_test_split`: Data splitting
  - `LogisticRegression`: Classification model
  - `accuracy_score`: Performance evaluation
  - `TfidfVectorizer`: Feature extraction

## Results

The Logistic Regression model achieves competitive accuracy in distinguishing between spam and legitimate emails, demonstrating the effectiveness of TF-IDF features combined with a linear classifier for text classification tasks.

## Future Improvements

- Experiment with other algorithms (Random Forest, SVM, Neural Networks)
- Implement cross-validation for more robust evaluation
- Add feature importance analysis
- Create a web interface for real-time predictions
- Optimize hyperparameters using GridSearchCV
- Implement additional preprocessing techniques (lemmatization, stemming)

## Author

[Your Name/Profile]

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request with improvements or bug fixes.

## Contact

For questions or suggestions, please open an issue in the repository.
