# Intrusion Detection System (IDS)

This repository contains the code and documentation for an Intrusion Detection System (IDS) project, which aims to classify network traffic as either normal or an attack using machine learning techniques. The dataset used for this project is the NSL-KDD dataset, a refined version of the KDD Cup 99 dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Intrusion Detection Systems (IDS) are critical for maintaining the security of network systems. This project involves developing an IDS that can accurately classify network traffic as either normal or an attack. The machine learning models used include Decision Tree and Logistic Regression.

## Dataset

The dataset used in this project is the NSL-KDD dataset, which contains 42 features and a target variable indicating the type of network traffic (normal or an attack). The attacks are categorized into four main types: DoS, Probe, R2L, and U2R.

## Preprocessing

### Data Loading

The dataset is loaded using Pandas:

```python
import pandas as pd

df = pd.read_csv('Train_data_1.csv')
```

### Handling Categorical Data

Categorical features are mapped to numerical values:

```python
proto_dict = {'tcp': 0, 'udp': 1, 'icmp': 2}
df['protocol_type'] = df['protocol_type'].map(proto_dict)

proto_dict_1 = {'normal': 0, 'dos': 1, 'r2l': 1, 'probe': 1, 'u2r': 1}
df['xAttack'] = df['xAttack'].map(proto_dict_1)
```

### Handling Missing Values

The dataset is checked for missing values:

```python
empty_row = df.isnull().all(axis=0)
empty_col = df.isnull().all(axis=1)
print(empty_row)
print(empty_col)
```

### Feature and Target Separation

Features and the target variable are separated:

```python
y = df['xAttack']
X = df.drop('xAttack', axis=1)
```

## Model Training and Evaluation

### Data Splitting

The dataset is split into training and validation sets:

```python
from sklearn.model_selection import train_test_split

Train_X, Val_x, Train_y, Val_y = train_test_split(X, y, test_size=0.2, random_state=1)
```

### Decision Tree Classifier

A Decision Tree Classifier is trained and evaluated:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

df_model = DecisionTreeClassifier()
df_model.fit(Train_X, Train_y)

prediction = df_model.predict(Val_x)
print(classification_report(Val_y, prediction))
print(accuracy_score(Val_y, prediction))
```

### Logistic Regression

A Logistic Regression model is also trained and evaluated for comparison:

```python
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=10000)
log_model.fit(Train_X, Train_y)

log_prediction = log_model.predict(Val_x)
print(classification_report(Val_y, log_prediction))
print(accuracy_score(Val_y, log_prediction))
```

## Results

The Decision Tree model achieved an accuracy of 99.78%, with perfect precision, recall, and F1-scores. The Logistic Regression model achieved an accuracy of 66.52%, with lower precision, recall, and F1-scores compared to the Decision Tree model.

### Decision Tree Classifier

```plaintext
Accuracy: 99.78%

Precision, Recall, F1-Score:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     13506
           1       1.00      1.00      1.00     11689

    accuracy                           1.00     25195
   macro avg       1.00      1.00      1.00     25195
weighted avg       1.00      1.00      1.00     25195
```

### Logistic Regression

```plaintext
Accuracy: 66.52%

Precision, Recall, F1-Score:

              precision    recall  f1-score   support

           0       0.85      0.46      0.59     13506
           1       0.59      0.91      0.72     11689

    accuracy                           0.67     25195
   macro avg       0.72      0.68      0.65     25195
weighted avg       0.73      0.67      0.65     25195
```

## Usage

To use this code, clone the repository and run the Jupyter notebooks provided. Ensure you have the necessary dependencies installed:

```bash
pip install pandas scikit-learn
```

Load the dataset and execute the cells to preprocess the data, train the models, and evaluate the results.

## Contributing

Contributions are welcome! Please create a pull request with your changes or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize any section as needed!
