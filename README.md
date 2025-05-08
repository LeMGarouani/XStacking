# XStacking : An Effective and inherently Explainable Framework for Stacked Ensemble Learning

This repository contains the implementation of `XStacking`, and inherently explainable framework for stacked ensemble learning. XStacking is designed to enhance both effectiveness and transparency ofstacked ensemble models decisions. 





---
### Folder structure

Here's a folder structure for XStacking:

```bash
XStacking/  # Root directory
├── datasets/              # dataset-related files
│   ├── classification/    #repository of classification datasets 
│   └── regression/        #repository of regression datasets                 
│   
├── EnsembleLearning/             # Source code for ensemble learing
│   ├── EnsembleLearning.py       # Stacked ensemble learning implementation
│   └── tuner.py                  # Hyperparameter tuning for the model
│
└── README.md               # This overview here :)
```
---
### XStacking algorithm
![XStacking Algorithm](https://github.com/LeMGarouani/XStacking/blob/main/EnsembleLearning/Framework.jpg)


---
## Examples of use

Below is a minimal working example of the `XStacking` algorithm.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC
import xgboost as xgb
import shap
from tuner import TuneClassifier
from EnsembleLearning import StackingClassifier, StackingRegressor

# Load the dataset
path="path to your dataset"
dataset = pd.read_csv(path, sep='\t')
X = dataset.drop('target', axis=1)
y = dataset['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define base learners
base_learners = [
    (DecisionTreeClassifier(max_depth=3)),
    (LogisticRegression()),
    (MLPClassifier(random_state=1, max_iter=700))
]

# Initialize the StackingClassifier or StackingRegressor
stacking_model = StackingClassifier(base_learners, meta_learner='xgb') # stacking for classification
stacking_model = StackingRegressor(base_learners, meta_learner='svr')  # stacking for regression

# Train the stacking the stacking model ; mode=xai for XStacking, normal for traditional stacking
stacking_model.fit(X_train, y_train,mode='xai', ponderation=False)    

# Make predictions and evaluate the model
predictions = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Stacking Model Accuracy: {accuracy}")
```


---
## License

This project is licensed under the MIT License. 
