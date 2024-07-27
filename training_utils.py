from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)

class SVMModel:
    def __init__(self):
        self.model = SVC()
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)



def evaluate_models(X_train, X_test, y_train, y_test):
    results = {}

    # Logistic Regression
    log_reg = LogisticRegressionModel()
    log_reg.train(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    results['Logistic Regression'] = accuracy_score(y_test, y_pred_log_reg)

    # Decision Tree
    decision_tree = DecisionTreeModel()
    decision_tree.train(X_train, y_train)
    y_pred_decision_tree = decision_tree.predict(X_test)
    results['Decision Tree'] = accuracy_score(y_test, y_pred_decision_tree)

    # Random Forest
    random_forest = RandomForestModel()
    random_forest.train(X_train, y_train)
    y_pred_random_forest = random_forest.predict(X_test)
    results['Random Forest'] = accuracy_score(y_test, y_pred_random_forest)

    # SVM
    svm = SVMModel()
    svm.train(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    results['SVM'] = accuracy_score(y_test, y_pred_svm)

    # Print results
    for model_name, accuracy in results.items():
        print(f'{model_name} Accuracy: {accuracy}')
