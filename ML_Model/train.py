

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.model_selection import GridSearchCV
import pickle


MODEL_PATH = "model_alan.pkl"

def train(model_path=MODEL_PATH):
    data  = pd.read_csv("KOI(alan).csv")
    x = data.iloc[:, 1:].values 
    y= data.iloc[:, 0].values 

    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))
    ])

    param_grid = {
    "svm__C": [5, 10, 25, 50],
    "svm__gamma": ["scale", 0.01, 0.05, 0.1]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid.fit(x, y)

    best_pipeline = grid.best_estimator_

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(best_pipeline, x, y, cv=kfold, scoring="accuracy")
    best_pipeline.fit(x,y)


    print("Cross-validation accuracies:", scores.round(2))
    # print(f"Mean accuracy: {scores.mean():.4f}")
    # print(f"Std deviation: {scores.std():.4f}")


    # from sklearn.metrics import classification_report

    # y_pred = best_pipeline.predict(x)

    # print(classification_report(y, y_pred))
    # print("Best parameters:", grid.best_params_)
    # print(f"Best CV score:{ grid.best_score_:.2f}")

    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print("Model saved to", model_path)
if __name__ == "__main__":
    train()
# assigned by AlanFrancis, should have some guts to copy this 
