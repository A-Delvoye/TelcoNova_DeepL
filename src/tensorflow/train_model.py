import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import src.mlflow_script as mlfs
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = "A-Delvoye"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/A-Delvoye/TelcoNova_DeepL.mlflow"
)
import mlflow

def make_preprocess_pipeline(numerical_features, categorical_features):
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor

def preprocess_data(df):
    feature_of_interest = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]

    target = 'Churn'

    categorical_features = [
        "DeviceProtection",
        "MultipleLines",	
        "InternetService",	
        "OnlineSecurity",
        "OnlineBackup",
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaymentMethod'
    ]

    numerical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'tenure', 'PhoneService', 'PaperlessBilling',
        'MonthlyCharges', 'TotalCharges'
    ]

    df["gender"]= df["gender"].apply(lambda x: 0 if x ==  'Male' else 1)
    for col in ["Partner", "PhoneService", 'PaperlessBilling', "Dependents", 'Churn']:
        df[col]= df[col].apply(lambda x: 0 if x ==  'No' else 1)
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.set_index("customerID", inplace=True)

    X = df.drop(columns=[target])
    y = df[target]

    # Séparation train/val/test (80/20 puis 20% de train pour val)
    X_train_0, X_test, y_train_0, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # On prend 20% de X_train pour validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_0, y_train_0, test_size=0.2, random_state=42, stratify=y_train_0
    )

    preprocessor = make_preprocess_pipeline(numerical_features, categorical_features)

    X_train = preprocessor.fit_transform(X_train)
    X_val  = preprocessor.transform(X_val)
    X_test  = preprocessor.transform(X_test)

    return df, X_train, X_val, X_test, y_train, y_val, y_test 


def build_model(X_train, learning_rate=0.001, loss='binary_crossentropy', model_metrics=['accuracy']):
    # Réseau avec 2 couches cachées de 64 neurones chacune
    # et une couche de sortie avec activation softmax pour classification
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Définition de la fonction de perte, de l'optimiseur et des métriques
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=model_metrics
    )

    model.summary()

    return model

def evaluate_model_metrics(model, X_test, y_test, threshold=0.5):
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calcul des métriques
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Affichage des résultats
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    
    # Retourner les métriques dans un dictionnaire
    metrics = {
        'roc_auc': auc,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }
    
    return metrics

if __name__ == "__main__":
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU:0'
    print("GPUs disponibles :", tf.config.list_physical_devices('GPU'))
    df, X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

    epochs=50
    batch_size=16
    verbose=1
    learning_rate=0.001
    loss='binary_crossentropy'
    model_metrics=['accuracy']
    optimizer='adam'


    model = build_model(X_train,learning_rate, loss, model_metrics)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    mlflow_data = mlfs.Mlflow_dict(
        X_test=X_test,
        y_test=y_test,
        params={
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'optimizer': optimizer
        },
        tags={
            'experiment_name': 'Classification binaire churn',
            'run_name': 'model_keras_v0',
            'model_type': 'tensor_flow'
        }
    )

    mlfs.log_dagshub(mlflow_data, model)




