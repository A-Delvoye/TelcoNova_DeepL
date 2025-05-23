import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime
import os
import src.mlflow_script as mlfs

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

def build_model(X_train, learning_rate=0.001):
    model = Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')  # Binaire
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision'), 'accuracy']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
    # Répertoires
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = "models/best_model.keras"
    os.makedirs("models", exist_ok=True)

    # Callbacks
    early_stop = EarlyStopping(monitor='val_recall', patience=10, mode='max', restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_recall', mode='max',
                                 save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint, tensorboard]
    )

    return history


if __name__ == "__main__":
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU:0'
    # Prétraitement
    df, X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

    # Construction du modèle
    model = build_model(X_train)

    # Entraînement avec TensorBoard
    history = train_model(model, X_train, y_train, X_val, y_val)

    epochs=30
    batch_size=16
    verbose=1
    learning_rate=0.001
    loss='binary_crossentropy'
    model_metrics=['accuracy']
    optimizer='adam'

    mlflow_data = mlfs.Mlflow_dict(
        X_test=X_test,
        y_test=y_test,
        params={
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'optimizer': optimizer,
            'loss': loss
        },
        tags={
            'experiment_name': 'Classification binaire churn',
            'run_name': f"model_keras_{loss}_v0",
            'model_type': 'tensor_flow'
        }
    )

    mlfs.log_dagshub(mlflow_data, model)
