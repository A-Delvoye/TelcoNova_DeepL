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
import torch
import torch.nn as nn

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

class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.hidden_layer1 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out

def build_model(X_train, y_train, X_test, y_test, learning_rate=0.001, num_epochs=30):
    # Vérifier la forme des données d'entrée
    input_dim = X_train.shape[1]  # Nombre de features
    
    # Déterminer le nombre de classes pour la sortie
    if len(y_train.unique()) > 2:
        output_dim = len(y_train.unique())  # Classification multi-classes
    else:
        output_dim = 1  # Classification binaire
    

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)


    # Initialiser le modèle
    model = NeuralNetworkClassificationModel(input_dim, output_dim)
    
    # Définir la fonction de perte selon le type de classification
    if output_dim > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimiseur
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Stocker les pertes pour visualisation
    train_losses = np.zeros(num_epochs)
    test_losses = np.zeros(num_epochs)
    
    # Boucle d'entraînement
    for epoch in range(num_epochs):
        # Passage en mode entraînement
        model.train()
        
        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Forward pass
        output_train = model(X_train)
        
        # S'assurer que les dimensions correspondent pour la fonction de perte
        if output_dim == 1:
            loss_train = criterion(output_train.squeeze(), y_train.float())
        else:
            loss_train = criterion(output_train, y_train)
        
        # Backward pass et mise à jour des poids
        loss_train.backward()
        optimizer.step()
        
        # Évaluation sur les données de test (sans calculer de gradients)
        with torch.no_grad():
            model.eval()
            output_test = model(X_test)
            
            if output_dim == 1:
                loss_test = criterion(output_test.squeeze(), y_test.float())
            else:
                loss_test = criterion(output_test, y_test)
        
        # Enregistrer les pertes
        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()
        
        # Afficher les pertes toutes les 5 époques
        if (epoch + 1) % 5 == 0:
            print(f'Époque [{epoch+1}/{num_epochs}], Perte train: {loss_train.item():.4f}, Perte test: {loss_test.item():.4f}')
        
    return model
    

if __name__ == "__main__":
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU:0'
    print("GPUs disponibles :", tf.config.list_physical_devices('GPU'))
    df, X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

    epochs=30
    verbose=1
    learning_rate=0.001
    loss='CrossEntropyLoss'
    optimizer='adam'


    model = build_model(X_train, y_train, X_test, y_test, learning_rate, epochs)

    mlflow_data = mlfs.Mlflow_dict(
        X_test=X_test,
        y_test=y_test,
        params={
            'learning_rate': learning_rate,
            'epochs': epochs,
            'optimizer': optimizer,
            "loss" : loss
        },
        tags={
            'experiment_name': 'Classification binaire churn',
            'run_name': 'model_torch_v0',
            'model_type': 'torch'
        }
    )

    mlfs.log_dagshub(mlflow_data, model)




