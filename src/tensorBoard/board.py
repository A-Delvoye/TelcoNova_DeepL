import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

from dotenv import load_dotenv
load_dotenv()

# Configuration MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = "A-Delvoye"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/A-Delvoye/TelcoNova_DeepL.mlflow"
import mlflow
import src.mlflow_script as mlfs


def make_preprocess_pipeline(numerical_features, categorical_features):
    """Pipeline de préprocessing amélioré"""
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Médiane plus robuste aux outliers
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Plus explicite
    )

    return preprocessor


def advanced_feature_engineering(df):
    """Ingénierie de features avancée"""
    df = df.copy()
    
    # Conversion des variables booléennes
    df["gender"] = df["gender"].map({'Male': 0, 'Female': 1})
    bool_cols = ["Partner", "PhoneService", 'PaperlessBilling', "Dependents", 'Churn']
    for col in bool_cols:
        df[col] = df[col].map({'No': 0, 'Yes': 1})
    
    # Nettoyage TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Features d'ingénierie
    df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 pour éviter division par 0
    df['ChargesPerService'] = df['MonthlyCharges'] / (
        df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
            lambda x: sum([1 for val in x if val not in ['No', 'No internet service']]), axis=1) + 1
    )
    
    # Segmentation de la tenure
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                              labels=['0-12', '13-24', '25-48', '49-72'])
    
    # Segmentation des charges
    df['ChargeGroup'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df


def preprocess_data(df, test_size=0.2, val_size=0.2):
    """Préprocessing des données amélioré"""
    
    # Ingénierie de features
    df = advanced_feature_engineering(df)
    
    # Définition des features
    numerical_features = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'AvgMonthlyCharges', 'ChargesPerService'
    ]
    
    categorical_features = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'TenureGroup', 'ChargeGroup'
    ]
    
    target = 'Churn'
    
    # Suppression des lignes avec des valeurs manquantes critiques
    df.dropna(subset=[target], inplace=True)
    df.set_index("customerID", inplace=True)
    
    X = df[numerical_features + categorical_features]
    y = df[target]
    
    # Division stratifiée des données
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42, stratify=y_train_val
    )
    
    # Préprocessing
    preprocessor = make_preprocess_pipeline(numerical_features, categorical_features)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    return (df, X_train_processed, X_val_processed, X_test_processed, 
            y_train, y_val, y_test, preprocessor)


def build_improved_model(input_shape, learning_rate=0.001, dropout_rate=0.3, 
                        l1_reg=0.01, l2_reg=0.01):
    """Modèle neural network amélioré avec régularisation"""
    
    model = Sequential([
        Input(shape=(input_shape,)),
        
        # Première couche avec plus de neurones
        Dense(256, activation='relu', 
              kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Deuxième couche
        Dense(128, activation='relu',
              kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(dropout_rate * 0.8),
        
        # Troisième couche
        Dense(64, activation='relu',
              kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(dropout_rate * 0.6),
        
        # Quatrième couche plus petite pour la finalisation
        Dense(32, activation='relu',
              kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(dropout_rate * 0.4),
        
        # Couche de sortie
        Dense(1, activation='sigmoid')
    ])
    
    # Optimiseur avec décroissance du learning rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def get_class_weights(y_train):
    """Calcul des poids de classe pour gérer le déséquilibre"""
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, class_weights))


def train_improved_model(model, X_train, y_train, X_val, y_val, 
                        epochs=150, batch_size=32, patience=15):
    """Entraînement amélioré avec callbacks optimisés"""
    
    # Répertoires
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    checkpoint_path = "models/best_churn_model.keras"
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs/fit", exist_ok=True)
    
    # Poids de classe pour gérer le déséquilibre
    class_weights = get_class_weights(y_train)
    
    # Callbacks améliorés
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=patience,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Entraînement
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """Évaluation complète du modèle"""
    
    # Prédictions
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Métriques
    auc_score = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    print("=== ÉVALUATION DU MODÈLE ===")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    print("\nMatrice de confusion:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'auc_score': auc_score,
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }


def main():
    """Fonction principale d'exécution"""
    
    # Chargement des données
    print("Chargement des données...")
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Configuration GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Préprocessing
    print("Préprocessing des données...")
    (df, X_train, X_val, X_test, y_train, y_val, y_test, preprocessor) = preprocess_data(df)
    
    print(f"Forme des données:")
    print(f"  - Train: {X_train.shape}")
    print(f"  - Validation: {X_val.shape}")
    print(f"  - Test: {X_test.shape}")
    
    # Construction du modèle
    print("Construction du modèle...")
    model = build_improved_model(
        input_shape=X_train.shape[1],
        learning_rate=0.001,
        dropout_rate=0.3,
        l1_reg=0.01,
        l2_reg=0.01
    )
    
    model.summary()
    
    # Entraînement
    print("Entraînement du modèle...")
    history = train_improved_model(
        model, X_train, y_train, X_val, y_val,
        epochs=150, batch_size=32, patience=15
    )
    
    # Évaluation
    print("Évaluation du modèle...")
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    # Logging MLflow avec votre script existant
    print("Logging dans MLflow...")
    mlflow_data = mlfs.Mlflow_dict(
        X_test=X_test,
        y_test=y_test,
        y_pred=evaluation_results['y_pred'],
        y_pred_proba=evaluation_results['y_pred_proba'],
        roc_auc=evaluation_results['auc_score'],
        f1_score=evaluation_results['f1_score'],
        recall=evaluation_results['recall'],
        precision=evaluation_results['precision'],
        params={
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 150,
            'optimizer': 'adam',
            'loss': 'binary_crossentropy',
            'dropout_rate': 0.3,
            'l1_reg': 0.01,
            'l2_reg': 0.01,
            'patience': 15,
            'architecture': '256-128-64-32-1',
            'class_weight': 'balanced'
        },
        metrics={
            'val_auc': max(history.history.get('val_auc', [0])),
            'val_loss': min(history.history.get('val_loss', [float('inf')])),
            'val_accuracy': max(history.history.get('val_accuracy', [0])),
            'final_epoch': len(history.history.get('loss', [])),
            'best_threshold': 0.5
        },
        tags={
            'experiment_name': 'Classification binaire churn - Amélioré',
            'run_name': f"improved_model_auc_{evaluation_results['auc_score']:.4f}",
            'model_type': 'tensorflow',
            'version': 'v2.0',
            'feature_engineering': 'advanced',
            'regularization': 'l1_l2',
            'architecture_type': 'deep_neural_network'
        },
        input_example=X_test[:5]  # 5 exemples pour la documentation
    )
    
    # Utilisation de votre fonction log_dagshub
    run_id = mlfs.log_dagshub(mlflow_data, model)
    
    print(f"Entraînement terminé! MLflow Run ID: {run_id}")
    return model, history, evaluation_results


if __name__ == "__main__":
    model, history, evaluation_results = main()