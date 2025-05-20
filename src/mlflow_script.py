import os
import tempfile
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import tensorflow as tf
import io
from dotenv import load_dotenv
load_dotenv()

# Vérifier si PyTorch est disponible
try:
    import torch
    import torch.onnx
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

@dataclass
class Mlflow_dict:
    """Classe qui définit la structure des données à logger dans MLflow."""
    
    # Données de test (peuvent être fournies au lieu des métriques calculées)
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_pred_proba: Optional[np.ndarray] = None
    
    # Métriques (peuvent être calculées automatiquement ou fournies directement)
    roc_auc: Optional[float] = None
    f1_score: Optional[float] = None
    recall: Optional[float] = None
    precision: Optional[float] = None
    
    # Paramètres du modèle (optionnels)
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Métriques additionnelles (optionnelles)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Artefacts additionnels (optionnels)
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    # Tags (optionnels)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Exemple d'input pour le logging du modèle (obligatoire)
    input_example: Any = None

def calculate_metrics(X_test, y_test, model, threshold=0.5):
    """
    Calcule toutes les métriques nécessaires à partir d'un modèle et des données de test.
    
    Args:
        X_test: Données de test
        y_test: Vraies étiquettes
        model: Modèle TensorFlow ou PyTorch
        threshold: Seuil pour la classification binaire (défaut: 0.5)
        
    Returns:
        Un dictionnaire avec toutes les métriques et prédictions
    """
    # Vérifier si le modèle est un modèle PyTorch ou TensorFlow
    if PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            if isinstance(X_test, np.ndarray):
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            else:
                X_test_tensor = X_test
                
            y_pred_proba = model(X_test_tensor).numpy()
            
    elif isinstance(model, tf.keras.Model):
        y_pred_proba = model.predict(X_test).ravel()
    else:
        raise TypeError("Le modèle doit être un modèle TensorFlow ou PyTorch")
    
    # Conversion des prédictions probabilistes en classes binaires
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calcul des métriques
    metrics_dict = {
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1_score': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred)
    }
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    metrics_dict['confusion_matrix'] = cm
    
    # Calculer les points de la courbe ROC pour visualisation ultérieure
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    metrics_dict['roc_curve'] = (fpr, tpr)
    
    return metrics_dict

def save_confusion_matrix(y_true, y_pred, labels=None):
    """
    Crée et sauvegarde la matrice de confusion sous forme d'image.
    
    Args:
        y_true: Les étiquettes réelles
        y_pred: Les prédictions du modèle
        labels: Les noms des classes (optionnel)
    
    Returns:
        Le chemin vers l'image temporaire
    """
    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Créer une figure pour la matrice de confusion
    plt.figure(figsize=(10, 8))
    
    # Déterminer les labels
    if labels is None:
        labels = ['Négatif', 'Positif'] if cm.shape[0] == 2 else [f'Classe {i}' for i in range(cm.shape[0])]
    
    # Créer le heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Prédiction')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion')
    
    # Sauvegarder la figure dans un fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name)
    plt.close()
    
    return temp_file.name

def save_roc_curve(fpr, tpr, roc_auc):
    """
    Crée et sauvegarde la courbe ROC sous forme d'image.
    
    Args:
        fpr: Taux de faux positifs
        tpr: Taux de vrais positifs
        roc_auc: Valeur AUC-ROC
    
    Returns:
        Le chemin vers l'image temporaire
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Sauvegarder la figure dans un fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    plt.savefig(temp_file.name)
    plt.close()
    
    return temp_file.name

def log_tensorflow_model(model, input_example=None):
    """
    Sauvegarde un modèle TensorFlow dans MLflow.
    
    Args:
        model: Le modèle TensorFlow
        input_example: Un exemple d'entrée pour le modèle (optionnel)
    """
    # Log du modèle TensorFlow directement
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=None
    )

def log_pytorch_model(model, input_example=None):
    """
    Sauvegarde un modèle PyTorch dans MLflow.
    
    Args:
        model: Le modèle PyTorch
        input_example: Un exemple d'entrée pour le modèle (optionnel)
    """
    if not PYTORCH_AVAILABLE:
        print("PyTorch n'est pas disponible dans l'environnement.")
        return
        
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Log du modèle PyTorch
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=None
    )

def log_dagshub(mlflow_dict: Mlflow_dict, model, threshold=0.5):
    """
    Configure MLflow avec DagsHub et log les métriques, paramètres et artefacts.
    
    Args:
        mlflow_dict: Instance de Mlflow_dict contenant toutes les données à logger
        model: Le modèle à sauvegarder (TensorFlow ou PyTorch)
        threshold: Seuil pour la classification binaire (défaut: 0.5)
    """
    # Calcul automatique des métriques si elles ne sont pas fournies
    metrics_computed = False
    
    # Si X_test et y_test sont fournis, mais pas les métriques, les calculer
    if mlflow_dict.X_test is not None and mlflow_dict.y_test is not None:
        if any(metric is None for metric in [mlflow_dict.roc_auc, mlflow_dict.f1_score, 
                                            mlflow_dict.recall, mlflow_dict.precision]):
            print("Calcul automatique des métriques...")
            metrics = calculate_metrics(mlflow_dict.X_test, mlflow_dict.y_test, model, threshold)
            
            # Mise à jour des métriques dans mlflow_dict
            mlflow_dict.roc_auc = metrics['roc_auc']
            mlflow_dict.f1_score = metrics['f1_score']
            mlflow_dict.recall = metrics['recall']
            mlflow_dict.precision = metrics['precision']
            
            # Si y_pred et y_pred_proba ne sont pas fournis, les ajouter
            if mlflow_dict.y_pred is None:
                mlflow_dict.y_pred = metrics['y_pred']
            if mlflow_dict.y_pred_proba is None:
                mlflow_dict.y_pred_proba = metrics['y_pred_proba']
            
            # Stocker les points de la courbe ROC pour visualisation
            fpr, tpr = metrics['roc_curve']
            metrics_computed = True
    
    # Vérification que toutes les métriques obligatoires sont présentes
    required_metrics = ['roc_auc', 'f1_score', 'recall', 'precision']
    missing_metrics = [metric for metric in required_metrics 
                      if getattr(mlflow_dict, metric) is None]
    
    if missing_metrics:
        raise ValueError(f"Métriques obligatoires manquantes: {', '.join(missing_metrics)}. "
                         f"Fournissez ces métriques ou les données X_test et y_test pour calcul automatique.")
    
    # Configuration de MLflow avec DagsHub
    os.environ["MLFLOW_TRACKING_USERNAME"] = "A-Delvoye"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
    os.environ["MLFLOW_TRACKING_URI"] = (
        "https://dagshub.com/A-Delvoye/TelcoNova_DeepL.mlflow"
    )
    
    # Configuration du nom de l'expérience
    experiment_name = mlflow_dict.tags.get('experiment_name', 'Default Experiment')
    mlflow.set_experiment(experiment_name)
    
    # Démarrer une nouvelle run
    with mlflow.start_run(run_name=mlflow_dict.tags.get('run_name', None)):
        # Log des métriques obligatoires
        mlflow.log_metric("roc_auc", mlflow_dict.roc_auc)
        mlflow.log_metric("f1_score", mlflow_dict.f1_score)
        mlflow.log_metric("recall", mlflow_dict.recall)
        mlflow.log_metric("precision", mlflow_dict.precision)
        
        # Log des métriques additionnelles
        for metric_name, metric_value in mlflow_dict.metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log des paramètres
        for param_name, param_value in mlflow_dict.params.items():
            mlflow.log_param(param_name, param_value)
        
        # Log des tags
        for tag_name, tag_value in mlflow_dict.tags.items():
            mlflow.set_tag(tag_name, tag_value)
        
        # Créer et logger la matrice de confusion comme artefact
        if mlflow_dict.y_test is not None and mlflow_dict.y_pred is not None:
            conf_matrix_path = save_confusion_matrix(mlflow_dict.y_test, mlflow_dict.y_pred)
            mlflow.log_artifact(conf_matrix_path, "confusion_matrix")
            os.unlink(conf_matrix_path)  # Supprimer le fichier temporaire
            
            # Logger également la courbe ROC si les données sont disponibles
            if mlflow_dict.y_test is not None and mlflow_dict.y_pred_proba is not None:
                if metrics_computed:
                    roc_curve_path = save_roc_curve(fpr, tpr, mlflow_dict.roc_auc)
                else:
                    fpr, tpr, _ = roc_curve(mlflow_dict.y_test, mlflow_dict.y_pred_proba)
                    roc_curve_path = save_roc_curve(fpr, tpr, mlflow_dict.roc_auc)
                    
                mlflow.log_artifact(roc_curve_path, "roc_curve")
                os.unlink(roc_curve_path)  # Supprimer le fichier temporaire
        
        # Log des artefacts additionnels
        for artifact_name, artifact_path in mlflow_dict.artifacts.items():
            mlflow.log_artifact(artifact_path, artifact_name)
        
        # Définir l'exemple d'entrée s'il n'est pas fourni
        if mlflow_dict.input_example is None and mlflow_dict.X_test is not None:
            mlflow_dict.input_example = mlflow_dict.X_test[0:1]
        
        # Log du modèle en fonction de son type
        if isinstance(model, tf.keras.Model):
            log_tensorflow_model(model, mlflow_dict.input_example)
        elif PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            log_pytorch_model(model, mlflow_dict.input_example)
        else:
            print(f"Type de modèle non supporté: {type(model)}")
            
        print(f"Run terminée. Run ID: {mlflow.active_run().info.run_id}")
        
        # Retourner l'ID de la run pour référence
        return mlflow.active_run().info.run_id

# Exemple d'utilisation
"""
# Option 1: Laisser la fonction calculer automatiquement les métriques

# Créer l'objet Mlflow_dict avec seulement les données de test et le modèle
mlflow_data = Mlflow_dict(
    X_test=X_test,
    y_test=y_test,
    params={
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 20,
        'optimizer': 'adam'
    },
    metrics={
        'accuracy': 0.88,
        'val_loss': 0.23
    },
    tags={
        'experiment_name': 'Classification binaire',
        'run_name': 'model_v1',
        'model_type': 'neural_network'
    }
    # input_example sera automatiquement défini à partir de X_test[0:1]
)

# Option 2: Fournir directement toutes les métriques précalculées
mlflow_data = Mlflow_dict(
    roc_auc=0.95,
    f1_score=0.87,
    recall=0.89,
    precision=0.85,
    y_test=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    params={
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 20,
        'optimizer': 'adam'
    },
    metrics={
        'accuracy': 0.88,
        'val_loss': 0.23
    },
    tags={
        'experiment_name': 'Classification binaire',
        'run_name': 'model_v1',
        'model_type': 'neural_network'
    },
    input_example=X_test[0:1]  # Un exemple d'entrée pour le modèle
)

# Logger le modèle et les métriques dans MLflow/DagsHub
run_id = log_dagshub(mlflow_data, model)
"""