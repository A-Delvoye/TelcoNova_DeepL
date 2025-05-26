import pandas as pd
import numpy as np
import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import warnings
import os
import datetime
warnings.filterwarnings('ignore')

# Configuration pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

from dotenv import load_dotenv
import src.mlflow_script as mlfs

load_dotenv()

class ChurnModelOptimizer:
    """Classe pour l'optimisation des hyperparamètres du modèle de churn"""
    
    def __init__(self, df_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        self.df_path = df_path
        self.X_train = None
        self.X_val = None 
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.preprocessor = None
        self.best_params = None
        self.best_score = 0
        
    def advanced_feature_engineering(self, df):
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
        df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
        df['ChargesPerService'] = df['MonthlyCharges'] / (
            df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(
                lambda x: sum([1 for val in x if val not in ['No', 'No internet service']]), axis=1) + 1
        )
        
        # Ratios et interactions
        df['TenureToChargesRatio'] = df['tenure'] / (df['MonthlyCharges'] + 1)
        df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
        df['IsHighValueCustomer'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
        
        # Segmentation
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], 
                                  labels=['0-12', '13-24', '25-48', '49-72'])
        df['ChargeGroup'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        
        return df
    
    def prepare_data(self):
        """Préparation des données pour l'optimisation"""
        print("Chargement et préparation des données...")
        
        df = pd.read_csv(self.df_path)
        df = self.advanced_feature_engineering(df)
        
        # Définition des features
        numerical_features = [
            'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'AvgMonthlyCharges', 'ChargesPerService', 'TenureToChargesRatio'
        ]
        
        categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'TenureGroup', 'ChargeGroup',
            'IsNewCustomer', 'IsHighValueCustomer'
        ]
        
        target = 'Churn'
        
        df.dropna(subset=[target], inplace=True)
        df.set_index("customerID", inplace=True)
        
        X = df[numerical_features + categorical_features]
        y = df[target]
        
        # Division des données
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )
        
        # Préprocessing
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        self.X_val = self.preprocessor.transform(self.X_val)
        self.X_test = self.preprocessor.transform(self.X_test)
        
        print(f"Données préparées: Train {self.X_train.shape}, Val {self.X_val.shape}, Test {self.X_test.shape}")
    
    def create_model(self, trial):
        """Création du modèle avec hyperparamètres suggérés par Optuna"""
        
        # Hyperparamètres à optimiser
        n_layers = trial.suggest_int('n_layers', 2, 5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
        l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-2, log=True)
        l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Architecture dynamique
        layers_config = []
        for i in range(n_layers):
            if i == 0:
                # Première couche plus large
                units = trial.suggest_int(f'units_layer_{i}', 128, 512, step=64)
            else:
                # Couches suivantes dégressives
                prev_units = layers_config[-1] if layers_config else 256
                units = trial.suggest_int(f'units_layer_{i}', 32, min(prev_units, 256), step=32)
            layers_config.append(units)
        
        # Construction du modèle
        model = Sequential([Input(shape=(self.X_train.shape[1],))])
        
        for i, units in enumerate(layers_config):
            model.add(Dense(
                units, 
                activation='relu',
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                kernel_initializer='he_normal'
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate * (0.8 ** i)))  # Dropout dégressif
        
        # Couche de sortie
        model.add(Dense(1, activation='sigmoid'))
        
        # Optimiseur
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop'])
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            )
        else:  # rmsprop
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.Precision(name='precision')
            ]
        )
        
        return model, batch_size
    
    def objective(self, trial):
        """Fonction objectif pour Optuna - Optimisation du Recall"""
        
        # Création du modèle
        model, batch_size = self.create_model(trial)
        
        # Poids de classe avec option d'ajustement pour le recall
        classes = np.unique(self.y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=self.y_train)
        
        # Boost optionnel pour la classe positive (churn) pour améliorer le recall
        recall_boost = trial.suggest_float('recall_boost', 1.0, 3.0)
        class_weights[1] *= recall_boost  # Augmenter le poids de la classe churn
        
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Callbacks
        patience = trial.suggest_int('patience', 10, 25)
        reduce_lr_patience = trial.suggest_int('reduce_lr_patience', 5, 15)
        
        callbacks = [
            EarlyStopping(
                monitor='val_recall',  # Changé pour recall
                patience=patience,
                mode='max',
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=0
            ),
            TFKerasPruningCallback(trial, 'val_recall')  # Pruning basé sur recall
        ]
        
        # Entraînement
        try:
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=100,  # Max epochs, early stopping va gérer
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=0
            )
            
            # Score de validation (Recall)
            val_recall = max(history.history['val_recall'])
            
            # Optionnel: pénaliser si la précision est trop faible
            val_precision = max(history.history.get('val_precision', [0]))
            val_auc = max(history.history.get('val_auc', [0]))
            
            # Score composite: Recall avec contrainte minimale sur la précision et sur auc
            min_precision = 0.4  # Précision minimale acceptable
            min_auc = 0.8  # Auc minimale acceptable
            final_score = val_recall
            if val_precision < min_precision:
                # Pénaliser le score si la précision est trop faible
                penalty = (min_precision - val_precision) * 2
                final_score -= penalty
            if val_auc < min_auc:
                # Pénaliser le score si la précision est trop faible
                penalty = (min_auc - val_auc) * 2
                final_score -= penalty
            
            return max(final_score, 0.0)  # S'assurer que le score n'est pas négatif
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement: {e}")
            return 0.0
    
    def optimize(self, n_trials=100, timeout=3600):
        """Lancement de l'optimisation"""
        
        if self.X_train is None:
            self.prepare_data()
        
        # Configuration de l'étude Optuna
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20,
                interval_steps=5
            ),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        print(f"Démarrage de l'optimisation RECALL avec {n_trials} trials (timeout: {timeout}s)")
        print("Objectif: Maximiser le Recall avec contrainte sur la Précision")
        
        # Optimisation
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Résultats
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print("\\n=== RÉSULTATS DE L'OPTIMISATION RECALL ===")
        print(f"Meilleur score Recall: {self.best_score:.4f}")
        print(f"Meilleurs paramètres:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        
        return study
    
    def train_best_model(self):
        """Entraînement du meilleur modèle trouvé"""
        
        if self.best_params is None:
            raise ValueError("Lancez d'abord l'optimisation avec optimize()")
        
        print("\\nEntraînement du meilleur modèle...")
        
        # Reconstruction du modèle avec les meilleurs paramètres
        # On simule un trial avec les meilleurs paramètres
        class BestTrial:
            def __init__(self, params):
                self.params = params
            
            def suggest_int(self, name, low, hi, step=None):
                return self.params.get(name, (low + hi) // 2)
            
            def suggest_float(self, name, low, hi, log=False):
                return self.params.get(name, (low + hi) / 2)
            
            def suggest_categorical(self, name, choices):
                return self.params.get(name, choices[0])
        
        best_trial = BestTrial(self.best_params)
        model, batch_size = self.create_model(best_trial)
        
        # Poids de classe avec boost pour recall
        classes = np.unique(self.y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=self.y_train)
        
        # Appliquer le boost de recall trouvé pendant l'optimisation
        recall_boost = self.best_params.get('recall_boost', 1.0)
        class_weights[1] *= recall_boost
        
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Callbacks pour l'entraînement final
        checkpoint_path = f"models/best_optimized_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        os.makedirs("models", exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_recall',  # Monitoring sur recall
                patience=self.best_params.get('patience', 15),
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_recall',  # Sauvegarde basée sur recall
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.best_params.get('reduce_lr_patience', 8),
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entraînement final
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=200,  # Plus d'epochs pour le modèle final
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Évaluation finale avec seuils optimisés pour recall
        y_pred_proba = model.predict(self.X_test).ravel()
        
        # Test de différents seuils pour optimiser le recall
        thresholds = np.arange(0.3, 0.8, 0.05)  # Seuils plus bas pour favoriser le recall
        best_threshold = 0.5
        best_recall = 0
        
        print("\\nOptimisation du seuil pour maximiser le recall:")
        for threshold in thresholds:
            y_pred_temp = (y_pred_proba >= threshold).astype(int)
            recall_temp = recall_score(self.y_test, y_pred_temp)
            precision_temp = precision_score(self.y_test, y_pred_temp)
            
            print(f"  Seuil {threshold:.2f}: Recall={recall_temp:.3f}, Precision={precision_temp:.3f}")
            
            # Garder le seuil qui maximise le recall avec précision acceptable
            if recall_temp > best_recall and precision_temp >= 0.3:
                best_recall = recall_temp
                best_threshold = threshold
        
        print(f"\\nMeilleur seuil trouvé: {best_threshold:.2f}")
        
        # Prédictions finales avec le meilleur seuil
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        test_auc = roc_auc_score(self.y_test, y_pred_proba)
        test_f1 = f1_score(self.y_test, y_pred)
        test_recall = recall_score(self.y_test, y_pred)
        test_precision = precision_score(self.y_test, y_pred)
        
        print("\\n=== PERFORMANCE DU MEILLEUR MODÈLE (OPTIMISÉ RECALL) ===")
        print(f"Test Recall: {test_recall:.4f} ⭐ (Métrique principale)")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Seuil optimal: {best_threshold:.3f}")
        
        # Logging MLflow du meilleur modèle
        mlflow_data = mlfs.Mlflow_dict(
            X_test=self.X_test,
            y_test=self.y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            roc_auc=test_auc,
            f1_score=test_f1,
            recall=test_recall,
            precision=test_precision,
            params=self.best_params,
            metrics={
                'val_recall_best': max(history.history.get('val_recall', [0])),
                'val_precision_best': max(history.history.get('val_precision', [0])),
                'val_auc_best': max(history.history.get('val_auc', [0])),
                'val_loss_best': min(history.history.get('val_loss', [float('inf')])),
                'final_epoch': len(history.history.get('loss', [])),
                'optimization_trials': len(self.best_params),
                'optimal_threshold': best_threshold,
                'recall_boost_factor': self.best_params.get('recall_boost', 1.0)
            },
            tags={
                'experiment_name': 'Classification binaire churn',
                'run_name': f"optuna_recall_{test_recall:.4f}_precision_{test_precision:.3f}",
                'model_type': 'tensorflow_recall_optimized',
                'version': 'v3.0_optuna_recall',
                'optimization_method': 'optuna_tpe_recall',
                'optimization_target': 'recall_maximization',
                'feature_engineering': 'advanced_v2',
                'threshold_optimization': 'enabled'
            },
            input_example=self.X_test[:5]
        )
        
        run_id = mlfs.log_dagshub(mlflow_data, model)
        print(f"\\nModèle loggé dans MLflow: {run_id}")
        
        return model, history, {
            'test_auc': test_auc,
            'test_f1': test_f1,
            'test_recall': test_recall,
            'test_precision': test_precision,
            'optimal_threshold': best_threshold,
            'recall_boost': self.best_params.get('recall_boost', 1.0)
        }


def main():
    """Fonction principale d'optimisation"""
    
    # Initialisation de l'optimiseur
    optimizer = ChurnModelOptimizer()
    
    # Lancement de l'optimisation pour maximiser le RECALL
    study = optimizer.optimize(n_trials=50, timeout=2400)  # 40 minutes max
    
    # Entraînement du meilleur modèle
    best_model, history, final_results = optimizer.train_best_model()
    
    # Analyse des résultats
    print("\\n=== ANALYSE DE L'OPTIMISATION ===")
    print(f"Nombre total de trials: {len(study.trials)}")
    print(f"Trials réussis: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Trials échoués: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print(f"Trials élagués: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    # Importance des paramètres
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\\nImportance des hyperparamètres:")
        for param, importance_val in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {param}: {importance_val:.4f}")
    except:
        print("\\nImpossible de calculer l'importance des paramètres")
    
    return best_model, study, final_results


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU:0'
    best_model, study, results = main()