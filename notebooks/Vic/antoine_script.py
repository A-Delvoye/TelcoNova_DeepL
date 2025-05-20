import pandas as pd
import missingno as msno
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Chargement du dataset
data = pd.read_csv('/home/antoine/Documents/Projets_DEV_IA/Brief_Deep_Learning_01/TelcoNova_DeepL/data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Nettoyage et conversion des types
df = data.copy()

df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

df["gender"]= data["gender"].apply(lambda x: 0 if x ==  'Male' else 1)
for col in ["Partner", "PhoneService", 'PaperlessBilling', 'Churn', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'Dependents','OnlineSecurity']:
    data[col]= data[col].apply(lambda x: 0 if x ==  'No' else 1)

# Encodage binaire manuel
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'gender']

for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0})

# Encodage des colonnes catégorielles restantes
df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'Contract', 'PaymentMethod'], drop_first=True)


# # %%
# df['Partner'] = LabelEncoder().fit_transform(df['Partner'])
# df['Dependents'] = LabelEncoder().fit_transform(df['Dependents'])
# df['PhoneService'] = LabelEncoder().fit_transform(df['PhoneService'])
# df['OnlineSecurity'] = LabelEncoder().fit_transform(df['OnlineSecurity'])
# df['OnlineBackup'] = LabelEncoder().fit_transform(df['OnlineBackup'])
# df['DeviceProtection'] = LabelEncoder().fit_transform(df['DeviceProtection'])
# df['TechSupport'] = LabelEncoder().fit_transform(df['TechSupport'])
# df['StreamingTV'] = LabelEncoder().fit_transform(df['StreamingTV'])
# df['StreamingMovies'] = LabelEncoder().fit_transform(df['StreamingMovies'])
# df['PaperlessBilling'] = LabelEncoder().fit_transform(df['PaperlessBilling'])
# df['Churn'] = LabelEncoder().fit_transform(df['Churn'])

# Suppression de l'identifiant client
df.drop(columns=['customerID'], inplace=True)

# Séparation des features et de la cible
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Construction du modèle
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # sortie binaire
])

# Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=1, class_weight={0:1, 1:3})

# Évaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")

# Rapport de classification
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Matrice de confusion")

# Sauvegarde dans le dossier output
plt.tight_layout()
plt.savefig("output/02_confusion_matrix.png")
plt.close()

# Courbes d’apprentissage
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Crée le dossier output s'il n'existe pas
os.makedirs("output", exist_ok=True)

plt.savefig("output/01.png")


print(f"Min probability: {np.min(y_pred_probs):.4f}")
print(f"Max probability: {np.max(y_pred_probs):.4f}")