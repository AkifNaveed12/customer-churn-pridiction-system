"""
Customer Churn Prediction using Artificial Neural Network (ANN)
Use Case: Predict whether a telecom customer will leave the company (churn) or not

Dataset: Telco Customer Churn
Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Or use the direct link in the code below
"""

# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("CUSTOMER CHURN PREDICTION USING ARTIFICIAL NEURAL NETWORK (ANN)")
print("="*70)

# Step 2: Load the Dataset
print("\n[1] Loading Dataset...")
# You can download the CSV and place it in your directory, or use this URL
url = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(url)
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {df.shape}")
    print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
except:
    print("✗ Error loading dataset. Please download from Kaggle and update the path.")
    exit()

# Step 3: Data Exploration
print("\n[2] Data Exploration...")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nTarget Variable Distribution:")
print(df['Churn'].value_counts())

# Step 4: Data Preprocessing
print("\n[3] Data Preprocessing...")

# Remove customerID as it's not useful for prediction
df = df.drop('customerID', axis=1)

# Handle TotalCharges (convert to numeric and fill missing values)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert target variable to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Encode categorical variables
le = LabelEncoder()
categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

print(f"✓ Preprocessing completed!")
print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# Step 5: Split the Data
print("\n[4] Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Step 6: Feature Scaling
print("\n[5] Feature Scaling...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("✓ Features scaled using StandardScaler")

# Step 7: Build the ANN Model
print("\n[6] Building ANN Architecture...")
model = Sequential([
    # Input Layer + First Hidden Layer
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    
    # Second Hidden Layer
    Dense(32, activation='relu'),
    Dropout(0.3),
    
    # Third Hidden Layer
    Dense(16, activation='relu'),
    Dropout(0.2),
    
    # Output Layer
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n✓ ANN Model Architecture:")
model.summary()

# Step 8: Train the Model
print("\n[7] Training the Model...")
print("This may take a few minutes...")

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("\n✓ Training completed!")

# Step 9: Evaluate the Model
print("\n[8] Model Evaluation...")

# Predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Test Accuracy: {accuracy*100:.2f}%")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 10: Visualizations
print("\n[9] Generating Visualizations...")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Training History - Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Training History - Loss
axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
axes[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel('False Positive Rate')
axes[1, 1].set_ylabel('True Positive Rate')
axes[1, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc="lower right")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ann_churn_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'ann_churn_analysis.png'")
plt.show()

# Step 11: Make Predictions on New Data
print("\n[10] Making Predictions on Sample Data...")

# Example: Predict churn for first 5 test samples
sample_predictions = y_pred_prob[:5]
sample_actual = y_test.iloc[:5].values

print("\nSample Predictions:")
print("-" * 60)
for i, (pred, actual) in enumerate(zip(sample_predictions, sample_actual)):
    pred_label = "CHURN" if pred > 0.5 else "NO CHURN"
    actual_label = "CHURN" if actual == 1 else "NO CHURN"
    print(f"Customer {i+1}: Predicted = {pred_label} ({pred[0]:.2%}), Actual = {actual_label}")

# Step 12: Save the Model
print("\n[11] Saving the Model...")
model.save('customer_churn_ann_model.h5')
print("✓ Model saved as 'customer_churn_ann_model.h5'")

# Final Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Model Type: Artificial Neural Network (ANN)")
print(f"✓ Use Case: Customer Churn Prediction")
print(f"✓ Dataset: Telco Customer Churn")
print(f"✓ Total Samples: {len(df)}")
print(f"✓ Features: {X.shape[1]}")
print(f"✓ Test Accuracy: {accuracy*100:.2f}%")
print(f"✓ AUC Score: {roc_auc:.2f}")
print("="*70)
print("\n✓ Use case completed successfully!")
print("You can now present this to your instructor.")
print("\nFiles generated:")
print("  1. ann_churn_analysis.png (visualizations)")
print("  2. customer_churn_ann_model.h5 (trained model)")