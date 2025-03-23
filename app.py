import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from scipy.io import arff
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling

# Argument Parser for CLI execution
parser = argparse.ArgumentParser(description="Run Improved AdaBoost on Seismic-Bumps dataset")
parser.add_argument("--debug", action="store_true", help="Enable debugging mode")
args = parser.parse_args()

DEBUG = args.debug
def debug_print(message):
    if DEBUG:
        print(message)

# Load ARFF file
debug_print("Loading dataset from .arff file...")
data, meta = arff.loadarff("dataset/seismic-bumps.arff")
df = pd.DataFrame(data)

# Decode byte-encoded categorical variables (if any)
df = df.apply(lambda col: col.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x) if col.dtype == 'object' else col)

# Print dataset info
debug_print(f"Dataset shape: {df.shape}")
debug_print(df.head())

# Convert categorical features into numeric
debug_print("Processing categorical features...")
categorical_columns = ['seismic', 'seismoacoustic', 'shift', 'ghazard']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Ensure target variable is numeric
debug_print("Mapping target variable...")
df['class'] = df['class'].astype(int).map({0: -1, 1: 1})

# Handle missing values
debug_print("Handling missing values...")
df.dropna(inplace=True)

if df.empty:
    raise ValueError("Error: Processed dataset is empty after preprocessing!")

# Define feature matrix and target variable
X = df.drop(columns=['class'])
y = df['class']
debug_print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

debug_print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Apply SMOTE to balance dataset
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
debug_print(f"After SMOTE - Training samples: {X_train.shape[0]}, Class distribution: {np.bincount(y_train + 1)}")

# Improved AdaBoost Implementation
class ImprovedAdaBoost:
    def __init__(self, n_estimators=100, k=0.05, b=1.0):
        self.n_estimators = n_estimators
        self.k = k
        self.b = b
        self.alphas = []
        self.models = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        D = np.ones(n_samples) / n_samples  # Initialize sample weights
        
        # Increase weight of minority class
        D[y == 1] *= 5
        D /= np.sum(D)

        for t in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=D)
            y_pred = model.predict(X)
            err = np.sum(D * (y_pred != y)) / np.sum(D)
            
            if err > 0.5:
                continue

            alpha_t = 0.5 * np.log((1 - err) / err) + self.k * np.exp(self.b * (2 * err - 1))
            D *= np.exp(-alpha_t * y * y_pred)
            D /= np.sum(D)
            
            self.models.append(model)
            self.alphas.append(alpha_t)
            
            debug_print(f"Iteration {t+1}/{self.n_estimators}: Error = {err:.4f}, Alpha = {alpha_t:.4f}")
    
    def predict(self, X):
        final_pred = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final_pred += alpha * model.predict(X)
        return np.sign(final_pred)

# Train Improved AdaBoost
debug_print("Training Improved AdaBoost...")
improved_ada = ImprovedAdaBoost(n_estimators=100, k=0.05, b=1.0)
improved_ada.fit(X_train, y_train)

# Make predictions
debug_print("Making predictions...")
y_pred = improved_ada.predict(X_test)

# Model Evaluation
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    
    debug_print("\nImproved AdaBoost Performance:")
    debug_print(f"Accuracy: {accuracy:.4f}")
    debug_print(f"Precision: {precision:.4f}")
    debug_print(f"Recall: {recall:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Non-Hazardous", "Hazardous"], 
                yticklabels=["Non-Hazardous", "Hazardous"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for Improved AdaBoost")
    plt.savefig("confusion_matrix.png")  # Save plot for CLI execution
    
    debug_print("\nClassification Report:\n" + classification_report(y_test, y_pred, zero_division=1))

# Evaluate Model
evaluate_model(y_test, y_pred)

debug_print("Script execution completed successfully.")
