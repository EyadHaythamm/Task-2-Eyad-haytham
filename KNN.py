import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load and clean dataset
df = pd.read_csv("KNNAlgorithmDataset.csv")

if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Handle any empty trailing columns caused by trailing commas in the CSV
unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
df = df.drop(columns=unnamed_cols)

# Encode categorical target ('M' -> 1, 'B' -> 0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Evaluate performance on test set
y_pred = knn.predict(X_test_scaled)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4%}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
