
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
from pyngrok import ngrok
import subprocess

# assuming ml flow wil be running locally on 5000, ideally shd be a permanent host
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

# Load the dataset from your HuggingFace dataset repository
Xtrain_path = "hf://datasets/Shivam174/gltourism-prediction-data/Xtrain.csv"
Xtest_path = "hf://datasets/Shivam174/gltourism-prediction-data/Xtest.csv"
ytrain_path = "hf://datasets/Shivam174/gltourism-prediction-data/ytrain.csv"
ytest_path = "hf://datasets/Shivam174/gltourism-prediction-data/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)

# Define categorical and numerical features
categorical_features = ['TypeofContact', 'CityTier', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched']
numerical_features = ['Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch', 'MonthlyIncome']

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline with preprocessing and model
clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

# Set up MLflow experiment
mlflow.set_experiment("WellnessTourism_PurchasePrediction")

with mlflow.start_run():
    # Train model
    clf.fit(X_train, y_train)

    # Predict on test data
    y_pred = clf.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log model parameters (example: number of trees)
    mlflow.log_param("n_estimators", clf.named_steps['classifier'].n_estimators)

    # Log the model itself
    mlflow.sklearn.log_model(clf, "random_forest_model")
    print(f"Model trained with Accuracy: {acc:.4f}, F1: {f1:.4f}")

# Save the model locally
joblib.dump(clf, "best_tourism_model.joblib")

repo_id = "Shivam174/tourism-prediction-model"

# Create repo if it doesn't exist
create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

# Upload model file
api.upload_file(
    path_or_fileobj="best_tourism_model.joblib",
    path_in_repo="best_tourism_model.joblib",
    repo_id=repo_id,
    repo_type="model"
)
print("Model uploaded successfully.")
