# Import necessary libraries for data handling, machine learning, and evaluation
import pandas as pd  # For data manipulation (loading, cleaning)
from sklearn.ensemble import (  # Tree-based classifiers
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)  # Logistic Regression classifier
from sklearn.metrics import (
    accuracy_score,  # For performance evaluation
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
)  # To split data into training and testing sets
from sklearn.preprocessing import StandardScaler  # To normalize feature values
from sklearn.svm import SVC  # Support Vector Machine classifier

# Define the file path to your dataset
file_path = "E:/workspace/Manoj_Portfolio/data_sets/u.csv"
# Load the data, handling potential errors
try:
    # Load tab-separated data (MovieLens format: user_id, item_id, rating, timestamp)
    # No headers in the file, so we name the columns manually
    data = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
except FileNotFoundError:
    print(f"Error: '{file_path}' not found.")  # If file doesn’t exist
    exit()
except Exception as e:
    print(
        f"Error loading file: {e}"
    )  # For other loading issues (e.g., bad format)
    exit()

# --- Data Preprocessing ---
# Remove rows with missing values (though MovieLens data is usually clean)
data = data.dropna()
# Remove duplicate rows (e.g., if a user rated the same movie twice with the same timestamp)
data = data.drop_duplicates()
# Print basic info to understand the data
print(f"Dataset shape: {data.shape}")  # Number of rows and columns
print(
    f"Unique ratings: {data['rating'].unique()}"
)  # Show unique rating values (e.g., 1, 2, 3, 4, 5)

# Define features (X) and target (y)
# Features: user_id, item_id, timestamp (what we’ll use to predict ratings)
X = data[["user_id", "item_id", "timestamp"]]
# Target: rating (what we want to predict, values 1 to 5)
y = data["rating"]

# Normalize features so they’re on the same scale (important for SVM and Logistic Regression)
scaler = StandardScaler()  # Create a scaler object
X = scaler.fit_transform(X)  # Fit and transform the features (mean=0, std=1)

# Split data into training (80%) and testing (20%) sets
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Define Classifiers ---
# Create a dictionary of classifiers to compare
classifiers = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),  # 100 trees, reproducible
    "SVM": SVC(
        kernel="rbf", C=1, gamma="scale"
    ),  # RBF kernel, default regularization
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),  # Max 1000 iterations to ensure convergence
    "Gradient Boost": GradientBoostingClassifier(
        max_depth=3, n_estimators=100, random_state=42
    ),  # 100 boosted trees, depth 3
}

# --- Evaluate Classifiers ---
results = []  # List to store results for each classifier
for clf_name, clf in classifiers.items():  # Loop through each classifier
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    # Predict on training data (to check overfitting)
    y_train_pred = clf.predict(X_train)
    # Predict on testing data (to evaluate generalization)
    y_test_pred = clf.predict(X_test)

    # Calculate errors (1 - accuracy) for training and testing sets
    training_error = 1 - accuracy_score(y_train, y_train_pred)
    testing_error = 1 - accuracy_score(y_test, y_test_pred)

    # Determine if the model is overfitting, underfitting, or a good fit
    # Tolerance of 0.05 allows small differences to still be "Good Fit"
    error_eval = (
        "Good Fit"
        if abs(training_error - testing_error) < 0.05
        else (
            "Overfitting" if training_error < testing_error else "Underfitting"
        )
    )

    # Get detailed metrics (precision, recall, F1) for the test set
    report = classification_report(y_test, y_test_pred, output_dict=True)
    # Store results: classifier name, errors, evaluation, and report
    results.append(
        [clf_name, training_error, testing_error, error_eval, report]
    )

# --- Display Results Table ---
# Convert results to a DataFrame for nice formatting
result_df = pd.DataFrame(
    results,
    columns=[
        "Classifier",
        "Training Error",
        "Testing Error",
        "Error Evaluation",
        "Report",
    ],
)
print("\nClassifier Performance:")
# Print only the main columns (excluding the report for now), no index for cleaner output
print(
    result_df[
        ["Classifier", "Training Error", "Testing Error", "Error Evaluation"]
    ].to_string(index=False)
)

# --- Suggest Improvements ---
print("\nImprovement Suggestions:")
for _, row in result_df.iterrows():  # Loop through each row in the results
    clf_name, report = (
        row["Classifier"],
        row["Report"],
    )  # Get classifier name and its report
    macro_avg = report.get(
        "macro avg", {}
    )  # Get macro-average metrics (averages across all classes)

    # Check if macro_avg exists (safety check)
    if not macro_avg:
        print(
            f"Warning: No macro avg metrics for {clf_name}. Skipping suggestions."
        )
        continue

    # Identify areas needing improvement (threshold: 0.8)
    areas_to_improve = []
    if macro_avg.get("precision", 1.0) < 0.8:  # If precision is below 0.8
        areas_to_improve.append("precision")
    if macro_avg.get("recall", 1.0) < 0.8:  # If recall is below 0.8
        areas_to_improve.append("recall")

    # Define possible solutions for each area
    solutions = {}
    for area in areas_to_improve:
        if area == "precision":
            solutions[area] = [
                "Tune hyperparameters (e.g., increase C in SVM/Logistic, reduce max_depth in trees).",  # Adjust model complexity
                "Adjust prediction threshold via predict_proba (e.g., increase for higher precision).",  # Shift decision boundary
            ]
        elif area == "recall":
            solutions[area] = [
                'Use class_weight="balanced" to prioritize minority classes.',  # Handle imbalanced ratings
                "Lower the prediction threshold to capture more positives.",  # Shift decision boundary
            ]

    # Print the suggestions for this classifier
    print(f"\nModel: {clf_name}")
    # Display macro-average metrics with 2 decimal places (or N/A if missing)
    print(
        f"Macro Avg Precision: {macro_avg.get('precision', 'N/A'):.2f}, Recall: {macro_avg.get('recall', 'N/A'):.2f}, F1: {macro_avg.get('f1-score', 'N/A'):.2f}"
    )
    for (
        area,
        solution,
    ) in solutions.items():  # Loop through areas and their solutions
        print(f"Area to improve: {area}")
        for s in solution:  # Print each solution
            print(f"- {s}")
