import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")




# At this point, we need to confirm which column is our target variable
# For this code template, let's assume we've identified the target column as 'treatment'
# In practical implementation, we'd confirm this based on dataset inspection

# Data Preprocessing and Cleaning Function
def preprocess_data(df, target_column):
    """
    Preprocess the mental health survey data for modeling.

    Args:
        df: Original dataframe
        target_column: The column name for our prediction target

    Returns:
        Processed dataframe, feature names, target series
    """
    print("\n=== Data Preprocessing ===")

    # Make a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()

    # Ensure target column exists
    if target_column not in processed_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset")

    # Extract target variable
    y = processed_df[target_column]

    # Remove any irrelevant columns (like ID, timestamps, etc.)
    # This is a placeholder - you'll need to identify these columns in the real dataset
    columns_to_drop = ['Timestamp'] if 'Timestamp' in processed_df.columns else []
    columns_to_drop.append(target_column)  # Add target column to drop list

    # Drop unnecessary columns
    X = processed_df.drop(columns=columns_to_drop, errors='ignore')

    # Check for columns with high percentage of missing values
    missing_percentages = X.isnull().mean() * 100
    high_missing_cols = missing_percentages[missing_percentages > 50].index.tolist()
    print(f"Columns with >50% missing values that will be dropped: {high_missing_cols}")
    X = X.drop(columns=high_missing_cols, errors='ignore')

    # Get categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Handle missing values for numerical columns
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    # Handle missing values for categorical columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encode categorical columns with low cardinality
    for col in cat_cols:
        if X[col].nunique() < 10:  # Only one-hot encode if few unique values
            one_hot = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X, one_hot], axis=1)
            X = X.drop(columns=[col])
        else:
            # For high cardinality, we might want to use other approaches
            # For now, we'll drop these columns
            print(f"Dropping high cardinality column: {col} with {X[col].nunique()} unique values")
            X = X.drop(columns=[col])

    print(f"Processed dataframe shape: {X.shape}")
    return X, y


# Feature Engineering Function
def engineer_features(X, y):
    """
    Create new features and select the most important ones.

    Args:
        X: Feature dataframe
        y: Target series

    Returns:
        DataFrame with selected features
    """
    print("\n=== Feature Engineering ===")

    # This is where we would create new features based on domain knowledge
    # For example, we might combine related questions, create interaction terms, etc.
    # For now, we'll use a basic feature selection approach

    # Select top k features
    selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
    X_selected = selector.fit_transform(X, y)

    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]

    print(f"Selected {len(selected_features)} features")

    return X_selected, selected_features


def clean_data(df):
    """
    Clean the dataset by fixing outliers and standardizing values.

    Args:
        df: Original dataframe

    Returns:
        Cleaned dataframe
    """
    print("\n=== Data Cleaning ===")
    df_cleaned = df.copy()

    # Fix Age outliers
    print(f"Age range before cleaning: {df_cleaned['Age'].min()} to {df_cleaned['Age'].max()}")

    # Identify potential age outliers
    age_outliers = df_cleaned[(df_cleaned['Age'] > 100) | (df_cleaned['Age'] < 18)].shape[0]
    print(f"Found {age_outliers} age values outside the reasonable range (18-100)")

    # Replace unreasonable ages with NaN and then fill with median of valid ages
    valid_ages = df_cleaned[(df_cleaned['Age'] >= 18) & (df_cleaned['Age'] <= 100)]['Age']
    median_age = valid_ages.median()

    # Replace outliers with NaN and then fill with median
    df_cleaned.loc[(df_cleaned['Age'] > 100) | (df_cleaned['Age'] < 18), 'Age'] = np.nan
    df_cleaned['Age'] = df_cleaned['Age'].fillna(median_age)

    print(f"Age range after cleaning: {df_cleaned['Age'].min()} to {df_cleaned['Age'].max()}")

    # Standardize Gender values
    if 'Gender' in df_cleaned.columns:
        print("\nCleaning Gender values...")
        print(f"Unique gender values before cleaning: {df_cleaned['Gender'].nunique()}")

        # Create a mapping dictionary for gender standardization
        gender_mapping = {
            # Map male variations
            'male': 'Male',
            'm': 'Male',
            'M': 'Male',
            'Male': 'Male',
            'maile': 'Male',
            'Mal': 'Male',
            'Make': 'Male',
            'Man': 'Male',
            'msle': 'Male',
            'Mail': 'Male',
            'Malr': 'Male',
            'Cis Male': 'Male',
            'cis male': 'Male',
            'Cis Man': 'Male',

            # Map female variations
            'female': 'Female',
            'f': 'Female',
            'F': 'Female',
            'Female': 'Female',
            'Woman': 'Female',
            'woman': 'Female',
            'Femake': 'Female',
            'Cis Female': 'Female',
            'femail': 'Female',

            # Map non-binary/other variations
            'Trans-female': 'Non-binary/Other',
            'Trans woman': 'Non-binary/Other',
            'non-binary': 'Non-binary/Other',
            'Enby': 'Non-binary/Other',
            'fluid': 'Non-binary/Other',
            'Androgyne': 'Non-binary/Other',
            'Agender': 'Non-binary/Other',
        }

        # Apply mapping - for any value not in the mapping, keep the original value
        df_cleaned['Gender'] = df_cleaned['Gender'].apply(
            lambda x: gender_mapping.get(x, 'Other' if pd.notna(x) else np.nan)
        )

        # Fill NaN values with 'Other'
        df_cleaned['Gender'] = df_cleaned['Gender'].fillna('Other')

        print(f"Unique gender values after cleaning: {df_cleaned['Gender'].nunique()}")
        print(f"Gender distribution after cleaning: \n{df_cleaned['Gender'].value_counts()}")

    # Check and fix other potential data issues

    # Fix potential issues in self_employed (if present)
    if 'self_employed' in df_cleaned.columns:
        # Standardize Yes/No values
        df_cleaned['self_employed'] = df_cleaned['self_employed'].apply(
            lambda x: 'Yes' if str(x).lower() == 'yes' else 'No' if str(x).lower() == 'no' else x
        )

    # Fix potential issues in family_history (if present)
    if 'family_history' in df_cleaned.columns:
        # Standardize Yes/No values
        df_cleaned['family_history'] = df_cleaned['family_history'].apply(
            lambda x: 'Yes' if str(x).lower() == 'yes' else 'No' if str(x).lower() == 'no' else x
        )

    # Fix potential issues in treatment (if present)
    if 'treatment' in df_cleaned.columns:
        # Standardize Yes/No values
        df_cleaned['treatment'] = df_cleaned['treatment'].apply(
            lambda x: 'Yes' if str(x).lower() == 'yes' else 'No' if str(x).lower() == 'no' else x
        )

    # Standardize work_interfere values (if present)
    if 'work_interfere' in df_cleaned.columns:
        # Fill missing values with "Don't know"
        df_cleaned['work_interfere'] = df_cleaned['work_interfere'].fillna("Don't know")

    # Check for duplicate rows
    duplicates = df_cleaned.duplicated().sum()
    if duplicates > 0:
        print(f"\nFound {duplicates} duplicate rows, removing them...")
        df_cleaned = df_cleaned.drop_duplicates()

    return df_cleaned


# Model Training and Evaluation Function
def train_and_evaluate_models(X, y):
    """
    Train and evaluate multiple ML models.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        Dictionary of trained models and their performance metrics
    """
    print("\n=== Model Training and Evaluation ===")

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Check for class imbalance
    class_counts = np.bincount(y)
    if min(class_counts) / max(class_counts) < 0.5:
        print("Detected class imbalance, applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"Class distribution after SMOTE: {np.bincount(y_train)}")

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }

    # Store results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # For ROC AUC, we need probability predictions
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None

        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred
        }

        # Display metrics
        print(f"{name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        if auc:
            print(f"  AUC:       {auc:.4f}")

    return results, X_train, X_test, y_train, y_test


# Hyperparameter Tuning Function
def tune_best_model(best_model_name, model, X_train, y_train):
    """
    Perform hyperparameter tuning on the best performing model.

    Args:
        best_model_name: Name of the best model
        model: The model object
        X_train: Training features
        y_train: Training target

    Returns:
        The best model with tuned hyperparameters
    """
    print(f"\n=== Hyperparameter Tuning for {best_model_name} ===")

    # Define parameter grids for different models
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'linear']
        },
        'Decision Tree': {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    }

    # Get the right parameter grid
    if best_model_name in param_grids:
        param_grid = param_grids[best_model_name]

        # Use RandomizedSearchCV for complex models
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Neural Network']:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=10,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                random_state=42
            )
        else:  # Use GridSearchCV for simpler models
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )

        # Fit the search
        search.fit(X_train, y_train)

        # Display best parameters and score
        print(f"Best parameters: {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")

        return search.best_estimator_
    else:
        print(f"No parameter grid defined for {best_model_name}")
        return model


# Visualization Functions
def visualize_results(results, X_test, y_test, selected_features=None):
    """
    Create visualizations for model evaluation.

    Args:
        results: Dictionary with model results
        X_test: Test features
        y_test: Test target
        selected_features: List of feature names
    """
    print("\n=== Visualizing Results ===")

    # Model performance comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    precisions = [results[name]['precision'] for name in model_names]
    recalls = [results[name]['recall'] for name in model_names]
    f1_scores = [results[name]['f1'] for name in model_names]

    # Set up the figure
    plt.figure(figsize=(12, 6))

    # Create a grouped bar chart
    x = np.arange(len(model_names))
    width = 0.2

    plt.bar(x - 1.5 * width, accuracies, width, label='Accuracy')
    plt.bar(x - 0.5 * width, precisions, width, label='Precision')
    plt.bar(x + 0.5 * width, recalls, width, label='Recall')
    plt.bar(x + 1.5 * width, f1_scores, width, label='F1 Score')

    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

    # Find the best model based on F1 score
    best_model_name = model_names[np.argmax(f1_scores)]
    best_model = results[best_model_name]['model']
    best_preds = results[best_model_name]['y_pred']

    # Confusion Matrix for the best model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    # ROC Curve for models that support probability predictions
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        if result['auc'] is not None:
            model = result['model']
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig('roc_curves.png')
    plt.close()

    # Feature importance for the best model (if supported)
    if hasattr(best_model, 'feature_importances_') and selected_features is not None:
        plt.figure(figsize=(12, 8))

        # Get feature importances
        importances = best_model.feature_importances_

        # Only display top 15 features if there are many
        if len(selected_features) > 15:
            # Get indices of top 15 features
            indices = np.argsort(importances)[-15:]
            top_features = [selected_features[i] for i in indices]
            top_importances = importances[indices]

            # Create horizontal bar chart
            plt.barh(range(len(top_importances)), top_importances)
            plt.yticks(range(len(top_importances)), top_features)
        else:
            # Create horizontal bar chart for all features
            indices = np.argsort(importances)
            plt.barh(range(len(importances)), importances[indices])
            plt.yticks(range(len(importances)), [selected_features[i] for i in indices])

        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    # PCA or t-SNE for visualization
    if X_test.shape[1] > 2:  # Only if we have more than 2 dimensions
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)

        plt.figure(figsize=(10, 8))
        for i, label in enumerate(np.unique(y_test)):
            plt.scatter(
                X_pca[y_test == label, 0],
                X_pca[y_test == label, 1],
                label=f'Class {label}'
            )
        plt.title('PCA of Test Data')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.savefig('pca_visualization.png')
        plt.close()

        # t-SNE (can be slow for large datasets)
        if X_test.shape[0] < 2000:  # Only do t-SNE for smaller datasets
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X_test)

            plt.figure(figsize=(10, 8))
            for i, label in enumerate(np.unique(y_test)):
                plt.scatter(
                    X_tsne[y_test == label, 0],
                    X_tsne[y_test == label, 1],
                    label=f'Class {label}'
                )
            plt.title('t-SNE of Test Data')
            plt.xlabel('t-SNE Feature 1')
            plt.ylabel('t-SNE Feature 2')
            plt.legend()
            plt.savefig('tsne_visualization.png')
            plt.close()


def main():
    # Load the dataset
    print("Loading the dataset...")
    data = pd.read_csv('survey.csv')
    # Clean the data to fix outliers and standardize values
    data = clean_data(data)

    # Display dataset information
    print("\nDataset Information:")
    print(f"Shape: {data.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(data.head())

    # Display column information
    print("\nColumn information:")
    print(data.info())

    # Check for missing values
    print("\nMissing values per column:")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])

    # Display basic statistics
    print("\nBasic statistics for numerical columns:")
    print(data.describe())

    # Identify the target variable (treatment)
    # Assuming the target column is named 'treatment' - we'll verify this and adjust
    target_candidates = [col for col in data.columns if 'treat' in col.lower()]
    print(f"\nPossible target columns: {target_candidates}")

    # Data Exploration
    print("\n=== Exploratory Data Analysis ===")

    # Find categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\nCategorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")

    # Based on dataset investigation, let's identify the target column
    # For now we'll assume it's 'treatment' or similar, but we'll verify this is correct
    target_col = 'treatment' if 'treatment' in data.columns else None

    if target_col:
        print(f"\nTarget variable distribution:")
        print(data[target_col].value_counts())
        data[target_col] = data[target_col].map({"Yes": 1, "No": 0})
        print(f"Percentage seeking treatment: {data[target_col].mean() * 100:.2f}%")
    else:
        print("\nTarget column not found. Manual inspection required.")
        # Display unique values for columns that might be the target
        for col in target_candidates:
            print(f"\nUnique values for {col}:")
            print(data[col].value_counts())

    # Let's examine example features to better understand the data
    print("\nExploring sample categorical features:")
    for col in categorical_cols[:5]:  # Show first 5 categorical columns
        print(f"\nUnique values for {col}:")
        print(data[col].value_counts())
    """Main execution function for the mental health treatment prediction project."""
    print("=== Mental Health Treatment Prediction Project ===")


    # Load and explore the dataset
    #data = pd.read_csv('survey.csv')

    # Clean the data to fix outliers and standardize values
    #data = clean_data(data)



    # Identify the target column - adjust this based on your specific dataset
    # For the OSMI Mental Health in Tech Survey, the target might be named differently
    target_candidates = [col for col in data.columns if 'treatment' in col.lower()]
    print(f"Potential target columns: {target_candidates}")

    # For this template, we'll assume the first candidate is our target
    if target_candidates:
        target_column = target_candidates[0]
    else:
        # Fallback to a common name if none found
        target_column = 'treatment'

    print(f"Using '{target_column}' as the target column")

    # Preprocess the data
    X, y = preprocess_data(data, target_column)
    y = y.map({"Yes": 1, "No": 0})

    # Engineer features
    print(y)
    print(data.head())
    X_selected, selected_features = engineer_features(X, y)

    # Train and evaluate models
    results, X_train, X_test, y_train, y_test = train_and_evaluate_models(X_selected, y)

    # Find the best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    print(f"\nBest model: {best_model_name} with F1 score: {results[best_model_name]['f1']:.4f}")

    # Tune the best model
    best_tuned_model = tune_best_model(
        best_model_name,
        results[best_model_name]['model'],
        X_train,
        y_train
    )

    # Evaluate the tuned model
    y_pred_tuned = best_tuned_model.predict(X_test)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    precision_tuned = precision_score(y_test, y_pred_tuned)
    recall_tuned = recall_score(y_test, y_pred_tuned)
    f1_tuned = f1_score(y_test, y_pred_tuned)

    print("\nTuned Model Results:")
    print(f"  Accuracy:  {accuracy_tuned:.4f}")
    print(f"  Precision: {precision_tuned:.4f}")
    print(f"  Recall:    {recall_tuned:.4f}")
    print(f"  F1 Score:  {f1_tuned:.4f}")

    # Update the best model in results
    results[best_model_name]['model'] = best_tuned_model
    results[best_model_name]['y_pred'] = y_pred_tuned
    results[best_model_name]['accuracy'] = accuracy_tuned
    results[best_model_name]['precision'] = precision_tuned
    results[best_model_name]['recall'] = recall_tuned
    results[best_model_name]['f1'] = f1_tuned

    # Visualize results
    visualize_results(results, X_test, y_test, selected_features)

    print("\n=== Project Complete ===")


if __name__ == "__main__":
    main()