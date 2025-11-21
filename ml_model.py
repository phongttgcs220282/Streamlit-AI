import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

try:
    # Load dataset
    df = pd.read_csv("dataset.csv")

    # Features and target
    X = df[['tenure', 'Contract', 'InternetService', 'MonthlyCharges']]
    y = df['Churn'].map({'Yes': 1, 'No': 0})  # Convert to 0/1

    # Column types
    numeric_features = ['tenure', 'MonthlyCharges']
    categorical_features = ['Contract', 'InternetService']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    # Build model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    model.fit(X, y)
except FileNotFoundError:
    raise FileNotFoundError("The dataset.csv file was not found.")
except pd.errors.EmptyDataError:
    raise ValueError("The dataset.csv file is empty.")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the dataset: {e}")

# Ensure two blank lines before the function definition


def predict_churn(tenure, contract, internet, monthly):
    """Predict churn probability"""
    try:
        # Validate inputs
        if not isinstance(tenure, (int, float)) or tenure < 0:
            raise ValueError("Tenure must be a non-negative number.")
        if not isinstance(monthly, (int, float)) or monthly < 0:
            raise ValueError("MonthlyCharges must be a non-negative number.")
        if not isinstance(contract, str) or not isinstance(internet, str):
            raise ValueError("Contract and InternetService must be strings.")

        pred = model.predict_proba([
            [tenure, contract, internet, monthly]
        ])[0][1]  # Break long line for readability
        return float(pred)
    except Exception as e:
        return {"error": str(e)}
