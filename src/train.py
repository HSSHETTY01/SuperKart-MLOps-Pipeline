import os
import numpy as np
import pandas as pd
import skops.io as sio
from huggingface_hub import HfApi, login
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    print("--- Executing Model Training Step ---")
    train_df = pd.read_csv("data/superkart_train.csv")
    test_df = pd.read_csv("data/superkart_test.csv")

    target_col = "Product_Store_Sales_Total"
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    num_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "regressor__n_estimators": [50, 100],
        "regressor__max_depth": [5, 10],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"R²: {r2_score(y_test, y_pred):.4f} | RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/model.skops"
    sio.dump(best_model, model_path)
    print(f"✅ Model serialized to {model_path}")

    hf_token = os.getenv("HF_TOKEN")
    hf_user = os.getenv("HF_USERNAME", "HSSHETTY01")
    if hf_token:
        login(token=hf_token)
        api = HfApi(token=hf_token)
        repo_id = f"{hf_user}/SuperKart-RandomForest-Model"
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=model_path, path_in_repo="model.skops", repo_id=repo_id, repo_type="model")
        print(f"✅ Model registered to Hugging Face Model Hub: {repo_id}")

if __name__ == "__main__":
    main()
