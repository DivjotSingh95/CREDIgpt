from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent
FEATURES_FILE = DATA_DIR / "selected_features.csv"
TARGET_FILE = DATA_DIR / "application_train.csv"
MODEL_FILE = DATA_DIR / "xgboost_default_model.json"
REPORT_FILE = DATA_DIR / "model_report.txt"

PRIMARY_KEY = "SK_ID_CURR"
TARGET_COL = "TARGET"

CATEGORICAL_COLUMNS = [
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.read_csv(FEATURES_FILE, encoding="latin-1")
    targets = pd.read_csv(TARGET_FILE, usecols=[PRIMARY_KEY, TARGET_COL], encoding="latin-1")

    data = features.merge(targets, on=PRIMARY_KEY, how="inner")
    y = data[TARGET_COL]
    X = data.drop(columns=[PRIMARY_KEY, TARGET_COL])
    return X, y


def encode_categoricals(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoded = X.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoded[col] = encoded[col].astype(str).fillna("missing")
        encoded[col] = encoder.fit_transform(encoded[col])
        encoders[col] = encoder

    return encoded, encoders


def build_model(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )


def format_report(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    feature_importance: pd.DataFrame,
) -> str:
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, target_names=["No Default (0)", "Default (1)"])

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]

    lines = [
        "=" * 60,
        "XGBoost Default Prediction Model Report",
        "=" * 60,
        "",
        "Dataset",
        "-" * 40,
        f"Features file: {FEATURES_FILE.name}",
        f"Target file:   {TARGET_FILE.name}",
        f"Feature count: {len(feature_importance)}",
        f"Test default rate: {y_test.mean():.2%}",
        "",
        "Model Performance (hold-out test set)",
        "-" * 40,
        f"ROC-AUC:   {auc:.4f}",
        f"Accuracy:  {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"F1 Score:  {f1:.4f}",
        f"Best threshold (Youden): {best_threshold:.4f}",
        "",
        "Confusion Matrix",
        "-" * 40,
        "                 Predicted 0   Predicted 1",
        f"Actual 0 (no default)   {cm[0, 0]:>8}   {cm[0, 1]:>8}",
        f"Actual 1 (default)      {cm[1, 0]:>8}   {cm[1, 1]:>8}",
        "",
        "Classification Report",
        "-" * 40,
        cls_report.rstrip(),
        "",
        "Top 15 Feature Importances (gain)",
        "-" * 40,
    ]

    for _, row in feature_importance.head(15).iterrows():
        lines.append(f"{row['feature']:<35} {row['importance']:.4f}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    X, y = load_data()
    X_encoded, _ = encode_categoricals(X)

    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model(scale_pos_weight)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    importance = pd.DataFrame(
        {
            "feature": X_encoded.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    report = format_report(y_test, y_pred, y_prob, importance)
    print(report)

    REPORT_FILE.write_text(report, encoding="utf-8")
    model.save_model(MODEL_FILE)

    print(f"\nReport saved to: {REPORT_FILE.name}")
    print(f"Model saved to:  {MODEL_FILE.name}")


if __name__ == "__main__":
    main()
