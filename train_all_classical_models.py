import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore")

base_dir = Path(__file__).parent
DATA_DIR = base_dir / "data"
MODEL_DIR = base_dir / "models"
MODEL_DIR.mkdir(exist_ok=True)

PATIENTS_PATH = DATA_DIR / "patients_dataton_actualizado.csv"
VISITS_PATH = DATA_DIR / "visits_dataton_actualizado.csv"

EVENT_TARGET = "cardiotoxicity_event_next_90d"
RISK_TARGET = "baseline_risk_group"

RANDOM_STATE = 42
TEST_SIZE = 0.25
EVENT_THRESHOLD = 0.25


NUMERIC_FEATURES = [
    "age",
    "bmi",
    "baseline_lvef",
    "baseline_gls",
    "baseline_troponin",
    "baseline_ntprobnp",
    "cumulative_doxorubicin_mg_m2",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cancer_type",
    "stage",
    "treatment_family",
]

BOOLEAN_FEATURES = [
    "hypertension",
    "diabetes",
    "dyslipidemia",
    "ckd",
    "prior_cvd",
    "anthracycline_exposure",
    "antiher2_exposure",
    "immunotherapy_exposure",
    "radiotherapy_left_chest",
    "acei_arb",
    "beta_blocker",
    "statin",
    "sglt2",
    "anticoagulation",
]


RISK_ORDER = {
    "low": 0,
    "moderate": 1,
    "high": 2,
    "very_high": 3,
    "Bajo": 0,
    "Moderado": 1,
    "Alto": 2,
    "Muy alto": 3,
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    patients_df = pd.read_csv(PATIENTS_PATH)
    visits_df = pd.read_csv(VISITS_PATH)
    return patients_df, visits_df


def calculate_guideline_prediction(row: pd.Series) -> tuple[str, float]:
    """Misma lógica base que el GuidelineRiskEngine del dashboard.

    Devuelve:
        guideline_group: low/moderate/high/very_high
        guideline_prob: probabilidad aproximada asociada al grupo
    """

    points = 0

    age = row.get("age", 0)
    prior_cvd = bool(row.get("prior_cvd", False))
    hypertension = bool(row.get("hypertension", False))
    diabetes = bool(row.get("diabetes", False))
    ckd = bool(row.get("ckd", False))

    baseline_lvef = float(row.get("baseline_lvef", 60))
    baseline_gls = float(row.get("baseline_gls", -20))
    baseline_troponin = float(row.get("baseline_troponin", 0))
    baseline_ntprobnp = float(row.get("baseline_ntprobnp", 0))

    anthracycline_exposure = bool(row.get("anthracycline_exposure", False))
    antiher2_exposure = bool(row.get("antiher2_exposure", False))
    immunotherapy_exposure = bool(row.get("immunotherapy_exposure", False))
    radiotherapy_left_chest = bool(row.get("radiotherapy_left_chest", False))

    if age >= 75:
        points += 2
    elif age >= 65:
        points += 1

    if prior_cvd:
        points += 3
    if hypertension:
        points += 1
    if diabetes:
        points += 1
    if ckd:
        points += 1

    if baseline_lvef < 50:
        points += 3
    elif baseline_lvef < 55:
        points += 2

    if baseline_gls > -18:
        points += 1

    if baseline_troponin > 20:
        points += 1

    if baseline_ntprobnp > 400:
        points += 1

    if anthracycline_exposure:
        points += 2

    if antiher2_exposure:
        points += 2

    if immunotherapy_exposure:
        points += 1

    if radiotherapy_left_chest:
        points += 1

    if points <= 2:
        return "low", 0.08

    if points <= 5:
        return "moderate", 0.19

    if points <= 8:
        return "high", 0.36

    return "very_high", 0.58


def add_event_target_from_visits(
    patients_df: pd.DataFrame,
    visits_df: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega el target longitudinal de visits a nivel paciente.

    Se usa max() porque si un paciente tiene al menos una visita con evento futuro,
    clínicamente se considera que el paciente presentó evento durante el seguimiento.
    """

    patients_df = patients_df.copy()
    visits_df = visits_df.copy()

    if EVENT_TARGET not in visits_df.columns:
        raise ValueError(f"No existe {EVENT_TARGET} en visits.")

    event_by_patient = (
        visits_df.groupby("patient_id")[EVENT_TARGET]
        .max()
        .reset_index()
    )

    patients_df = patients_df.merge(
        event_by_patient,
        on="patient_id",
        how="left",
    )

    patients_df[EVENT_TARGET] = (
        patients_df[EVENT_TARGET]
        .fillna(0)
        .astype(int)
    )

    return patients_df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df[RISK_TARGET] = (
        df[RISK_TARGET]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    risk_mapping = {
        "bajo": "low",
        "low": "low",
        "moderado": "moderate",
        "medio": "moderate",
        "moderate": "moderate",
        "alto": "high",
        "high": "high",
        "muy alto": "very_high",
        "muy_alto": "very_high",
        "very high": "very_high",
        "very_high": "very_high",
    }

    df[RISK_TARGET] = df[RISK_TARGET].map(risk_mapping)

    if df[RISK_TARGET].isna().any():
        bad_values = df.loc[df[RISK_TARGET].isna(), RISK_TARGET].unique()
        raise ValueError(f"Valores de riesgo no reconocidos: {bad_values}")

    return df


def validate_columns(df: pd.DataFrame, target: str) -> list[str]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
    required = features + [target]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    return features


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    boolean_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bool", boolean_transformer, BOOLEAN_FEATURES),
        ]
    )


def build_candidate_models(task_type: str) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()

    if task_type == "binary":
        logistic = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    else:
        logistic = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            multi_class="auto",
            random_state=RANDOM_STATE,
        )

    models = {
        "logistic_regression": logistic,
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    return {
        name: Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }


def build_quantum_inspired_models(task_type: str) -> dict[str, Pipeline]:
    """Modelos quantum-inspired basados en mapas de características no lineales.

    Nystroem aproxima kernels no lineales y actúa como un "feature map" inspirado
    en enfoques kernel/quantum-kernel, manteniendo ejecución viable para el Datatón.
    """

    preprocessor = build_preprocessor()

    models = {
        "quantum_inspired_rbf_kernel": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("quantum_feature_map", Nystroem(
                    kernel="rbf",
                    gamma=0.08,
                    n_components=250,
                    random_state=RANDOM_STATE,
                )),
                ("model", SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
        "quantum_inspired_poly_kernel": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("quantum_feature_map", Nystroem(
                    kernel="poly",
                    degree=3,
                    gamma=0.04,
                    n_components=250,
                    random_state=RANDOM_STATE,
                )),
                ("model", SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
    }

    return models


def evaluate_binary_model(
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= EVENT_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model_name": model_name,
        "auc_roc": roc_auc_score(y_test, y_prob),
        "auc_pr": average_precision_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "recall_sensitivity": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "threshold": EVENT_THRESHOLD,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_multiclass_model(
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    try:
        auc_ovr = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=pipeline.classes_,
        )
    except Exception:
        auc_ovr = np.nan

    return {
        "model_name": model_name,
        "auc_ovr_macro": auc_ovr,
        "accuracy": accuracy_score(y_test, y_pred),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def train_event_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("ENTRENANDO MODELOS CLASICOS BINARIOS DE EVENTO 90D")
    print("=" * 80)

    features = validate_columns(df, EVENT_TARGET)
    X = df[features]
    y = df[EVENT_TARGET].astype(int)

    print(f"Pacientes: {len(X)}")
    print(f"Eventos: {y.sum()}")
    print(f"Tasa evento: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipelines = build_candidate_models(task_type="binary")

    results = []
    trained = {}

    for name, pipeline in pipelines.items():
        print(f"Entrenando {name}...")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_binary_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        results.append(metrics)
        trained[name] = pipeline

        print(
            f"AUC={metrics['auc_roc']:.3f} | "
            f"Recall={metrics['recall_sensitivity']:.3f} | "
            f"F1={metrics['f1']:.3f}"
        )

    results_df = pd.DataFrame(results)

    results_df["selection_score"] = (
        0.45 * results_df["auc_roc"]
        + 0.35 * results_df["recall_sensitivity"]
        + 0.20 * results_df["f1"]
    )

    results_df = results_df.sort_values("selection_score", ascending=False)
    best_name = results_df.iloc[0]["model_name"]
    best_model = trained[best_name]

    joblib.dump(best_model, MODEL_DIR / "best_event_model.joblib")

    results_df.to_csv(
        MODEL_DIR / "event_models_comparison.csv",
        index=False,
    )

    metadata = {
        "task": "binary_event_prediction",
        "target": EVENT_TARGET,
        "best_model_name": best_name,
        "threshold": EVENT_THRESHOLD,
        "features": features,
        "selection_criteria": "0.45*AUC + 0.35*Recall + 0.20*F1",
    }

    with open(MODEL_DIR / "best_event_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\nMejor modelo clasico evento:", best_name)


def train_risk_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("ENTRENANDO MODELOS CLASICOS MULTICLASE DE RIESGO")
    print("=" * 80)

    features = validate_columns(df, RISK_TARGET)
    X = df[features]
    y = df[RISK_TARGET].astype(str)

    print(f"Pacientes: {len(X)}")
    print("Distribución target:")
    print(y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipelines = build_candidate_models(task_type="multiclass")

    results = []
    trained = {}

    for name, pipeline in pipelines.items():
        print(f"Entrenando {name}...")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_multiclass_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        results.append(metrics)
        trained[name] = pipeline

        print(
            f"F1_macro={metrics['f1_macro']:.3f} | "
            f"Recall_macro={metrics['recall_macro']:.3f} | "
            f"Accuracy={metrics['accuracy']:.3f}"
        )

    results_df = pd.DataFrame(results)

    results_df["selection_score"] = (
        0.40 * results_df["f1_macro"]
        + 0.30 * results_df["recall_macro"]
        + 0.20 * results_df["accuracy"]
        + 0.10 * results_df["auc_ovr_macro"].fillna(0)
    )

    results_df = results_df.sort_values("selection_score", ascending=False)
    best_name = results_df.iloc[0]["model_name"]
    best_model = trained[best_name]

    joblib.dump(best_model, MODEL_DIR / "best_risk_model.joblib")

    results_df.to_csv(
        MODEL_DIR / "risk_models_comparison.csv",
        index=False,
    )

    metadata = {
        "task": "multiclass_risk_prediction",
        "target": RISK_TARGET,
        "best_model_name": best_name,
        "features": features,
        "classes": list(best_model.classes_),
        "selection_criteria": "0.40*F1_macro + 0.30*Recall_macro + 0.20*Accuracy + 0.10*AUC_OVR",
    }

    with open(MODEL_DIR / "best_risk_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\nMejor modelo clasico riesgo:", best_name)


def train_quantum_event_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("ENTRENANDO MODELOS QUANTUM-INSPIRED BINARIOS")
    print("=" * 80)

    features = validate_columns(df, EVENT_TARGET)
    X = df[features]
    y = df[EVENT_TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipelines = build_quantum_inspired_models(task_type="binary")

    results = []
    trained = {}

    for name, pipeline in pipelines.items():
        print(f"Entrenando {name}...")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_binary_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        results.append(metrics)
        trained[name] = pipeline

        print(
            f"AUC={metrics['auc_roc']:.3f} | "
            f"Recall={metrics['recall_sensitivity']:.3f} | "
            f"F1={metrics['f1']:.3f}"
        )

    results_df = pd.DataFrame(results)

    results_df["selection_score"] = (
        0.45 * results_df["auc_roc"]
        + 0.35 * results_df["recall_sensitivity"]
        + 0.20 * results_df["f1"]
    )

    results_df = results_df.sort_values("selection_score", ascending=False)

    best_name = results_df.iloc[0]["model_name"]
    best_model = trained[best_name]

    joblib.dump(best_model, MODEL_DIR / "best_quantum_event_model.joblib")

    results_df.to_csv(
        MODEL_DIR / "quantum_event_models_comparison.csv",
        index=False,
    )

    metadata = {
        "task": "quantum_inspired_binary_event_prediction",
        "target": EVENT_TARGET,
        "best_model_name": best_name,
        "threshold": EVENT_THRESHOLD,
        "features": features,
        "selection_criteria": "0.45*AUC + 0.35*Recall + 0.20*F1",
    }

    with open(MODEL_DIR / "best_quantum_event_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\nMejor modelo quantum-inspired evento:", best_name)


def train_quantum_risk_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("ENTRENANDO MODELOS QUANTUM-INSPIRED MULTICLASE")
    print("=" * 80)

    features = validate_columns(df, RISK_TARGET)
    X = df[features]
    y = df[RISK_TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipelines = build_quantum_inspired_models(task_type="multiclass")

    results = []
    trained = {}

    for name, pipeline in pipelines.items():
        print(f"Entrenando {name}...")
        pipeline.fit(X_train, y_train)

        metrics = evaluate_multiclass_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        results.append(metrics)
        trained[name] = pipeline

        print(
            f"F1_macro={metrics['f1_macro']:.3f} | "
            f"Recall_macro={metrics['recall_macro']:.3f} | "
            f"Accuracy={metrics['accuracy']:.3f}"
        )

    results_df = pd.DataFrame(results)

    results_df["selection_score"] = (
        0.40 * results_df["f1_macro"]
        + 0.30 * results_df["recall_macro"]
        + 0.20 * results_df["accuracy"]
        + 0.10 * results_df["auc_ovr_macro"].fillna(0)
    )

    results_df = results_df.sort_values("selection_score", ascending=False)

    best_name = results_df.iloc[0]["model_name"]
    best_model = trained[best_name]

    joblib.dump(best_model, MODEL_DIR / "best_quantum_risk_model.joblib")

    results_df.to_csv(
        MODEL_DIR / "quantum_risk_models_comparison.csv",
        index=False,
    )

    metadata = {
        "task": "quantum_inspired_multiclass_risk_prediction",
        "target": RISK_TARGET,
        "best_model_name": best_name,
        "features": features,
        "classes": list(best_model.classes_),
        "selection_criteria": "0.40*F1_macro + 0.30*Recall_macro + 0.20*Accuracy + 0.10*AUC_OVR",
    }

    with open(MODEL_DIR / "best_quantum_risk_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\nMejor modelo quantum-inspired riesgo:", best_name)


def generate_population_predictions(df: pd.DataFrame) -> None:
    classical_risk_model = joblib.load(MODEL_DIR / "best_risk_model.joblib")
    classical_event_model = joblib.load(MODEL_DIR / "best_event_model.joblib")

    quantum_risk_path = MODEL_DIR / "best_quantum_risk_model.joblib"
    quantum_event_path = MODEL_DIR / "best_quantum_event_model.joblib"

    quantum_risk_model = joblib.load(quantum_risk_path) if quantum_risk_path.exists() else classical_risk_model
    quantum_event_model = joblib.load(quantum_event_path) if quantum_event_path.exists() else classical_event_model

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
    X = df[features]

    classical_risk_pred = classical_risk_model.predict(X)
    classical_event_prob = classical_event_model.predict_proba(X)[:, 1]

    quantum_risk_pred = quantum_risk_model.predict(X)
    quantum_event_prob = quantum_event_model.predict_proba(X)[:, 1]

    guideline_predictions = df.apply(
        calculate_guideline_prediction,
        axis=1,
    )

    guideline_group = guideline_predictions.apply(lambda x: x[0])
    guideline_prob = guideline_predictions.apply(lambda x: x[1])

    predictions_df = pd.DataFrame(
        {
            "patient_id": df["patient_id"],
            "event_90d": df[EVENT_TARGET].astype(bool),
            "treatment_family": df["treatment_family"],

            "baseline_risk_group": df[RISK_TARGET],

            "guideline_group": guideline_group,
            "guideline_prob": guideline_prob,

            "classical_group": classical_risk_pred,
            "classical_prob": classical_event_prob,

            "quantum_group": quantum_risk_pred,
            "quantum_prob": quantum_event_prob,
        }
    )

    predictions_df["guideline_rank"] = predictions_df["guideline_group"].map(RISK_ORDER)
    predictions_df["classical_rank"] = predictions_df["classical_group"].map(RISK_ORDER)
    predictions_df["quantum_rank"] = predictions_df["quantum_group"].map(RISK_ORDER)

    missing_rank_cols = ["guideline_rank", "classical_rank", "quantum_rank"]
    for col in missing_rank_cols:
        if predictions_df[col].isna().any():
            bad_rows = predictions_df[predictions_df[col].isna()].head(10)
            raise ValueError(
                f"Hay valores de riesgo no mapeados en {col}. "
                f"Ejemplos:\n{bad_rows[['patient_id', 'guideline_group', 'classical_group', 'quantum_group']]}"
            )

    predictions_df["classical_discrepancy"] = (
        predictions_df["classical_rank"] - predictions_df["guideline_rank"]
    )

    predictions_df["quantum_discrepancy"] = (
        predictions_df["quantum_rank"] - predictions_df["guideline_rank"]
    )

    predictions_df["missed_by_guidelines_classical"] = (
        (predictions_df["guideline_rank"] <= 1)
        & (predictions_df["classical_rank"] >= 2)
        & (predictions_df["event_90d"] == True)
    )

    predictions_df["missed_by_guidelines_quantum"] = (
        (predictions_df["guideline_rank"] <= 1)
        & (predictions_df["quantum_rank"] >= 2)
        & (predictions_df["event_90d"] == True)
    )

    predictions_df.to_csv(
        MODEL_DIR / "population_predictions.csv",
        index=False,
    )

    print("Predicciones poblacionales guardadas en models/population_predictions.csv")


def main() -> None:
    patients_df, visits_df = load_data()

    patients_df = add_event_target_from_visits(
        patients_df,
        visits_df,
    )

    patients_df = normalize_data(patients_df)

    train_event_model(patients_df)
    train_risk_model(patients_df)

    train_quantum_event_model(patients_df)
    train_quantum_risk_model(patients_df)

    generate_population_predictions(patients_df)

    print("\n" + "=" * 80)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 80)
    print("Modelos generados:")
    print("- models/best_event_model.joblib")
    print("- models/best_event_model_metadata.json")
    print("- models/event_models_comparison.csv")
    print("- models/best_risk_model.joblib")
    print("- models/best_risk_model_metadata.json")
    print("- models/risk_models_comparison.csv")
    print("- models/best_quantum_event_model.joblib")
    print("- models/best_quantum_event_model_metadata.json")
    print("- models/quantum_event_models_comparison.csv")
    print("- models/best_quantum_risk_model.joblib")
    print("- models/best_quantum_risk_model_metadata.json")
    print("- models/quantum_risk_models_comparison.csv")
    print("- models/population_predictions.csv")


if __name__ == "__main__":
    main()
