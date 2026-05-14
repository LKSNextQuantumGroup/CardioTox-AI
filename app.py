
"""
CardioTox-AI Streamlit Dashboard v3
===================================

Versión mejorada:
- La explicabilidad queda integrada dentro de la vista individual del paciente.
- La primera pestaña es un panel clínico completo del paciente.
- Se mantienen:
  - Comparativa individual Guías vs IA clásica vs IA cuántica.
  - SHAP local.
  - Explicación generativa.
  - Evidencia científica.
  - Pacientes similares.
  - Seguimiento recomendado.
  - Evaluación global de modelos.
  - Discrepancias e infraclasificación.

Ejecución:
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List
import math
import random

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Domain layer
# ============================================================

class RiskGroup(str, Enum):
    LOW = "Bajo"
    MODERATE = "Moderado"
    HIGH = "Alto"
    VERY_HIGH = "Muy alto"


RISK_ORDER = {
    RiskGroup.LOW.value: 0,
    RiskGroup.MODERATE.value: 1,
    RiskGroup.HIGH.value: 2,
    RiskGroup.VERY_HIGH.value: 3,
}


@dataclass
class Patient:
    patient_id: str
    age: int
    sex: str
    cancer_type: str
    cancer_stage: str
    treatment_family: str
    baseline_lvef: float
    baseline_gls: float
    baseline_troponin: float
    baseline_ntprobnp: float
    hypertension: bool
    diabetes: bool
    dyslipidemia: bool
    chronic_kidney_disease: bool
    prior_cvd: bool
    anthracycline_exposure: bool
    antiher2_exposure: bool
    immunotherapy_exposure: bool
    radiotherapy_left: bool
    baseline_risk_prob: float
    baseline_risk_group: RiskGroup


@dataclass
class PredictionResult:
    model_name: str
    risk_probability: float
    risk_group: RiskGroup
    confidence: float
    explanation: str


@dataclass
class LiteratureReference:
    title: str
    source: str
    year: int
    relevance: str
    clinical_message: str


@dataclass
class SimilarPatient:
    patient_id: str
    similarity: float
    treatment_family: str
    baseline_risk_group: str
    observed_event_90d: bool
    explanation: str


class AppConfig:
    APP_TITLE = "CardioTox-AI"
    APP_SUBTITLE = "Predicción, explicación y seguimiento inteligente del riesgo de cardiotoxicidad"
    RANDOM_SEED = 42
    
    
# ==============================================================
# Guides upload and update
# ==============================================================

class GuidelineDocumentLoader:
    """Carga documentos de guías clínicas desde PDF o URL.

    En esta versión:
    - PDF: se deja preparado para integración real.
    - URL: mockeado para demo.
    """

    def load_from_pdf(self, uploaded_file) -> str:
        if uploaded_file is None:
            return ""

        # Versión demo: no extrae PDF real todavía.
        # Para versión real instalar: pip install pypdf
        # from pypdf import PdfReader
        # reader = PdfReader(uploaded_file)
        # text = "\n".join(page.extract_text() or "" for page in reader.pages)
        # return text

        return """
        Documento PDF cargado correctamente.
        Guía clínica de cardio-oncología.
        Se detectan secciones de estratificación de riesgo, FEVI, GLS, troponina,
        NT-proBNP, QTc, antraciclinas, anti-HER2, inmunoterapia y seguimiento.
        """

    def load_from_url(self, url: str) -> str:
        if not url:
            return ""

        # Versión demo.
        # Para versión real:
        # import requests
        # response = requests.get(url, timeout=20)
        # response.raise_for_status()
        # return response.text

        return f"""
        Documento recuperado desde URL:
        {url}

        Contenido simulado de guía clínica actualizada.
        Incluye criterios de riesgo basal, cardiotoxicidad sintomática y asintomática,
        seguimiento por tratamiento, biomarcadores, QTc y eventos cardiovasculares.
        """


class GuidelineAnalyzer:
    """Analiza una guía clínica y extrae cambios relevantes."""

    def analyze(self, document_text: str) -> dict:
        if not document_text.strip():
            return {}

        return {
            "version_detectada": "ESC Cardio-Oncology 2022 / versión demo",
            "resumen": (
                "La guía refuerza la estratificación basal del riesgo, el seguimiento "
                "según tratamiento oncológico y la monitorización de FEVI, GLS, troponina, "
                "NT-proBNP y QTc. También diferencia cardiotoxicidad sintomática y asintomática."
            ),
            "variables_detectadas": [
                "Edad",
                "Cardiopatía previa",
                "FEVI basal",
                "GLS basal",
                "Troponina",
                "NT-proBNP",
                "QTc",
                "Antraciclinas",
                "Anti-HER2",
                "Inmunoterapia",
                "Radioterapia torácica",
            ],
            "cambios_potenciales": [
                "Incorporar QTc como variable de toxicidad eléctrica.",
                "Diferenciar DC-RTC sintomática y asintomática.",
                "Reforzar seguimiento en pacientes con biomarcadores elevados.",
                "Ajustar frecuencia de visitas según tratamiento y riesgo.",
            ],
        }


class ClinicalRuleGenerator:
    """Genera reglas clínicas estructuradas a partir del análisis de la guía."""

    def generate_rules(self, guideline_analysis: dict) -> pd.DataFrame:
        if not guideline_analysis:
            return pd.DataFrame()

        rules = [
            {
                "rule_id": "R001",
                "nombre": "FEVI basal reducida",
                "condición": "baseline_lvef < 50",
                "riesgo_asignado": "Alto",
                "motivo": "FEVI basal reducida aumenta riesgo de cardiotoxicidad.",
                "estado": "Pendiente validación clínica",
            },
            {
                "rule_id": "R002",
                "nombre": "Antraciclinas + FEVI límite",
                "condición": "anthracycline_exposure == True and baseline_lvef < 55",
                "riesgo_asignado": "Alto",
                "motivo": "Exposición a antraciclinas con función ventricular límite.",
                "estado": "Pendiente validación clínica",
            },
            {
                "rule_id": "R003",
                "nombre": "Anti-HER2 + GLS alterado",
                "condición": "antiher2_exposure == True and baseline_gls > -18",
                "riesgo_asignado": "Alto",
                "motivo": "GLS alterado en paciente con tratamiento anti-HER2.",
                "estado": "Pendiente validación clínica",
            },
            {
                "rule_id": "R004",
                "nombre": "Biomarcadores elevados",
                "condición": "baseline_troponin > 20 or baseline_ntprobnp > 400",
                "riesgo_asignado": "Moderado/Alto",
                "motivo": "Elevación de biomarcadores cardíacos.",
                "estado": "Pendiente validación clínica",
            },
            {
                "rule_id": "R005",
                "nombre": "QTc prolongado",
                "condición": "qtc_prolonged == True",
                "riesgo_asignado": "Alto",
                "motivo": "Riesgo de toxicidad eléctrica y arritmias.",
                "estado": "Requiere variable no disponible",
            },
        ]

        return pd.DataFrame(rules)


class GuidelineImpactSimulator:
    """Simula impacto de nuevas reglas sobre el histórico."""

    def simulate(self, patients_df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
        if rules_df.empty:
            return pd.DataFrame()

        df = patients_df.copy()

        df["new_guideline_risk"] = df["baseline_risk_group"]

        if "baseline_lvef" in df.columns:
            df.loc[df["baseline_lvef"] < 50, "new_guideline_risk"] = "Alto"

        if {"anthracycline_exposure", "baseline_lvef"}.issubset(df.columns):
            df.loc[
                (df["anthracycline_exposure"] == True) & (df["baseline_lvef"] < 55),
                "new_guideline_risk",
            ] = "Alto"

        if {"antiher2_exposure", "baseline_gls"}.issubset(df.columns):
            df.loc[
                (df["antiher2_exposure"] == True) & (df["baseline_gls"] > -18),
                "new_guideline_risk",
            ] = "Alto"

        if {"baseline_troponin", "baseline_ntprobnp"}.issubset(df.columns):
            df.loc[
                (df["baseline_troponin"] > 20) | (df["baseline_ntprobnp"] > 400),
                "new_guideline_risk",
            ] = "Alto"

        changed = df[df["new_guideline_risk"] != df["baseline_risk_group"]]

        return changed[
            [
                "patient_id",
                "treatment_family",
                "baseline_risk_group",
                "new_guideline_risk",
                "baseline_lvef",
                "baseline_gls",
                "baseline_troponin",
                "baseline_ntprobnp",
                "cardiotoxicity_event_next_90d",
            ]
        ].copy()

class ScientificEvidenceScanner:
    """Busca publicaciones que podrían impactar futuras versiones de guías.

    Versión demo mockeada.
    En versión real podría conectarse a:
    - PubMed
    - Europe PMC
    - Semantic Scholar
    - CrossRef
    - ClinicalTrials.gov
    """

    def search_relevant_publications(self, topic: str = "cardio-oncology cardiotoxicity") -> pd.DataFrame:
        publications = [
            {
                "paper": "Anthracycline-induced cardiotoxicity: mechanisms, monitoring and prevention",
                "fuente": "European Heart Journal",
                "año": 2024,
                "área": "Antraciclinas",
                "impacto_potencial": "Puede reforzar el uso de biomarcadores y GLS en seguimiento precoz.",
                "nivel_evidencia": "Alto",
                "url": "https://pubmed.ncbi.nlm.nih.gov/",
            },
            {
                "paper": "Immune checkpoint inhibitor myocarditis: diagnosis and management",
                "fuente": "JACC CardioOncology",
                "año": 2024,
                "área": "Inmunoterapia",
                "impacto_potencial": "Puede modificar algoritmos de sospecha precoz de miocarditis.",
                "nivel_evidencia": "Alto",
                "url": "https://pubmed.ncbi.nlm.nih.gov/",
            },
            {
                "paper": "HER2-targeted therapy and cardiac surveillance strategies",
                "fuente": "Journal of Clinical Oncology",
                "año": 2023,
                "área": "Anti-HER2",
                "impacto_potencial": "Puede ajustar la frecuencia óptima de ecocardiogramas seriados.",
                "nivel_evidencia": "Moderado",
                "url": "https://pubmed.ncbi.nlm.nih.gov/",
            },
            {
                "paper": "Global longitudinal strain for early detection of cancer therapy-related cardiac dysfunction",
                "fuente": "Circulation Imaging",
                "año": 2023,
                "área": "GLS / FEVI",
                "impacto_potencial": "Refuerza el valor del GLS como señal precoz antes de caída de FEVI.",
                "nivel_evidencia": "Alto",
                "url": "https://pubmed.ncbi.nlm.nih.gov/",
            },
            {
                "paper": "QT prolongation and arrhythmia risk in targeted cancer therapies",
                "fuente": "Heart Rhythm",
                "año": 2024,
                "área": "QTc / arritmias",
                "impacto_potencial": "Refuerza la necesidad de incluir QTc en el dataset y en el motor de reglas.",
                "nivel_evidencia": "Moderado",
                "url": "https://pubmed.ncbi.nlm.nih.gov/",
            },
        ]

        return pd.DataFrame(publications)
# ============================================================
# Repository layer
# ============================================================

class DataRepository:
    def __init__(self, seed: int = AppConfig.RANDOM_SEED) -> None:
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def load_patients(self, n: int = 1000) -> pd.DataFrame:
        cancer_types = ["Mama", "Linfoma", "Pulmón", "Colon", "Mieloma", "Ovario"]
        stages = ["I", "II", "III", "IV"]
        treatments = ["Antraciclinas", "Anti-HER2", "Inmunoterapia", "VEGF", "TKI", "Mixto"]

        rows = []
        for i in range(1, n + 1):
            treatment = np.random.choice(treatments, p=[0.24, 0.22, 0.16, 0.14, 0.12, 0.12])
            age = int(np.clip(np.random.normal(63, 11), 25, 90))

            prior_cvd = np.random.rand() < 0.22
            htn = np.random.rand() < 0.38
            diabetes = np.random.rand() < 0.18
            ckd = np.random.rand() < 0.11
            dyslipidemia = np.random.rand() < 0.31

            baseline_lvef = round(float(np.clip(np.random.normal(61, 6) - (np.random.uniform(2, 8) if prior_cvd else 0), 38, 72)), 1)
            baseline_gls = round(float(np.random.normal(-20, 2.8) + (np.random.uniform(1, 3) if prior_cvd else 0)), 1)
            baseline_troponin = round(abs(np.random.normal(12, 8)), 1)
            baseline_ntprobnp = round(abs(np.random.normal(230, 180)), 1)

            true_prob = self._true_event_probability(
                age, prior_cvd, htn, diabetes, ckd,
                baseline_lvef, baseline_gls, baseline_troponin, baseline_ntprobnp, treatment
            )
            baseline_risk_prob = float(np.clip(true_prob + np.random.normal(0, 0.06), 0.02, 0.85))
            event = np.random.rand() < true_prob

            rows.append(
                {
                    "patient_id": f"P{i:05d}",
                    "age": age,
                    "sex": np.random.choice(["Mujer", "Hombre"], p=[0.57, 0.43]),
                    "cancer_type": np.random.choice(cancer_types),
                    "cancer_stage": np.random.choice(stages, p=[0.22, 0.31, 0.30, 0.17]),
                    "treatment_family": treatment,
                    "baseline_lvef": baseline_lvef,
                    "baseline_gls": baseline_gls,
                    "baseline_troponin": baseline_troponin,
                    "baseline_ntprobnp": baseline_ntprobnp,
                    "hypertension": htn,
                    "diabetes": diabetes,
                    "dyslipidemia": dyslipidemia,
                    "chronic_kidney_disease": ckd,
                    "prior_cvd": prior_cvd,
                    "anthracycline_exposure": treatment in ["Antraciclinas", "Mixto"],
                    "antiher2_exposure": treatment in ["Anti-HER2", "Mixto"],
                    "immunotherapy_exposure": treatment == "Inmunoterapia",
                    "radiotherapy_left": np.random.rand() < 0.18,
                    "baseline_risk_prob": round(baseline_risk_prob, 4),
                    "baseline_risk_group": self._prob_to_group(baseline_risk_prob).value,
                    "cardiotoxicity_event_next_90d": bool(event),
                    "true_event_probability": round(float(true_prob), 4),
                }
            )

        return pd.DataFrame(rows)

    def load_visits(self, patients: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, p in patients.iterrows():
            visits = np.random.randint(3, 9)
            for v in range(1, visits + 1):
                deterioration = 0.0
                if p["cardiotoxicity_event_next_90d"] and v > 2:
                    deterioration = np.random.uniform(1.5, 6.0)

                rows.append(
                    {
                        "patient_id": p["patient_id"],
                        "visit_id": f"{p['patient_id']}_V{v}",
                        "month": int(v * np.random.choice([1, 2, 3])),
                        "visit_lvef": round(p["baseline_lvef"] - deterioration + np.random.normal(0, 1.2), 1),
                        "visit_gls": round(p["baseline_gls"] + deterioration / 2 + np.random.normal(0, 0.8), 1),
                        "visit_troponin": round(max(0.1, p["baseline_troponin"] + deterioration * 3 + np.random.normal(0, 3)), 1),
                        "visit_ntprobnp": round(max(20, p["baseline_ntprobnp"] + deterioration * 80 + np.random.normal(0, 60)), 1),
                        "symptoms_hf": np.random.rand() < (0.05 + deterioration / 30),
                        "cardiotoxicity_event_now": bool(deterioration > 3.2 and np.random.rand() < 0.50),
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def _true_event_probability(
        age: int,
        prior_cvd: bool,
        htn: bool,
        diabetes: bool,
        ckd: bool,
        lvef: float,
        gls: float,
        troponin: float,
        ntprobnp: float,
        treatment: str,
    ) -> float:
        score = -2.4
        score += 0.027 * (age - 60)
        score += 0.78 if prior_cvd else 0
        score += 0.32 if htn else 0
        score += 0.30 if diabetes else 0
        score += 0.42 if ckd else 0
        score += 0.085 * max(0, 56 - lvef)
        score += 0.10 * max(0, gls + 18)
        score += 0.022 * max(0, troponin - 14)
        score += 0.0012 * max(0, ntprobnp - 300)

        treatment_weight = {
            "Antraciclinas": 0.65,
            "Anti-HER2": 0.56,
            "Inmunoterapia": 0.45,
            "VEGF": 0.40,
            "TKI": 0.35,
            "Mixto": 0.98,
        }
        score += treatment_weight.get(treatment, 0)

        if treatment == "Mixto" and lvef < 58:
            score += 0.28
        if treatment == "Anti-HER2" and gls > -19:
            score += 0.22
        if treatment == "Inmunoterapia" and troponin > 18:
            score += 0.20

        return float(np.clip(1 / (1 + math.exp(-score)), 0.02, 0.90))

    @staticmethod
    def _prob_to_group(prob: float) -> RiskGroup:
        if prob < 0.12:
            return RiskGroup.LOW
        if prob < 0.25:
            return RiskGroup.MODERATE
        if prob < 0.45:
            return RiskGroup.HIGH
        return RiskGroup.VERY_HIGH


# ============================================================
# Models
# ============================================================

class GuidelineRiskEngine:
    def predict(self, patient: Patient) -> PredictionResult:
        points = 0
        points += 2 if patient.age >= 75 else 1 if patient.age >= 65 else 0
        points += 3 if patient.prior_cvd else 0
        points += 1 if patient.hypertension else 0
        points += 1 if patient.diabetes else 0
        points += 1 if patient.chronic_kidney_disease else 0
        points += 3 if patient.baseline_lvef < 50 else 2 if patient.baseline_lvef < 55 else 0
        points += 1 if patient.baseline_gls > -18 else 0
        points += 1 if patient.baseline_troponin > 20 else 0
        points += 1 if patient.baseline_ntprobnp > 400 else 0
        points += 2 if patient.anthracycline_exposure else 0
        points += 2 if patient.antiher2_exposure else 0
        points += 1 if patient.immunotherapy_exposure else 0
        points += 1 if patient.radiotherapy_left else 0

        if points <= 2:
            group, prob = RiskGroup.LOW, 0.08
        elif points <= 5:
            group, prob = RiskGroup.MODERATE, 0.19
        elif points <= 8:
            group, prob = RiskGroup.HIGH, 0.36
        else:
            group, prob = RiskGroup.VERY_HIGH, 0.58

        return PredictionResult(
            "Guías ESC/HFA-ICOS",
            prob,
            group,
            0.82,
            f"Regla clínica mockeada con puntuación {points}. Considera edad, comorbilidades, FEVI, GLS, biomarcadores y tratamiento.",
        )


class ClassicalAIModel:
    def predict(self, patient: Patient) -> PredictionResult:
        prob = patient.baseline_risk_prob
        prob += 0.065 if patient.baseline_lvef < 55 else 0.0
        prob += 0.045 if patient.baseline_gls > -18 else 0.0
        prob += 0.055 if patient.baseline_troponin > 20 else 0.0
        prob += 0.060 if patient.treatment_family == "Mixto" else 0.0
        prob += 0.045 if patient.prior_cvd else 0.0
        prob = float(np.clip(prob + np.random.normal(0, 0.01), 0.02, 0.88))

        return PredictionResult(
            "IA clásica",
            round(prob, 4),
            self._prob_to_group(prob),
            0.88,
            "Modelo clásico mockeado: aprende patrones históricos combinando riesgo basal, biomarcadores, tratamiento y antecedentes cardiovasculares.",
        )

    @staticmethod
    def _prob_to_group(prob: float) -> RiskGroup:
        if prob < 0.12:
            return RiskGroup.LOW
        if prob < 0.25:
            return RiskGroup.MODERATE
        if prob < 0.45:
            return RiskGroup.HIGH
        return RiskGroup.VERY_HIGH


class QuantumInspiredModel:
    def predict(self, patient: Patient) -> PredictionResult:
        prob = patient.baseline_risk_prob
        prob += 0.055 if patient.anthracycline_exposure and patient.baseline_lvef < 58 else 0.0
        prob += 0.060 if patient.antiher2_exposure and patient.baseline_gls > -19 else 0.0
        prob += 0.045 if patient.prior_cvd and patient.baseline_ntprobnp > 350 else 0.0
        prob += 0.045 if patient.immunotherapy_exposure and patient.baseline_troponin > 16 else 0.0
        prob += 0.050 if patient.treatment_family == "Mixto" and patient.baseline_lvef < 60 and patient.baseline_gls > -20 else 0.0
        prob = float(np.clip(prob + np.random.normal(0, 0.008), 0.02, 0.92))

        return PredictionResult(
            "IA cuántica",
            round(prob, 4),
            self._prob_to_group(prob),
            0.84,
            "Modelo quantum-inspired mockeado: captura interacciones no lineales entre tratamiento, función ventricular y biomarcadores.",
        )

    @staticmethod
    def _prob_to_group(prob: float) -> RiskGroup:
        if prob < 0.12:
            return RiskGroup.LOW
        if prob < 0.25:
            return RiskGroup.MODERATE
        if prob < 0.45:
            return RiskGroup.HIGH
        return RiskGroup.VERY_HIGH


# ============================================================
# Services
# ============================================================

class PatientMapper:
    @staticmethod
    def row_to_patient(row: pd.Series) -> Patient:
        return Patient(
            patient_id=row["patient_id"],
            age=int(row["age"]),
            sex=row["sex"],
            cancer_type=row["cancer_type"],
            cancer_stage=row["cancer_stage"],
            treatment_family=row["treatment_family"],
            baseline_lvef=float(row["baseline_lvef"]),
            baseline_gls=float(row["baseline_gls"]),
            baseline_troponin=float(row["baseline_troponin"]),
            baseline_ntprobnp=float(row["baseline_ntprobnp"]),
            hypertension=bool(row["hypertension"]),
            diabetes=bool(row["diabetes"]),
            dyslipidemia=bool(row["dyslipidemia"]),
            chronic_kidney_disease=bool(row["chronic_kidney_disease"]),
            prior_cvd=bool(row["prior_cvd"]),
            anthracycline_exposure=bool(row["anthracycline_exposure"]),
            antiher2_exposure=bool(row["antiher2_exposure"]),
            immunotherapy_exposure=bool(row["immunotherapy_exposure"]),
            radiotherapy_left=bool(row["radiotherapy_left"]),
            baseline_risk_prob=float(row["baseline_risk_prob"]),
            baseline_risk_group=RiskGroup(row["baseline_risk_group"]),
        )


class PredictionService:
    def __init__(self) -> None:
        self.guideline_engine = GuidelineRiskEngine()
        self.classical_model = ClassicalAIModel()
        self.quantum_model = QuantumInspiredModel()

    def predict_patient(self, patient: Patient) -> List[PredictionResult]:
        return [
            self.guideline_engine.predict(patient),
            self.classical_model.predict(patient),
            self.quantum_model.predict(patient),
        ]

    def score_population(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in patients_df.iterrows():
            patient = PatientMapper.row_to_patient(row)
            predictions = self.predict_patient(patient)

            record = {
                "patient_id": patient.patient_id,
                "event_90d": bool(row["cardiotoxicity_event_next_90d"]),
                "treatment_family": patient.treatment_family,
                "baseline_risk_group": patient.baseline_risk_group.value,
            }

            for p in predictions:
                key = self._model_key(p.model_name)
                record[f"{key}_prob"] = p.risk_probability
                record[f"{key}_group"] = p.risk_group.value

            rows.append(record)

        scored = pd.DataFrame(rows)
        scored["guideline_rank"] = scored["guideline_group"].map(RISK_ORDER)
        scored["classical_rank"] = scored["classical_group"].map(RISK_ORDER)
        scored["quantum_rank"] = scored["quantum_group"].map(RISK_ORDER)

        scored["classical_discrepancy"] = scored["classical_rank"] - scored["guideline_rank"]
        scored["quantum_discrepancy"] = scored["quantum_rank"] - scored["guideline_rank"]

        scored["missed_by_guidelines_classical"] = (
            (scored["guideline_rank"] <= 1) & (scored["classical_rank"] >= 2) & scored["event_90d"]
        )
        scored["missed_by_guidelines_quantum"] = (
            (scored["guideline_rank"] <= 1) & (scored["quantum_rank"] >= 2) & scored["event_90d"]
        )
        return scored

    @staticmethod
    def _model_key(model_name: str) -> str:
        if "Guías" in model_name:
            return "guideline"
        if "clásica" in model_name:
            return "classical"
        return "quantum"


class ModelEvaluationService:
    def evaluate(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        y_true = scored_df["event_90d"].astype(int).values
        rows = []
        for name, col in [
            ("Guías ESC/HFA-ICOS", "guideline_prob"),
            ("IA clásica", "classical_prob"),
            ("IA cuántica", "quantum_prob"),
        ]:
            y_prob = scored_df[col].values
            y_pred = (y_prob >= 0.25).astype(int)
            rows.append(
                {
                    "Modelo": name,
                    "AUC estimado": self._pseudo_auc(y_true, y_prob),
                    "Sensibilidad": self._sensitivity(y_true, y_pred),
                    "Especificidad": self._specificity(y_true, y_pred),
                    "F1": self._f1(y_true, y_pred),
                    "Pacientes alto/muy alto": int(y_pred.sum()),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _pseudo_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        positives = y_score[y_true == 1]
        negatives = y_score[y_true == 0]
        if len(positives) == 0 or len(negatives) == 0:
            return 0.5
        sample_pos = np.random.choice(positives, size=min(1500, len(positives)), replace=True)
        sample_neg = np.random.choice(negatives, size=min(1500, len(negatives)), replace=True)
        return round(float(np.mean(sample_pos[:, None] > sample_neg[None, :])), 3)

    @staticmethod
    def _sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        return round(float(tp / max(1, tp + fn)), 3)

    @staticmethod
    def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return round(float(tn / max(1, tn + fp)), 3)

    @staticmethod
    def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return round(float(2 * precision * recall / max(1e-9, precision + recall)), 3)


class ExplainabilityService:
    def explain_patient(self, patient: Patient) -> pd.DataFrame:
        data = [
            ("Tratamiento oncológico", 0.23, "Aumenta riesgo"),
            ("FEVI basal", 0.18, "Aumenta riesgo" if patient.baseline_lvef < 55 else "Reduce riesgo"),
            ("GLS basal", 0.14, "Aumenta riesgo" if patient.baseline_gls > -18 else "Reduce riesgo"),
            ("Troponina basal", 0.13, "Aumenta riesgo" if patient.baseline_troponin > 20 else "Neutro"),
            ("Edad", 0.10, "Aumenta riesgo" if patient.age >= 65 else "Neutro"),
            ("Cardiopatía previa", 0.09, "Aumenta riesgo" if patient.prior_cvd else "Reduce riesgo"),
            ("NT-proBNP basal", 0.08, "Aumenta riesgo" if patient.baseline_ntprobnp > 400 else "Neutro"),
            ("HTA / diabetes / ERC", 0.05, "Aumenta riesgo"),
        ]
        return pd.DataFrame(data, columns=["variable", "impacto", "dirección"])

    def generate_patient_summary(self, patient: Patient, predictions: List[PredictionResult]) -> str:
        guideline, classical, quantum = predictions
        highest = max(predictions, key=lambda p: p.risk_probability)
        text = (
            f"El paciente {patient.patient_id} presenta una probabilidad máxima estimada de "
            f"{highest.risk_probability:.1%} según {highest.model_name}. "
            f"Las guías lo clasifican como {guideline.risk_group.value.lower()}, "
            f"la IA clásica como {classical.risk_group.value.lower()} y la IA cuántica como "
            f"{quantum.risk_group.value.lower()}."
        )
        if RISK_ORDER[quantum.risk_group.value] > RISK_ORDER[guideline.risk_group.value]:
            text += (
                " La discrepancia sugiere que el modelo cuántico detecta interacciones clínicas complejas "
                "entre tratamiento, biomarcadores y función ventricular que podrían no estar totalmente capturadas por reglas clínicas."
            )
        else:
            text += " Existe buena concordancia entre los enfoques, reforzando la recomendación clínica."
        return text


class LiteratureService:
    def get_references_for_patient(self, patient: Patient) -> List[LiteratureReference]:
        refs = [
            LiteratureReference("2022 ESC Guidelines on cardio-oncology", "European Society of Cardiology", 2022, "Guía clínica principal", "Estratificación basal y monitorización con FEVI, GLS y biomarcadores."),
            LiteratureReference("HFA-ICOS risk assessment tools", "Heart Failure Association / ICOS", 2020, "Riesgo basal", "Herramientas para clasificación de riesgo antes de terapias cardiotóxicas."),
        ]
        if patient.anthracycline_exposure:
            refs.append(LiteratureReference("Anthracycline cardiotoxicity and prevention strategies", "Cardio-Oncology Review", 2021, "Antraciclinas", "Dosis acumulada, FEVI, GLS y biomarcadores elevan el riesgo de disfunción ventricular."))
        if patient.antiher2_exposure:
            refs.append(LiteratureReference("HER2-targeted therapies and cardiac dysfunction", "Journal of Clinical Oncology", 2021, "Anti-HER2", "El seguimiento seriado de función ventricular es clave para detectar toxicidad precoz."))
        if patient.immunotherapy_exposure:
            refs.append(LiteratureReference("Immune checkpoint inhibitors and myocarditis", "European Heart Journal", 2022, "Inmunoterapia", "La miocarditis por inmunoterapia requiere vigilancia estrecha."))
        return refs


class SimilarPatientService:
    def find_similar_patients(self, patient: Patient, patients_df: pd.DataFrame, top_k: int = 5) -> List[SimilarPatient]:
        df = patients_df.copy()
        df["similarity_score"] = (
            1.0
            - abs(df["age"] - patient.age) / 80 * 0.22
            - abs(df["baseline_lvef"] - patient.baseline_lvef) / 40 * 0.23
            - abs(df["baseline_troponin"] - patient.baseline_troponin) / 100 * 0.12
            - (df["treatment_family"] != patient.treatment_family).astype(float) * 0.30
            - (df["prior_cvd"] != patient.prior_cvd).astype(float) * 0.13
        )
        df = df[df["patient_id"] != patient.patient_id].sort_values("similarity_score", ascending=False).head(top_k)
        return [
            SimilarPatient(
                row["patient_id"],
                round(float(row["similarity_score"]), 3),
                row["treatment_family"],
                row["baseline_risk_group"],
                bool(row["cardiotoxicity_event_next_90d"]),
                f"Similar por edad, FEVI, tratamiento {row['treatment_family']} y perfil cardiovascular basal.",
            )
            for _, row in df.iterrows()
        ]


class FollowUpPlanner:
    def build_follow_up_plan(self, patient: Patient, final_risk: RiskGroup) -> pd.DataFrame:
        actions = [
            ("Evaluación basal", "Semana 0", "ECG, FEVI, GLS, troponina, NT-proBNP", "Completado", "Alta"),
            ("Primer control", "Semana 2-4", "Troponina, NT-proBNP, síntomas", "Pendiente", "Alta" if final_risk in [RiskGroup.HIGH, RiskGroup.VERY_HIGH] else "Media"),
            ("Ecocardiograma", "Semana 8-12", "FEVI y GLS", "Pendiente", "Alta" if final_risk in [RiskGroup.HIGH, RiskGroup.VERY_HIGH] else "Media"),
            ("Revisión tratamiento", "Semana 12", "Reevaluar cardioprotección y tratamiento oncológico", "Pendiente", "Media"),
            ("Seguimiento longitudinal", "Cada 3 meses", "ECG, biomarcadores y eco según riesgo", "Planificado", "Media"),
        ]
        if patient.antiher2_exposure:
            actions.append(("Seguimiento anti-HER2", "Cada 3 meses", "Ecocardiograma seriado", "Planificado", "Alta"))
        if patient.immunotherapy_exposure:
            actions.append(("Vigilancia miocarditis", "Si síntomas", "Troponina, ECG, RM cardíaca si sospecha", "Planificado", "Alta"))
        return pd.DataFrame(actions, columns=["acción", "momento", "detalle", "estado", "prioridad"])


# ============================================================
# UI layer
# ============================================================

class DashboardApp:
    def __init__(self) -> None:
        self.repository = DataRepository()
        self.prediction_service = PredictionService()
        self.evaluation_service = ModelEvaluationService()
        self.xai_service = ExplainabilityService()
        self.literature_service = LiteratureService()
        self.similar_patient_service = SimilarPatientService()
        self.follow_up_planner = FollowUpPlanner()
        
        self.guideline_loader = GuidelineDocumentLoader()
        self.guideline_analyzer = GuidelineAnalyzer()
        self.rule_generator = ClinicalRuleGenerator()
        self.guideline_simulator = GuidelineImpactSimulator()
        
        self.evidence_scanner = ScientificEvidenceScanner()
        
        self.patients_df = self.repository.load_patients()
        self.visits_df = self.repository.load_visits(self.patients_df)
        self.scored_df = self.prediction_service.score_population(self.patients_df)
        self.metrics_df = self.evaluation_service.evaluate(self.scored_df)

    def run(self) -> None:
        st.set_page_config(
            page_title=AppConfig.APP_TITLE,
            page_icon="❤️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        self._render_header()
        row = self._render_sidebar()
        patient = PatientMapper.row_to_patient(row)
        predictions = self.prediction_service.predict_patient(patient)
        final_risk = max(predictions, key=lambda p: p.risk_probability).risk_group

        tabs = st.tabs(
            [
                "1. Paciente individual",
                "2. Evaluación global de modelos",
                "3. Discrepancias e infraclasificación",
                "4. Vista poblacional",
                "5. Gestor de Guías Clínicas",
            ]
        )

        with tabs[0]:
            self._render_patient_workspace(patient, predictions, final_risk)

        with tabs[1]:
            self._render_global_model_evaluation()

        with tabs[2]:
            self._render_discrepancy_analysis()

        with tabs[3]:
            self._render_population_view()
            
        with tabs[4]:
            self._render_guideline_manager()

    def _render_header(self) -> None:
        st.markdown(
            f"""
            <div style="padding: 1.2rem 1.5rem; border-radius: 1.2rem;
                        background: linear-gradient(90deg, #071c33, #193b68, #6d3fc7);
                        color: white; margin-bottom: 1rem;">
                <h1 style="margin: 0;">{AppConfig.APP_TITLE}</h1>
                <p style="font-size: 1.05rem; margin: 0.3rem 0 0 0;">
                    {AppConfig.APP_SUBTITLE}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_sidebar(self) -> pd.Series:
        st.sidebar.title("CardioTox-AI")
        patient_id = st.sidebar.selectbox("Paciente", self.patients_df["patient_id"].tolist())
        st.sidebar.markdown("---")
        st.sidebar.caption("Demo mockeada. Lista para conectar modelos reales y datasets del Datatón.")
        return self.patients_df[self.patients_df["patient_id"] == patient_id].iloc[0]

    def _render_patient_workspace(self, patient: Patient, predictions: List[PredictionResult], final_risk: RiskGroup) -> None:
        st.subheader("Vista individual del paciente")

        left, right = st.columns([1, 2.2])

        with left:
            with st.container(border=True):
                st.markdown("### Datos del paciente")
                st.write(f"**Paciente ID:** {patient.patient_id}")
                st.write(f"**Edad:** {patient.age}")
                st.write(f"**Sexo:** {patient.sex}")
                st.write(f"**Cáncer:** {patient.cancer_type} · Estadio {patient.cancer_stage}")
                st.write(f"**Tratamiento:** {patient.treatment_family}")
                st.write(f"**FEVI basal:** {patient.baseline_lvef:.1f}%")
                st.write(f"**GLS basal:** {patient.baseline_gls:.1f}%")
                st.write(f"**Riesgo registrado BD:** {patient.baseline_risk_prob:.1%}")

        with right:
            self._render_model_risk_cards(predictions)

        st.markdown("---")

        row1_col1, row1_col2, row1_col3 = st.columns([1.1, 1.1, 1])

        with row1_col1:
            self._render_local_shap(patient)

        with row1_col2:
            with st.container(border=True):
                st.markdown("### Explicación generativa")
                st.write(self.xai_service.generate_patient_summary(patient, predictions))
                st.caption("En la versión real, esta explicación se generará con IA generativa sobre reglas, SHAP, literatura y pacientes similares.")

        with row1_col3:
            self._render_compact_follow_up(patient, final_risk)

        row2_col1, row2_col2, row2_col3 = st.columns([1.1, 1.1, 1.3])

        with row2_col1:
            self._render_patient_literature(patient)

        with row2_col2:
            self._render_similar_patients(patient)

        with row2_col3:
            self._render_patient_follow_up_timeline(patient, final_risk)

        st.markdown("---")
        self._render_patient_evolution(patient)

    def _render_model_risk_cards(self, predictions: List[PredictionResult]) -> None:
        guideline, classical, quantum = predictions
        cols = st.columns([1, 1, 1, 1.2])

        card_data = [
            ("Guías ESC/HFA-ICOS", guideline),
            ("IA clásica", classical),
            ("IA cuántica", quantum),
        ]

        for col, (title, pred) in zip(cols[:3], card_data):
            with col:
                with st.container(border=True):
                    st.markdown(f"### {title}")
                    st.metric("Riesgo", pred.risk_group.value, f"{pred.risk_probability:.0%}")
                    st.caption(f"Confianza estimada: {pred.confidence:.0%}")

        with cols[3]:
            with st.container(border=True):
                st.markdown("### Discrepancia")
                diff = RISK_ORDER[quantum.risk_group.value] - RISK_ORDER[guideline.risk_group.value]
                if diff > 0:
                    st.error("Los modelos IA predicen mayor riesgo que las guías.")
                    st.write("El modelo cuántico podría estar recuperando pacientes de alto riesgo no capturados por reglas clínicas.")
                elif diff < 0:
                    st.warning("La IA estima menor riesgo que las guías.")
                    st.write("Revisar factores de protección y evolución longitudinal.")
                else:
                    st.success("Buena concordancia entre modelos.")
                    st.write("La recomendación se considera robusta.")

    def _render_local_shap(self, patient: Patient) -> None:
        with st.container(border=True):
            st.markdown("### Explicación SHAP")
            xai_df = self.xai_service.explain_patient(patient)
            fig = px.bar(
                xai_df.sort_values("impacto"),
                x="impacto",
                y="variable",
                orientation="h",
                color="dirección",
                title="Factores que más influyen en el riesgo",
            )
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)


    def _render_compact_follow_up(self, patient: Patient, final_risk: RiskGroup) -> None:
        with st.container(border=True):
            st.markdown("### Recomendación de seguimiento")
            st.metric("Riesgo final", final_risk.value)
            st.write("**Próxima visita:** Semana 2-4")

            checks = [
                "ECG",
                "Ecocardiograma FEVI/GLS",
                "Troponina ultrasensible",
                "NT-proBNP",
                "Revisión de tratamiento",
            ]

            for item in checks:
                st.write(f"✅ {item}")

            if st.button("Ver plan completo", use_container_width=True):
                self._show_follow_up_canvas(patient, final_risk)

    def _render_patient_literature(self, patient: Patient) -> None:
        with st.container(border=True):
            st.markdown("### Evidencia científica")
            for ref in self.literature_service.get_references_for_patient(patient):
                st.markdown(f"**{ref.title}**")
                st.caption(f"{ref.source} · {ref.year} · {ref.relevance}")
            st.link_button("Ver más literatura relacionada", "https://pubmed.ncbi.nlm.nih.gov/", use_container_width=True)

    def _render_similar_patients(self, patient: Patient) -> None:
        with st.container(border=True):
            st.markdown("### Pacientes similares")
            similar = self.similar_patient_service.find_similar_patients(patient, self.patients_df)
            df = pd.DataFrame(
                [
                    {
                        "ID": p.patient_id,
                        "Similitud": p.similarity,
                        "Riesgo": p.baseline_risk_group,
                        "Evento": "Sí" if p.observed_event_90d else "No",
                    }
                    for p in similar
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

    def _render_patient_follow_up_timeline(self, patient: Patient, final_risk: RiskGroup) -> None:
        with st.container(border=True):
            st.markdown("### Ruta de seguimiento")
            plan_df = self.follow_up_planner.build_follow_up_plan(patient, final_risk)
            timeline = plan_df.copy()
            timeline["start"] = range(len(timeline))
            timeline["end"] = range(1, len(timeline) + 1)
            fig = px.timeline(
                timeline,
                x_start="start",
                x_end="end",
                y="acción",
                color="prioridad",
                hover_data=["momento", "detalle", "estado"],
                title="Vista Gantt",
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=360, xaxis_title="Secuencia clínica", margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)

    def _render_patient_evolution(self, patient: Patient) -> None:
        with st.container(border=True):
            st.markdown("### Evolución longitudinal del paciente")
            patient_visits = self.visits_df[self.visits_df["patient_id"] == patient.patient_id]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=patient_visits["month"], y=patient_visits["visit_lvef"], mode="lines+markers", name="FEVI"))
            fig.add_trace(go.Scatter(x=patient_visits["month"], y=patient_visits["visit_gls"], mode="lines+markers", name="GLS"))
            fig.add_trace(go.Scatter(x=patient_visits["month"], y=patient_visits["visit_troponin"], mode="lines+markers", name="Troponina"))
            fig.update_layout(title="Evolución clínica", xaxis_title="Mes", yaxis_title="Valor")
            st.plotly_chart(fig, use_container_width=True)

    def _render_global_model_evaluation(self) -> None:
        st.subheader("Evaluación global de modelos en histórico")

        best = self.metrics_df.sort_values("AUC estimado", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mejor modelo", best["Modelo"])
        c2.metric("AUC estimado", best["AUC estimado"])
        c3.metric("Sensibilidad", best["Sensibilidad"])
        c4.metric("F1", best["F1"])

        st.dataframe(self.metrics_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                self.metrics_df,
                x="Modelo",
                y=["AUC estimado", "Sensibilidad", "Especificidad", "F1"],
                barmode="group",
                title="Comparativa de métricas",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            x = np.linspace(0, 1, 60)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=x ** 0.72, mode="lines", name="Guías ESC"))
            fig.add_trace(go.Scatter(x=x, y=x ** 0.46, mode="lines", name="IA clásica"))
            fig.add_trace(go.Scatter(x=x, y=x ** 0.39, mode="lines", name="IA cuántica"))
            fig.add_trace(go.Scatter(x=x, y=x, mode="lines", name="Línea base", line=dict(dash="dash")))
            fig.update_layout(title="Curvas ROC simuladas", xaxis_title="1 - Especificidad", yaxis_title="Sensibilidad")
            st.plotly_chart(fig, use_container_width=True)

    def _render_discrepancy_analysis(self) -> None:
        st.subheader("Discrepancias e infraclasificación")

        missed_classical = int(self.scored_df["missed_by_guidelines_classical"].sum())
        missed_quantum = int(self.scored_df["missed_by_guidelines_quantum"].sum())
        total_events = int(self.scored_df["event_90d"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Eventos reales 90d", total_events)
        c2.metric("Recuperados por IA clásica", missed_classical)
        c3.metric("Recuperados por IA cuántica", missed_quantum)

        recovered = self.scored_df[
            self.scored_df["missed_by_guidelines_quantum"] | self.scored_df["missed_by_guidelines_classical"]
        ]

        st.markdown("### Pacientes infraclasificados por guías")
        st.dataframe(
            recovered[
                [
                    "patient_id",
                    "treatment_family",
                    "guideline_group",
                    "classical_group",
                    "quantum_group",
                    "event_90d",
                    "guideline_prob",
                    "classical_prob",
                    "quantum_prob",
                ]
            ].head(40),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            heat = pd.crosstab(self.scored_df["guideline_group"], self.scored_df["quantum_group"])
            fig = px.imshow(
                heat,
                text_auto=True,
                title="Guías vs IA cuántica",
                labels=dict(x="IA cuántica", y="Guías", color="Pacientes"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            summary = (
                self.scored_df.groupby("quantum_discrepancy")["event_90d"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"quantum_discrepancy": "Diferencia IA cuántica - Guía", "count": "Pacientes", "mean": "Tasa evento"})
            )
            fig = px.bar(
                summary,
                x="Diferencia IA cuántica - Guía",
                y="Tasa evento",
                text_auto=".1%",
                title="Evento real según discrepancia",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Mensaje clave para el Datatón: los pacientes que la IA reclasifica por encima de las guías "
            "pueden representar perfiles clínicos infraclasificados, especialmente si presentan mayor tasa real de eventos."
        )

    def _render_population_view(self) -> None:
        st.subheader("Vista poblacional")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pacientes", f"{len(self.patients_df):,}")
        c2.metric("Visitas", f"{len(self.visits_df):,}")
        c3.metric("Evento 90 días", f"{self.patients_df['cardiotoxicity_event_next_90d'].mean():.1%}")
        c4.metric("Riesgo medio basal", f"{self.patients_df['baseline_risk_prob'].mean():.1%}")

        col1, col2 = st.columns(2)

        with col1:
            risk_counts = self.patients_df["baseline_risk_group"].value_counts().reset_index()
            risk_counts.columns = ["Riesgo", "Pacientes"]
            fig = px.pie(risk_counts, names="Riesgo", values="Pacientes", hole=0.45, title="Distribución de riesgo basal")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            event_by_treatment = (
                self.patients_df.groupby("treatment_family")["cardiotoxicity_event_next_90d"]
                .mean()
                .reset_index()
                .sort_values("cardiotoxicity_event_next_90d", ascending=False)
            )
            fig = px.bar(
                event_by_treatment,
                x="treatment_family",
                y="cardiotoxicity_event_next_90d",
                title="Eventos por tratamiento",
                text_auto=".1%",
            )
            st.plotly_chart(fig, use_container_width=True)
            
    @st.dialog("Plan completo de seguimiento cardio-oncológico", width="large")

    def _show_follow_up_canvas(self, patient: Patient, final_risk: RiskGroup) -> None:
        plan_df = self.follow_up_planner.build_follow_up_plan(patient, final_risk)

        st.markdown(
            f"""
            ### Paciente {patient.patient_id}

            **Riesgo final:** {final_risk.value}  
            **Tratamiento:** {patient.treatment_family}  
            **Objetivo:** anticipar cardiotoxicidad, priorizar visitas y reducir eventos cardiovasculares.
            """
        )

        c1, c2, c3 = st.columns(3)

        status_order = ["Completado", "Pendiente", "Planificado"]

        for col, status in zip([c1, c2, c3], status_order):
            with col:
                st.markdown(f"#### {status}")

                subset = plan_df[plan_df["estado"] == status]

                for _, row in subset.iterrows():
                    priority_icon = {
                        "Alta": "🔴",
                        "Media": "🟠",
                        "Baja": "🟢",
                    }.get(row["prioridad"], "🔵")

                    with st.container(border=True):
                        st.markdown(f"**{priority_icon} {row['acción']}**")
                        st.caption(row["momento"])
                        st.write(row["detalle"])
                        st.markdown(f"**Prioridad:** {row['prioridad']}")

        st.markdown("---")
        st.markdown("### Ruta temporal recomendada")

        timeline = plan_df.copy()
        timeline["start"] = range(len(timeline))
        timeline["end"] = range(1, len(timeline) + 1)

        fig = px.timeline(
            timeline,
            x_start="start",
            x_end="end",
            y="acción",
            color="prioridad",
            hover_data=["momento", "detalle", "estado"],
            title="Roadmap de seguimiento personalizado",
        )

        fig.update_yaxes(autorange="reversed")
        fig.update_layout(
            height=450,
            xaxis_title="Secuencia clínica",
            yaxis_title="Acción",
            margin=dict(l=10, r=10, t=50, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Recomendaciones clínicas automáticas")

        recommendations = [
            "Priorizar visita cardio-oncológica precoz.",
            "Monitorizar biomarcadores antes de los siguientes ciclos.",
            "Repetir ecocardiograma con FEVI y GLS según riesgo.",
            "Valorar cardioprotección si existe deterioro funcional o elevación de biomarcadores.",
            "Revisar tratamiento oncológico si aparece DC-RTC significativa.",
        ]

        for rec in recommendations:
            st.write(f"✅ {rec}")
            
            
    def _render_guideline_manager(self) -> None:
        st.subheader("Gestor de Guías Clínicas")

        st.info(
            "Este módulo permite cargar una nueva guía clínica, analizarla con IA, "
            "proponer reglas clínicas estructuradas y simular su impacto sobre el histórico. "
            "Las reglas propuestas requieren validación clínica antes de activarse."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 1. Cargar guía desde PDF")
            uploaded_pdf = st.file_uploader(
                "Subir guía clínica en PDF",
                type=["pdf"],
                help="Ejemplo: nueva guía ESC/HFA-ICOS en PDF.",
            )

        with col2:
            st.markdown("### 2. O recuperar guía desde URL")
            guideline_url = st.text_input(
                "URL de la guía clínica",
                placeholder="https://...",
            )

        document_text = ""

        if uploaded_pdf is not None:
            document_text = self.guideline_loader.load_from_pdf(uploaded_pdf)
            st.success("PDF cargado correctamente.")

        if guideline_url:
            document_text = self.guideline_loader.load_from_url(guideline_url)
            st.success("Documento recuperado desde URL.")

        st.markdown("---")

        if not document_text:
            st.warning("Sube un PDF o introduce una URL para iniciar el análisis.")
            return

        st.markdown("### 3. Análisis automático de la guía")

        analysis = self.guideline_analyzer.analyze(document_text)

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("#### Versión detectada")
            st.write(analysis["version_detectada"])

            st.markdown("#### Resumen IA")
            st.write(analysis["resumen"])

        with col_b:
            st.markdown("#### Variables clínicas detectadas")
            variables_df = pd.DataFrame(
                {"Variable": analysis["variables_detectadas"]}
            )
            st.dataframe(variables_df, use_container_width=True, hide_index=True)

        st.markdown("#### Cambios potenciales detectados")
        for change in analysis["cambios_potenciales"]:
            st.write(f"✅ {change}")
            
        st.markdown("---")
        
        st.markdown("### 4. Publicaciones recientes que podrían impactar futuras guías")

        st.info(
            "Este módulo identifica literatura científica relevante que podría influir en futuras "
            "actualizaciones de las guías clínicas. En la versión real se conectaría a PubMed, "
            "Europe PMC, Semantic Scholar o fuentes científicas validadas."
        )

        search_topic = st.text_input(
            "Tema de búsqueda científica",
            value="cardio-oncology cardiotoxicity anthracycline HER2 immunotherapy QTc GLS",
        )

        papers_df = self.evidence_scanner.search_relevant_publications(search_topic)

        st.dataframe(
            papers_df[
                [
                    "paper",
                    "fuente",
                    "año",
                    "área",
                    "impacto_potencial",
                    "nivel_evidencia",
                    "url",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            fig = px.histogram(
                papers_df,
                x="área",
                color="nivel_evidencia",
                title="Publicaciones por área clínica",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_p2:
            impact_summary = (
                papers_df.groupby("nivel_evidencia")
                .size()
                .reset_index(name="publicaciones")
            )

            fig = px.pie(
                impact_summary,
                names="nivel_evidencia",
                values="publicaciones",
                title="Nivel de evidencia detectado",
                hole=0.45,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Lectura generativa del impacto potencial")

        st.success(
            "La literatura reciente sugiere reforzar especialmente tres áreas del sistema: "
            "monitorización precoz con GLS y biomarcadores, detección de miocarditis asociada a inmunoterapia "
            "e incorporación de QTc para toxicidad eléctrica. Estas publicaciones podrían justificar futuras "
            "actualizaciones del motor de reglas clínicas."
        )

        st.markdown("---")

        st.markdown("### 5. Reglas clínicas propuestas")

        rules_df = self.rule_generator.generate_rules(analysis)

        st.dataframe(
            rules_df,
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "Estas reglas son una propuesta generada por IA. "
            "No deben aplicarse a pacientes sin revisión y validación por expertos clínicos."
        )

        st.markdown("---")

        st.markdown("### 6. Simulación de impacto sobre histórico")

        changed_patients_df = self.guideline_simulator.simulate(
            self.patients_df,
            rules_df,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Pacientes analizados", f"{len(self.patients_df):,}")
        c2.metric("Pacientes reclasificados", f"{len(changed_patients_df):,}")

        if len(self.patients_df) > 0:
            c3.metric(
                "% reclasificados",
                f"{len(changed_patients_df) / len(self.patients_df):.1%}",
            )

        if changed_patients_df.empty:
            st.success("La nueva propuesta de reglas no reclasifica pacientes en esta simulación.")
        else:
            st.dataframe(
                changed_patients_df.head(50),
                use_container_width=True,
                hide_index=True,
            )

            fig = px.histogram(
                changed_patients_df,
                x="baseline_risk_group",
                color="new_guideline_risk",
                barmode="group",
                title="Pacientes reclasificados por nuevas reglas",
                labels={
                    "baseline_risk_group": "Riesgo previo",
                    "new_guideline_risk": "Nuevo riesgo",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.markdown("### 7. Validación clínica")

        col_v1, col_v2, col_v3 = st.columns(3)

        with col_v1:
            st.button(
                "Guardar como borrador",
                use_container_width=True,
            )

        with col_v2:
            st.button(
                "Enviar a validación clínica",
                use_container_width=True,
            )

        with col_v3:
            st.button(
                "Activar versión validada",
                use_container_width=True,
                type="primary",
                disabled=True,
                help="Solo debería activarse tras validación clínica.",
            )

        st.caption(
            "Buenas prácticas: versionar cada conjunto de reglas, guardar fuente documental, "
            "fecha, usuario validador, cambios introducidos y métricas de impacto."
        )      


if __name__ == "__main__":
    DashboardApp().run()
