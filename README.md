# CardioTox-AI Dashboard v3

Versión con explicabilidad integrada dentro de la vista individual del paciente.

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pestañas

1. Paciente individual
2. Evaluación global de modelos
3. Discrepancias e infraclasificación
4. Vista poblacional

## Mejora principal

La vista de paciente integra:
- Datos clínicos.
- Comparación Guías vs IA clásica vs IA cuántica.
- Discrepancia.
- SHAP local.
- Explicación generativa.
- Evidencia científica.
- Pacientes similares.
- Ruta de seguimiento.
- Evolución longitudinal.