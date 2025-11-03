
# Proyecto Estadística I — MVP (local)

## Requisitos
- Python 3.10+

## Instalación rápida (Windows / Mac / Linux)
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

La app abrirá en el navegador (http://localhost:8501).

## Qué incluye este MVP
- Ingreso de datos manual o desde archivo CSV/TXT.
- Cálculo de: media, mediana, moda(s), varianza muestral, desviación estándar, cuartiles y deciles.
- Descripciones conceptuales breves (requisito del proyecto).

## Próximos pasos
- Módulo de probabilidad (Binomial, Poisson, Normal).
- Módulo de muestreo (IC y pruebas sencillas).
- Empaquetado para ejecución con doble clic (sin instalar nada).
