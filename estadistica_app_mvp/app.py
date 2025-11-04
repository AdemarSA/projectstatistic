import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy.stats import binom, poisson

import subprocess
import sys
import os

# --------------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------
st.set_page_config(page_title="Estadística I - Proyecto Final", page_icon="📊", layout="centered")

# --- CSS para mostrar todas las pestañas sin scroll horizontal ---
st.markdown("""
    <style>
        .stTabs [role="tablist"] {
            flex-wrap: wrap;
            justify-content: center;
        }
        .stTabs [role="tab"] {
            margin: 3px;
            padding: 6px 12px;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# BLOQUE DE BIENVENIDA GENERAL
# --------------------------------------------------------
st.title("📊 Proyecto Final — Estadística I. Ademar Sanayep Avalos 5990-15-2221")
st.write("""
Bienvenido/a a la aplicación del **Proyecto Final de Estadística I**.  
Esta herramienta fue creada para **realizar cálculos estadísticos y probabilísticos**
de manera interactiva y práctica.

Selecciona la sección que deseas utilizar:
- **📊 Sección 1: Estadística Descriptiva** → analiza y resume conjuntos de datos.  
- **📈 Sección 2: Probabilidad Binomial** → calcula probabilidades en experimentos discretos.  
- **📊 Sección 3: Distribución de Poisson** → calcula la probabilidad de eventos raros.  
- **📏 Sección 4: Intervalos de Confianza** → estima parámetros poblacionales a partir de muestras.
""")

# --------------------------------------------------------
# PESTAÑAS / SECCIONES
# --------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sección 1: Estadística Descriptiva",
    "📈 Sección 2: Probabilidad Binomial",
    "📊 Sección 3: Distribución de Poisson",
    "📏 Sección 4: Intervalos de Confianza"
])

# --------------------------------------------------------
# SECCIÓN 1: Estadística Descriptiva
# --------------------------------------------------------
with tab1:
    st.subheader("📊 Sección 1: Estadística Descriptiva")
    st.write("""
    Esta sección se enfoca en el **análisis descriptivo de datos**.  
    Calcula medidas como **media, mediana, moda, varianza, desviación estándar, cuartiles y deciles**,  
    para **resumir, interpretar y describir** la información contenida en un conjunto de valores numéricos.
    """)

    # ---- Ingreso de datos ----
    st.markdown("### Ingreso de datos")
    modo = st.radio("Selecciona el método de ingreso:", ["Manual", "Archivo (.csv / .txt)"], horizontal=True)

    def parse_text_to_numbers(texto):
        raw = texto.replace(";", ",").replace("\t", ",").replace("\n", ",").replace("  ", " ")
        piezas = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip() != ""]
        nums, errores = [], []
        for p in piezas:
            try:
                nums.append(float(p))
            except Exception:
                errores.append(p)
        return nums, errores

    datos = None

    if modo == "Manual":
        ejemplo = "1.2, 3.4, 2, 5.5, 3, 4.1"
        texto = st.text_area("Pega o escribe tus valores numéricos:", value=ejemplo, height=120)
        if st.button("Calcular medidas"):
            numeros, errores = parse_text_to_numbers(texto)
            if errores:
                st.warning(f"Se ignoraron {len(errores)} valores no numéricos: {errores}")
            datos = np.array(numeros, dtype=float) if numeros else None
    else:
        archivo = st.file_uploader("Sube un archivo .csv o .txt", type=["csv", "txt"])
        sep_opt = st.selectbox("Separador", [",", ";", "\\t (tab)", "Espacio"], index=0)
        sep = {"\\t (tab)": "\t", "Espacio": r"\s+"}.get(sep_opt, sep_opt)
        if archivo is not None:
            try:
                df = pd.read_csv(archivo, sep=sep, engine="python", header=None)
                st.caption("Vista previa del archivo:")
                st.dataframe(df.head())
                columna = st.selectbox("Selecciona la columna con datos numéricos:", options=list(df.columns))
                if st.button("Calcular medidas desde archivo"):
                    serie = pd.to_numeric(df[columna], errors="coerce").dropna()
                    datos = serie.to_numpy()
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

    st.markdown("---")
    st.subheader("Resultados")

    if datos is not None and len(datos) > 0:
        def medidas_basicas(x):
            x = np.asarray(x, dtype=float)
            n = x.size
            mean = float(np.mean(x))
            median = float(np.median(x))
            vals, counts = np.unique(x, return_counts=True)
            maxc = counts.max()
            modes = vals[counts == maxc]
            var = float(np.var(x, ddof=1)) if n > 1 else 0.0
            std = float(np.sqrt(var))
            q1 = float(np.percentile(x, 25))
            q2 = float(np.percentile(x, 50))
            q3 = float(np.percentile(x, 75))
            deciles = {f"D{k}": float(np.percentile(x, k * 10)) for k in range(1, 10)}
            return {"n": n, "mean": mean, "median": median, "modes": modes.tolist(),
                    "variance": var, "std": std, "q1": q1, "q2": q2, "q3": q3, "deciles": deciles}

        res = medidas_basicas(datos)

        c1, c2, c3 = st.columns(3)
        c1.metric("Tamaño (n)", res["n"])
        c2.metric("Media", f"{res['mean']:.4f}")
        c3.metric("Desv. estándar", f"{res['std']:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Mediana (Q2)", f"{res['median']:.4f}")
        c5.metric("Q1", f"{res['q1']:.4f}")
        c6.metric("Q3", f"{res['q3']:.4f}")

        st.subheader("Moda(s)")
        st.write(", ".join(str(m) for m in res["modes"]))

        st.subheader("Deciles (D1–D9)")
        st.table(pd.DataFrame.from_dict(res["deciles"], orient="index", columns=["Valor"]))

        st.markdown("---")
        st.subheader("📘 Interpretación profesional")
        st.markdown("""
        - **Media:** promedio aritmético que resume el valor central del conjunto.  
        - **Mediana (Q2):** divide los datos en dos partes iguales; útil cuando hay valores atípicos.  
        - **Moda:** representa el valor más frecuente.  
        - **Varianza y Desviación estándar:** indican el grado de dispersión de los datos.  
        - **Cuartiles y Deciles:** permiten identificar posiciones relativas dentro de la distribución.  
        - En conjunto, estas medidas permiten **resumir e interpretar** la información cuantitativa.
        """)
    else:
        st.info("Ingresa datos y presiona el botón de cálculo para ver los resultados.")

# --------------------------------------------------------
# SECCIÓN 2: Probabilidad Binomial
# --------------------------------------------------------
with tab2:
    st.subheader("📈 Sección 2: Probabilidad — Distribución Binomial")
    st.write("""
    Calcula la probabilidad de obtener **k éxitos** en **n** ensayos independientes,  
    con una probabilidad de éxito **p** en cada ensayo.
    """)

    n = st.number_input("Número de ensayos (n)", min_value=1, step=1, value=10)
    p = st.number_input("Probabilidad de éxito (p)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    k = st.number_input("Número de éxitos (k)", min_value=0, step=1, value=5)

    if st.button("Calcular probabilidad binomial"):
        prob_exacta = binom.pmf(k, n, p)
        prob_acumulada = binom.cdf(k, n, p)
        media = n * p
        varianza = n * p * (1 - p)
        desviacion = math.sqrt(varianza)

        st.subheader("Resultados")
        st.write(f"**P(X = {k})** = {prob_exacta:.6f}")
        st.write(f"**P(X ≤ {k})** = {prob_acumulada:.6f}")
        st.write(f"**Media (μ)** = {media:.3f}  |  **Varianza (σ²)** = {varianza:.3f}  |  **Desv. Est. (σ)** = {desviacion:.3f}")

        st.markdown("---")
        st.subheader("📘 Interpretación profesional")
        st.markdown(f"""
        - La probabilidad **P(X = {k})** representa la posibilidad exacta de obtener {k} éxitos en {n} ensayos.  
        - La función **acumulada P(X ≤ {k})** indica la probabilidad de obtener hasta {k} éxitos.  
        - La **media (μ)** y **desviación estándar (σ)** describen el comportamiento esperado de la variable binomial.  
        - Este modelo se aplica a **experimentos discretos** donde solo hay dos resultados: éxito o fracaso.
        """)

# --------------------------------------------------------
# SECCIÓN 3: Distribución de Poisson
# --------------------------------------------------------
with tab3:
    st.subheader("📊 Sección 3: Probabilidad — Distribución de Poisson")
    st.write("""
    Modela el número de eventos que ocurren en un intervalo fijo de tiempo o espacio,  
    cuando los eventos son **raros**, **independientes** y ocurren a una **tasa constante (λ)**.
    """)

    λ = st.number_input("Promedio de ocurrencias (λ)", min_value=0.0, step=0.1, value=2.0)
    k_pois = st.number_input("Número de eventos (k)", min_value=0, step=1, value=3)

    if st.button("Calcular probabilidad Poisson"):
        prob_exacta = poisson.pmf(k_pois, λ)
        prob_acumulada = poisson.cdf(k_pois, λ)

        st.subheader("Resultados")
        st.write(f"**P(X = {k_pois})** = {prob_exacta:.6f}")
        st.write(f"**P(X ≤ {k_pois})** = {prob_acumulada:.6f}")
        st.write(f"**Media (μ)** = {λ:.3f}  |  **Varianza (σ²)** = {λ:.3f}  |  **Desv. Est. (σ)** = {math.sqrt(λ):.3f}")

        st.markdown("---")
        st.subheader("📘 Interpretación profesional")
        st.markdown(f"""
        - **P(X = {k_pois})** representa la probabilidad exacta de que ocurran {k_pois} eventos en el intervalo.  
        - **P(X ≤ {k_pois})** indica la probabilidad acumulada de que ocurran hasta {k_pois} eventos.  
        - En Poisson, la **media y la varianza son iguales (μ = σ² = λ)**.  
        - Se usa en fenómenos como llamadas telefónicas, llegadas a un servicio o defectos por unidad.  
        """)

# --------------------------------------------------------
# SECCIÓN 4: Intervalos de Confianza
# --------------------------------------------------------
with tab4:
    st.subheader("📏 Sección 4: Intervalos de Confianza")
    st.write("""
    Permite estimar parámetros poblacionales como la **media (μ)** o la **proporción (p)**,  
    a partir de datos muestrales. El **nivel de confianza (90 %, 95 %, 99 %)**  
    indica la **probabilidad de que el intervalo contenga el valor real** del parámetro.
    """)

    tipo = st.radio("Selecciona el tipo de parámetro:", ["Media poblacional", "Proporción poblacional"], horizontal=True)
    confianza = st.selectbox("Nivel de confianza:", ["90%", "95%", "99%"], index=1)
    z_valores = {"90%": 1.645, "95%": 1.96, "99%": 2.575}
    z = z_valores[confianza]

    if tipo == "Media poblacional":
        x_bar = st.number_input("Media muestral (x̄)", step=0.1, value=50.0)
        s = st.number_input("Desviación estándar (s)", step=0.1, value=10.0)
        n = st.number_input("Tamaño de muestra (n)", min_value=1, step=1, value=30)

        if st.button("Calcular IC para la Media"):
            error = z * (s / math.sqrt(n))
            li, ls = x_bar - error, x_bar + error
            st.write(f"**IC para μ:** ({li:.3f}, {ls:.3f}) — Nivel {confianza}")
            st.write(f"**Margen de error:** ±{error:.3f}")

            st.markdown("---")
            st.subheader("📘 Interpretación profesional")
            st.markdown(f"""
            - Con un **nivel de confianza del {confianza}**, se estima que la **media poblacional (μ)**  
              se encuentra entre **{li:.3f}** y **{ls:.3f}**.  
            - No significa que μ "cae" en ese rango con probabilidad {confianza};  
              significa que, si repitiéramos muchos muestreos, **{confianza} de los intervalos** construidos  
              incluirían el valor real de μ.  
            - **Efectos sobre el ancho del intervalo:**  
              - Mayor **n** → menor margen de error.  
              - Mayor **s** o nivel de confianza → intervalo más **ancho**.  
              - Muestra aleatoria e independiente, y si n ≥ 30, el uso del modelo normal es válido (TLC).
            """)

    else:
        p_hat = st.number_input("Proporción muestral (p̂)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
        n = st.number_input("Tamaño de muestra (n)", min_value=1, step=1, value=100)

        if st.button("Calcular IC para la Proporción"):
            error = z * math.sqrt(p_hat * (1 - p_hat) / n)
            li, ls = p_hat - error, p_hat + error
            st.write(f"**IC para p:** ({li:.3f}, {ls:.3f}) — Nivel {confianza}")
            st.write(f"**Margen de error:** ±{error:.3f}")

            st.markdown("---")
            st.subheader("📘 Interpretación profesional")
            st.markdown(f"""
            - Con un **nivel de confianza del {confianza}**, se estima que la **proporción real (p)**  
              está entre **{li:.3f}** y **{ls:.3f}**.  
            - Si repitiéramos el proceso de muestreo muchas veces, aproximadamente **{confianza} de los intervalos**  
              incluirían la proporción poblacional verdadera.  
            - **Factores que influyen en el ancho del IC:**  
              - Mayor **n** → intervalo más estrecho.  
              - Proporciones cercanas a 0.5 → intervalos más amplios.  
              - Mayor **nivel de confianza** → intervalo más ancho.  
              - Condición normal: **n·p̂ ≥ 10** y **n·(1−p̂) ≥ 10**.
            """)

# if __name__ == "__main__":

#     # Ruta absoluta del script principal
#     script_path = os.path.abspath(__file__)

#     # Ejecuta Streamlit directamente
#     subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_path])
