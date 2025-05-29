import streamlit as st
import pandas as pd
import joblib
import json

# ----------------------------
# CARGA DE MODELOS Y ARCHIVOS
# ----------------------------
modelo_rf = joblib.load("modelo_rf.pkl")
df_reglas = joblib.load("df_reglas.pkl")

with open("productos.json", "r") as f:
    todos_los_productos = json.load(f)

# ----------------------------
# FUNCIÓN DE SUGERENCIA
# ----------------------------
def sugerir_producto_tabla(carrito_actual, descuento_actual, modelo, reglas_df):
    reglas_sencillas = reglas_df[
        (reglas_df['Base'].str.count(',') == 0) & (reglas_df['Agregar'].str.count(',') == 0)
    ].copy()

    reglas_aplicables = reglas_sencillas[reglas_sencillas['Base'].isin(carrito_actual)]

    if reglas_aplicables.empty:
        return pd.DataFrame(columns=[
            'Base', 'Recomendar', 'Confianza Apriori', 'Lift',
            'Descuento Promedio (%)', 'Probabilidad Aceptación (%)'
        ])

    resultados = []

    for _, regla in reglas_aplicables.iterrows():
        producto_sugerido = regla['Agregar']
        input_data = {f'carrito_{p}': int(p in carrito_actual) for p in todos_los_productos}
        input_data['Descuento_Ofertado'] = descuento_actual
        input_data['Num_Productos'] = len(carrito_actual)
        input_data['Ticket_Promedio'] = 10000  # Valor de ejemplo

        X_pred = pd.DataFrame([input_data])
        X_pred = X_pred.reindex(columns=modelo.feature_names_in_, fill_value=0)

        prob = modelo.predict_proba(X_pred)[0][1]

        resultados.append({
            'Base': regla['Base'],
            'Recomendar': producto_sugerido,
            'Confianza Apriori': round(regla['Confianza'], 2),
            'Lift': round(regla['Lift'], 2),
            'Descuento Promedio (%)': regla['Descuento Promedio (%)'],
            'Probabilidad Aceptación (%)': round(prob * 100, 2)
        })

    return pd.DataFrame(resultados).sort_values(by='Probabilidad Aceptación (%)', ascending=False)

# ----------------------------
# INTERFAZ STREAMLIT
# ----------------------------
st.title("🛒 Recomendación de Productos con Apriori + XGBoost")

# Selector múltiple para carrito simulado
carrito_usuario = st.multiselect("Selecciona los productos que el cliente ya tiene en su carrito:", options=todos_los_productos)

# Descuento simulado
descuento_usuario = st.selectbox("Selecciona el % de descuento ofrecido:", [0, 10, 15, 20, 30])

# Botón de ejecución
if st.button("💡 Sugerir producto"):
    if carrito_usuario:
        resultado = sugerir_producto_tabla(carrito_usuario, descuento_usuario, modelo_rf, df_reglas)
        if resultado.empty:
            st.warning("🤷 No se encontraron reglas aplicables para este carrito.")
        else:
            st.success("✅ Recomendaciones generadas:")
            st.dataframe(resultado)
    else:
        st.warning("⚠️ Por favor selecciona al menos un producto para simular el carrito.")
