import streamlit as st
import pandas as pd
import joblib
import json

# --- CARGA DE ARCHIVOS ---
clf = joblib.load("modelo_rf.pkl")
df_reglas = pd.read_pickle("df_reglas.pkl")
with open("productos.json") as f:
    todos_los_productos = json.load(f)

# --- INTERFAZ ---
st.title("🛒 Simulador de carrito inteligente - Apriori + Random Forest")

carrito = st.multiselect("Selecciona productos que llevaría el cliente:", todos_los_productos)
descuento = st.slider("Descuento ofrecido para el producto sugerido (%)", 0, 50, 15)

# --- FUNCIÓN DE RECOMENDACIÓN ---
def sugerir_producto_tabla(carrito_actual, descuento_actual, modelo, reglas_df):
    reglas_sencillas = reglas_df[
        (reglas_df['Base'].str.count(',') == 0) &
        (reglas_df['Agregar'].str.count(',') == 0)
    ].copy()

    reglas_aplicables = reglas_sencillas[reglas_sencillas['Base'].isin(carrito_actual)]
    
    if reglas_aplicables.empty:
        return pd.DataFrame()

    resultados = []
    for _, regla in reglas_aplicables.iterrows():
        producto_sugerido = regla['Agregar']
        input_data = {f'carrito_{p}': int(p in carrito_actual) for p in todos_los_productos}
        input_data['Descuento_Ofertado'] = descuento_actual
        input_data['Num_Productos'] = len(carrito_actual)
        input_data['Ticket_Promedio'] = 10000  # valor fijo de ejemplo

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

    return pd.DataFrame(resultados).sort_values(by='Probabilidad Aceptación (%)', ascending=False).reset_index(drop=True)

# --- BOTÓN PARA RECOMENDAR ---
if st.button("Generar recomendación"):
    if carrito:
        resultado = sugerir_producto_tabla(carrito, descuento, clf, df_reglas)
        if not resultado.empty:
            st.success("✅ Recomendación generada:")
            st.dataframe(resultado)
        else:
            st.warning("🤷 No hay asociaciones compatibles con el carrito.")
    else:
        st.warning("⚠️ Debes seleccionar al menos un producto.")
