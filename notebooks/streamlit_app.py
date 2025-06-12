import streamlit as st

# --- Page config ---
st.set_page_config(page_title="Accident Severity Prediction", page_icon="🚗")

import joblib
import pandas as pd

# --- Load resources ---

model = joblib.load('E:/accident_severity_analytics/models/model.pkl')
le_dict = joblib.load('E:/accident_severity_analytics/models/encoders.pkl')
columns = joblib.load('E:/accident_severity_analytics/models/columns.pkl')

print("✅ Files loaded successfully.")

st.title("🚗 Predicting the severity of accidents in America")
st.markdown("Enter the following data to obtain a prediction of the severity of the accident:")

user_input = {}

for col in columns:
    if col in le_dict:
        options = list(le_dict[col].classes_)
        selected = st.selectbox(f"{col}:", options)
        user_input[col] = le_dict[col].transform([selected])[0]
    else:
        user_input[col] = st.number_input(f"{col}:", step=1.0)

if st.button("🔍 Anticipate danger"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    severity_map = {
        1: "🔵 low",
        2: "🟢 medium",
        3: "🟠 High",
        4: "🔴 Very intense"
    }

    st.subheader("🌟 Result:")
    st.success(f"Expected severity: {severity_map.get(prediction, prediction)}")

    # --- CSV Download ---
    csv = input_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Result as CSV",
        data=csv,
        file_name='accident_prediction_result.csv',
        mime='text/csv'
    )

#streamlit run E:/accident_severity_analytics/notebooks/streamlit_app.py