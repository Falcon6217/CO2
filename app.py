import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="CO₂ Emission Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# DIRECTORY & PATH RESOLUTION
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "co2_emission_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "sustainability_data.csv")
TEST_RESULTS_PATH = os.path.join(MODEL_DIR, "test_results.csv")

# ==========================================================
# CUSTOM CSS STYLING
# ==========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 8px;
    }
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .result-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        color: #ffffff;
        padding: 24px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        margin: 20px 0;
    }
    .result-value {
        font-size: 42px;
        font-weight: 800;
        margin: 10px 0;
        letter-spacing: -0.5px;
    }
    .badge {
        display: inline-block;
        padding: 6px 14px;
        font-size: 14px;
        font-weight: 600;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.2);
        margin-top: 6px;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# HELPER FUNCTIONS & MODEL MANAGEMENT
# ==========================================================

def train_and_save_model():
    """Trains the Polynomial Regression model and saves it to disk."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    required_cols = ["Energy_Consumption", "Renewable_Percentage", "GDP", "CO2_Emissions"]
    df = df.dropna(subset=required_cols)
    
    X = df[["Energy_Consumption", "Renewable_Percentage", "GDP"]]
    y = df["CO2_Emissions"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    new_model = Pipeline(
        steps=[
            ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("linear_regression", LinearRegression())
        ]
    )
    new_model.fit(X_train, y_train)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(new_model, MODEL_PATH)
    
    # Save test results
    y_pred = new_model.predict(X_test)
    test_res = pd.DataFrame({
        "Actual_CO2_Emissions": y_test.values,
        "Predicted_CO2_Emissions": y_pred
    })
    test_res.to_csv(TEST_RESULTS_PATH, index=False)
    
    return new_model

@st.cache_resource
def load_model():
    """Loads the trained ML model from disk."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_historical_data():
    """Loads historical dataset for benchmarking."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

# ==========================================================
# APP HEADER
# ==========================================================

st.markdown('<div class="main-title">🌍 CO₂ Emission Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-Powered Machine Learning Model estimating CO₂ emissions based on Energy, Renewables, and GDP</div>',
    unsafe_allow_html=True
)

# Load dataset and model
historical_df = load_historical_data()
model = load_model()

# ==========================================================
# SIDEBAR CONFIGURATION & ABOUT
# ==========================================================

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=400&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("🌱 About the Project")
    st.write(
        """
        This sustainability application uses a **Polynomial Regression (Degree 2)** pipeline 
        to model complex non-linear relationships between socioeconomic indicators and carbon emissions:
        
        - ⚡ **Energy Consumption**: Total energy utilized (GWh / Mtoe)
        - 🍃 **Renewable Percentage**: Share of green & clean energy (%)
        - 💰 **GDP**: Gross Domestic Product ($ Billions)
        """
    )
    
    st.markdown("---")
    st.subheader("⚙️ Model Status")
    if model is not None:
        st.success("✅ Model is loaded and ready")
    else:
        st.warning("⚠️ Model not found")
        if st.button("🚀 Train Model Now", use_container_width=True):
            with st.spinner("Training model on dataset..."):
                model = train_and_save_model()
                st.cache_resource.clear()
                st.success("Model trained and loaded successfully!")
                st.rerun()

    st.markdown("---")
    st.caption("Built with Python • Scikit-Learn • Streamlit")

# Guard if model is missing
if model is None:
    st.error("❌ Trained model file was not found!")
    st.info("Click the **'🚀 Train Model Now'** button in the sidebar or run `python train.py` in your terminal.")
    st.stop()

# ==========================================================
# MAIN INTERFACE TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 Dataset & Insights", "📈 Batch CSV Prediction"])

# ----------------------------------------------------------
# TAB 1: SINGLE PREDICTION
# ----------------------------------------------------------
with tab1:
    st.subheader("📊 Enter Environmental & Economic Parameters")
    
    # Calculate dataset medians for sensible defaults
    default_energy = float(historical_df["Energy_Consumption"].median()) if historical_df is not None else 520.0
    default_renew = float(historical_df["Renewable_Percentage"].median()) if historical_df is not None else 25.0
    default_gdp = float(historical_df["GDP"].median()) if historical_df is not None else 1450.0

    col1, col2, col3 = st.columns(3)
    
    with col1:
        energy_consumption = st.number_input(
            "⚡ Energy Consumption",
            min_value=0.0,
            max_value=10000.0,
            value=default_energy,
            step=5.0,
            help="Total energy consumption index or units."
        )

    with col2:
        renewable_percentage = st.number_input(
            "🌱 Renewable Energy (%)",
            min_value=0.0,
            max_value=100.0,
            value=round(default_renew, 1),
            step=0.5,
            help="Percentage of energy originating from renewable sources."
        )

    with col3:
        gdp = st.number_input(
            "💰 Gross Domestic Product (GDP)",
            min_value=0.0,
            max_value=100000.0,
            value=default_gdp,
            step=50.0,
            help="GDP indicator in standard monetary units."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🔮 Calculate Estimated CO₂ Emissions", type="primary", use_container_width=True)

    if predict_button:
        input_data = pd.DataFrame({
            "Energy_Consumption": [energy_consumption],
            "Renewable_Percentage": [renewable_percentage],
            "GDP": [gdp]
        })

        try:
            prediction = model.predict(input_data)
            predicted_co2 = float(prediction[0])
            
            # Determine Emission Level Badge
            if historical_df is not None and "CO2_Emissions" in historical_df.columns:
                avg_co2 = historical_df["CO2_Emissions"].mean()
                diff_pct = ((predicted_co2 - avg_co2) / avg_co2) * 100
                diff_text = f"{'+' if diff_pct >= 0 else ''}{diff_pct:.1f}% vs Historical Average ({avg_co2:.1f})"
            else:
                diff_text = "Standard Assessment"

            if predicted_co2 < 850:
                badge_label = "🟢 Low Emission Profile"
            elif predicted_co2 < 930:
                badge_label = "🟡 Moderate Emission Profile"
            else:
                badge_label = "🔴 High Emission Profile"

            st.markdown(
                f"""
                <div class="result-card">
                    <div style="font-size: 18px; opacity: 0.9;">🌍 Estimated CO₂ Emissions</div>
                    <div class="result-value">{predicted_co2:,.2f} <span style="font-size: 20px; font-weight: normal;">Units</span></div>
                    <div class="badge">{badge_label}</div>
                    <div style="font-size: 13px; margin-top: 8px; opacity: 0.85;">{diff_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Details & Metrics
            st.markdown("### 📋 Parameter Breakdown")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric(label="Energy Consumption", value=f"{energy_consumption:,.1f}")
            with mcol2:
                st.metric(label="Renewable Share", value=f"{renewable_percentage:.1f}%")
            with mcol3:
                st.metric(label="GDP Index", value=f"${gdp:,.1f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

# ----------------------------------------------------------
# TAB 2: DATASET & HISTORICAL INSIGHTS
# ----------------------------------------------------------
with tab2:
    st.subheader("📚 Historical Sustainability Data")
    if historical_df is not None:
        st.dataframe(historical_df, use_container_width=True)
        
        st.markdown("### 📈 Summary Statistics")
        st.write(historical_df.describe())
        
        if "Year" in historical_df.columns and "CO2_Emissions" in historical_df.columns:
            st.markdown("### 📉 Historical CO₂ Emission Trend")
            chart_data = historical_df.set_index("Year")[["CO2_Emissions", "Energy_Consumption"]]
            st.line_chart(chart_data)
    else:
        st.info("No historical dataset found at `sustainability_data.csv`.")

# ----------------------------------------------------------
# TAB 3: BATCH PREDICTIONS (CSV)
# ----------------------------------------------------------
with tab3:
    st.subheader("📁 Upload CSV for Batch Predictions")
    st.write("Upload a CSV file containing `Energy_Consumption`, `Renewable_Percentage`, and `GDP` columns.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            required_inputs = ["Energy_Consumption", "Renewable_Percentage", "GDP"]
            
            if all(col in batch_df.columns for col in required_inputs):
                batch_predictions = model.predict(batch_df[required_inputs])
                batch_df["Predicted_CO2_Emissions"] = np.round(batch_predictions, 2)
                
                st.success(f"✅ Successfully generated predictions for {len(batch_df)} records!")
                st.dataframe(batch_df, use_container_width=True)
                
                # Download button
                csv_output = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_output,
                    file_name="co2_predictions_output.csv",
                    mime="text/csv"
                )
            else:
                missing = [col for col in required_inputs if col not in batch_df.columns]
                st.error(f"❌ Missing required columns: {missing}")
        except Exception as err:
            st.error(f"Error processing CSV: {err}")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.caption("CO₂ Emission Predictor • Machine Learning Regression Model • Powered by Streamlit")