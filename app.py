
# part1
# ==========================================================
# HOUSE PRICE PREDICTION SYSTEM
# Developed using Streamlit + Scikit-Learn
# ==========================================================

# Import Libraries
import streamlit as st
import joblib
import numpy as np

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# LOAD CSS
# ==========================================================

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# ==========================================================
# LOAD MACHINE LEARNING MODEL
# ==========================================================

model = joblib.load("house_price_model.pkl")

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""
<div class="hero">
    <h1>🏠 House Price Prediction</h1>
    <p>
        Predict the estimated market price of your house
        using Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# FORM CARD START
# ==========================================================

st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("📋 Enter House Details")

st.write("Fill in the details below to predict the estimated house price.")

# ==========================================================
# INPUT FIELDS
# ==========================================================

area = st.number_input(
    "📐 Area (Square Feet)",
    min_value=500,
    max_value=20000,
    value=1000,
    step=50
)

bedrooms = st.number_input(
    "🛏 Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "🛁 Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

stories = st.number_input(
    "🏢 Number of Stories",
    min_value=1,
    max_value=5,
    value=2
)

parking = st.number_input(
    "🚗 Parking Spaces",
    min_value=0,
    max_value=5,
    value=1
)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# PREDICT BUTTON
# ==========================================================

predict = st.button("🔮 Predict House Price")


# part2

# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    # Create feature array
    features = np.array([
        [
            area,
            bedrooms,
            bathrooms,
            stories,
            parking
        ]
    ])

    # Predict
    prediction = model.predict(features)

    # Display Result
    st.markdown(
        f"""
        <div class="result">
            <h2>🎉 Prediction Successful</h2>
            <h1>Rs. {prediction[0]:,.2f}</h1>
            <p>
                This is the estimated market price based on the
                details you entered.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# CLOSE FORM CARD
# ==========================================================

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# DIVIDER
# ==========================================================

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================================
# FOOTER


st.markdown("""
<div class="footer">
    <h4>🏠 House Price Prediction System</h4>
    <p>Powered by Machine Learning using Python & Streamlit.</p>
    <p>© 2026 | Developed by <b> Bibi_Tabassum</b></p>
</div>
""", unsafe_allow_html=True)

