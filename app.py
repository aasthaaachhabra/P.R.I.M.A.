"""
P.R.I.M.A. - Premium Risk Indexing and Modeling for Agriculture (v14.2 - Optimized Defaults)
=================================================================================================
A financial-grade business application with a dark navy theme, using traditional county-based
benchmarks, an Orchard Health Score, Economic Interpretation, and a Dynamic Premium Adjustment model.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow.keras as keras
import shap

# --- Page Setup & Global Configuration ---
st.set_page_config(
    page_title="P.R.I.M.A. | Agricultural Risk Modeling",
    page_icon=" ",
    layout="wide"
)

#<editor-fold desc="CSS Styling and UI Components">
def load_corporate_theme_css():
    """
    Loads a professional, corporate theme with a navy blue background and light blue/white text.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* --- General Styles & Typography --- */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0a192f; /* Navy Blue background */
        color: #ccd6f6; /* Light Blue/White text */
    }
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4 {
        color: #ffffff; /* White headers */
    }

    /* --- Content Cards & Sections --- */
    .section-container {
        background-color: #112240; /* Lighter Navy for cards */
        padding: 2.5rem;
        border-radius: 12px;
        border: 1px solid #233554; /* Border color */
        margin-top: 2rem;
    }
    .section-container h2 {
        border-bottom: 2px solid #64ffda; /* Light Blue accent */
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        color: #64ffda;
    }

    /* --- Metrics & Data Displays --- */
    [data-testid="stMetric"] {
        background-color: #0a192f; /* Dark Navy for metric background */
        border: 1px solid #233554;
        border-radius: 10px;
        padding: 1.5rem;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #8892b0; /* Lighter secondary text */
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    [data-testid="stMetricDelta"] {
        color: #64ffda; /* Light blue for delta */
    }

    /* --- Interactive Widgets --- */
    .st-emotion-cache-1r6slb0, .st-emotion-cache-1y4p8pa { /* Text input, selectbox */
        background-color: #112240;
    }
    .st-emotion-cache-1v0f73p { /* Slider track */
        background-color: #233554;
    }
    .st-emotion-cache-13k6pro { /* Slider thumb */
        background-color: #64ffda;
    }

    /* --- Buttons --- */
    .stButton > button {
        background-color: transparent;
        color: #64ffda; /* Light Blue text */
        border: 2px solid #64ffda; /* Light Blue border */
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        width: auto;
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
        float: right;
    }
    .stButton > button:hover {
        background-color: #64ffda; /* Light Blue background on hover */
        color: #0a192f; /* Dark text on hover */
    }

    /* --- Header & Title --- */
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    .title-container h1 {
        font-size: 2.75rem;
        font-weight: 700;
        letter-spacing: -1px;
    }
    .title-container h2 {
        font-size: 1.75rem;
        font-weight: 500;
        color: #8892b0; /* Lighter secondary text for subtitle */
    }

    /* --- Hide Streamlit Branding --- */
    #MainMenu, footer, .stDeployButton, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

def display_prima_header():
    """Renders the main header as specified by the user."""
    st.markdown(
        """
        <div class="title-container">
            <h1>P.R.I.M.A.: Premium Risk Indexing and Modeling for Agriculture</h1>
            <h2>Dynamic Premium Adjustment Model</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
#</editor-fold>

#<editor-fold desc="Core Logic & Backend Calculations">
@st.cache_resource
def load_artifacts():
    """Loads all necessary model files and data once."""
    try:
        artifacts = {
            'preprocessor': joblib.load('preprocessor.joblib'),
            'cart_model': joblib.load('cart_model.joblib'),
            'xgb_model': joblib.load('xgb_model.joblib'),
            'nn_model': keras.models.load_model('nn_model.keras'),
            'definitive_weights': np.load('definitive_weights.npy'),
            'context_data': joblib.load('context_data.joblib'),
            'training_data': pd.read_csv('training_data.csv'),
            'shap_background': pd.read_csv('shap_background_data.csv')
        }
        # Cache the SHAP explainer for performance
        artifacts['shap_explainer'] = shap.TreeExplainer(artifacts['xgb_model'], artifacts['shap_background'])
        return artifacts
    except FileNotFoundError:
        st.error("Fatal Error: Critical model files not found. Please ensure all 8 artifacts are in the application's root directory.", icon="🚨")
        return None

@st.cache_data
def get_premium_and_scores(_artifacts, user_inputs_tuple):
    """Performs all backend calculations, using traditional county-rate benchmark."""
    user_inputs = dict(user_inputs_tuple)
    user_df = pd.DataFrame([user_inputs])

    # --- Data Preparation ---
    df_predict = user_df.copy()
    guaranteed_yield = df_predict['APH_Yield_lbs_per_acre'].iloc[0] * df_predict['Coverage_Level_pct'].iloc[0]
    df_predict['Guaranteed_Yield_lbs_per_acre'] = guaranteed_yield

    # --- Phenology-Based Feature Engineering ---
    df_predict['Dormancy_Chill'] = df_predict['Total_Chill_Day_Units']
    df_predict['Dormancy_Frost_Risk'] = df_predict['Total_Frost_Days'] * 0.5
    df_predict['Bloom_Frost_Risk'] = df_predict['Total_Frost_Days'] * 0.5
    df_predict['Bloom_Rain_Risk'] = df_predict['Total_Precip_in'] * 0.3
    df_predict['Fruit_Growth_Heat_Risk'] = df_predict['Total_Heat_Stress_Days'] * 0.6
    df_predict['Hull_Split_Heat_Risk'] = df_predict['Total_Heat_Stress_Days'] * 0.4
    df_predict['Harvest_Rain_Risk'] = df_predict['Total_Precip_in'] * 0.2


    # --- P.R.I.M.A. AI Model Prediction ---
    context = _artifacts['context_data']
    preprocessor = _artifacts['preprocessor']
    final_input = pd.DataFrame(preprocessor.transform(df_predict), columns=preprocessor.get_feature_names_out())[context['selected_features']]

    cart_p = _artifacts['cart_model'].predict_proba(final_input)[:, 1][0]
    xgb_p = _artifacts['xgb_model'].predict_proba(final_input)[:, 1][0]
    nn_p = _artifacts['nn_model'].predict(final_input.values, verbose=0).flatten()[0]
    claim_probability = np.mean([cart_p, xgb_p, nn_p]) # Ensemble prediction

    liability_per_acre = guaranteed_yield * df_predict['Price_Election_per_lb'].iloc[0]
    expected_loss = claim_probability * liability_per_acre
    dynamic_premium = expected_loss * 1.4 # Applying a loss cost ratio of 1.4

    # --- Traditional Premium Calculation (County Actuarial Rate) ---
    county = user_df['County'].iloc[0]

    # This method simulates looking up the actuarial rate for the county from the historical data.
    # It finds the median premium rate per acre for the given county.
    county_rate = _artifacts['training_data'][_artifacts['training_data']['County'] == county]['Premium_Rate_per_acre'].median()

    # If the county is not found or has no data, fall back to the global median rate.
    if pd.isna(county_rate):
        county_rate = _artifacts['training_data']['Premium_Rate_per_acre'].median()

    traditional_premium = county_rate

    # --- Health Score & Other Metrics ---
    orchard_health_score = (1 - claim_probability) * 100
    prediction_std = np.std([cart_p, xgb_p, nn_p])
    model_confidence = max(0, 1 - (prediction_std / 0.5))
    shap_values = _artifacts['shap_explainer'](final_input)

    return {
        "dynamic_premium": dynamic_premium,
        "traditional_premium": traditional_premium, # Use traditional premium
        "claim_probability": claim_probability,
        "model_confidence": model_confidence,
        "orchard_health_score": orchard_health_score,
        "liability_per_acre": liability_per_acre,
        "shap_values": shap_values,
        "final_input": final_input
    }
#</editor-fold>

# --- UI Helper Functions ---
def st_shap(plot, height=None):
    """Helper to render SHAP plots in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- Main Application ---
def main():
    load_corporate_theme_css()
    display_prima_header()

    artifacts = load_artifacts()
    if artifacts is None:
        st.stop()

    # --- SECTION 1: USER INPUTS ---
    with st.container():
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.header("Step 1: Complete Your Farm Profile")
        st.write("Provide the following details about your operation and seasonal forecast to generate a personalized insurance quote. All fields are required.")

        # Subsection for Farm and Management Details
        st.subheader("Orchard & Management Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            user_inputs = {
                # Set default to a major county
                'County': st.selectbox("County", artifacts['context_data']['valid_counties'], index=artifacts['context_data']['valid_counties'].index('Fresno')),
                'Almond_Variety': st.selectbox("Almond Variety", artifacts['context_data']['valid_varieties']),
                # NEW DEFAULT: Increased experience
                'Farmer_Experience_years': st.slider("Farming Experience (Years)", 0, 60, 25),
            }
        with col2:
            user_inputs.update({
                # NEW DEFAULT: Optimal tree age
                'Tree_Age': st.slider("Average Tree Age", 4, 40, 18),
                'Planting_Density_trees_per_acre': st.slider("Planting Density (trees/acre)", 80, 200, 120),
                'Orchard_Size_acres': st.number_input("Orchard Size (acres)", 1.0, 1000.0, 50.0),
            })
        with col3:
            user_inputs.update({
                # NEW DEFAULT: More efficient irrigation
                'Irrigation_System_Type': st.selectbox("Irrigation System", artifacts['context_data']['valid_irrigation'], index=artifacts['context_data']['valid_irrigation'].index('Micro-sprinkler')),
                # NEW DEFAULT: Higher productivity
                'APH_Yield_lbs_per_acre': st.number_input("Approved Average Yield (lbs/acre)", 500, 5000, 2800),
                'Mgmt_Practices_IPM': 1 if st.radio("Integrated Pest Management (IPM) Used?", ('Yes', 'No'), horizontal=True, index=0) == 'Yes' else 0,
            })

        st.markdown("<hr style='border-color: #233554;'>", unsafe_allow_html=True)

        # Subsection for Insurance and Forecast Details
        col4, col5 = st.columns(2)
        with col4:
            st.subheader("Insurance Coverage Details")
            user_inputs.update({
                'Coverage_Level_pct': st.slider("Desired Coverage Level (%)", 50, 85, 63, help="The percentage of your approved yield you wish to insure.") / 100.0,
                'Price_Election_per_lb': st.slider("Price Election ($/lb)", 1.0, 4.0, 2.5, step=0.05, help="The price per pound used to calculate your insured value."),
            })
        with col5:
            st.subheader("Seasonal Weather Forecast")
            user_inputs.update({
                'Total_Chill_Day_Units': st.slider("Forecasted Chill Units", 0, 100, 20),
                # NEW DEFAULT: Lower weather risk
                'Total_Frost_Days': st.slider("Forecasted Frost Days", 0, 50, 5),
                'Total_Precip_in': st.slider("Forecasted Precipitation (inches)", 0.0, 50.0, 15.0),
                # NEW DEFAULT: Lower weather risk
                'Total_Heat_Stress_Days': st.slider("Forecasted Heat Stress Days", 0, 150, 25),
                # NEW DEFAULT: Lower weather risk
                'Avg_VPD_kPa': st.slider("Forecasted Air Dryness (Avg. VPD, kPa)", 0.1, 2.0, 0.5),
            })

        # Add fixed/hidden inputs
        user_inputs['Pollination_Management'] = 'Managed'
        user_inputs['Policy_Unit_Structure'] = 'Basic'

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Run Analysis Button ---
    run_button = st.button("Generate Premium Analysis", type="primary")

    if run_button:
        with st.spinner("Analyzing your farm's unique risk profile... This may take a moment."):
            results = get_premium_and_scores(artifacts, tuple(user_inputs.items()))
            st.session_state.results = results
            st.session_state.inputs = user_inputs

    # --- SECTION 2: RESULTS & INTERPRETATION ---
    if 'results' in st.session_state:
        results = st.session_state.results
        inputs = st.session_state.inputs
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.header("Step 2: Review Your Analysis")

        # --- Health Score and Claim Probability ---
        st.subheader("Orchard Risk Dashboard")
        colA, colB = st.columns(2)
        colA.metric(
            "Orchard Health Score",
            f"{results['orchard_health_score']:.1f} / 100",
            help="A score of 100 represents the lowest possible risk. Higher scores are better and lead to lower premiums."
        )
        colB.metric(
            "AI-Predicted Claim Probability",
            f"{results['claim_probability']:.2%}",
            help="This is the likelihood of a claim being filed based on your specific data. Lower is better."
        )
        st.info(
            "**Interpretation:** The **Health Score** is an overall measure of your orchard's resilience. It is directly influenced by the **Claim Probability**, which is the core risk assessment from our AI. A high Health Score indicates that your management practices and forecasted conditions are favorable.", icon="💡"
        )
        st.markdown("<hr style='border-color: #233554;'>", unsafe_allow_html=True)

        # --- Premium Comparison and Summary ---
        st.subheader("Premium Quote Summary")
        diff = results['dynamic_premium'] - results['traditional_premium']
        col1, col2, col3 = st.columns(3)
        col1.metric("Your P.R.I.M.A. Premium", f"${results['dynamic_premium']:.2f}", "per acre")
        col2.metric("Traditional County Rate", f"${results['traditional_premium']:.2f}", "per acre")
        col3.metric("Your Potential Savings", f"${-diff:.2f}", f"{(-diff/results['traditional_premium'])*100:.1f}% vs. Average", delta_color="inverse")

        st.info(
            "**Benchmark Interpretation:** The **Traditional County Rate** is based on the median premium rate from historical actuarial data for your selected county. It serves as a standard benchmark to measure the fairness of your personalized P.R.I.M.A. premium.", icon="💡"
        )
        st.markdown("<hr style='border-color: #233554;'>", unsafe_allow_html=True)

        # --- Economic Interpretation Section ---
        st.subheader("Economic Interpretation & Business Impact")
        liability_per_acre = results['liability_per_acre']
        premium_roi = liability_per_acre / results['dynamic_premium'] if results['dynamic_premium'] > 0 else 0
        breakeven_yield_loss = results['dynamic_premium'] / inputs['Price_Election_per_lb'] if inputs['Price_Election_per_lb'] > 0 else 0

        econ_col1, econ_col2, econ_col3 = st.columns(3)
        econ_col1.metric(
            "Total Coverage (Liability)",
            f"${liability_per_acre:,.2f} / acre",
            help="This is the maximum potential payout per acre in the event of a total covered loss."
        )
        econ_col2.metric(
            "Coverage per Dollar Spent",
            f"${premium_roi:.2f}",
            help="For every $1 spent on your premium, you receive this amount in potential coverage."
        )
        econ_col3.metric(
            "Breakeven Yield Loss",
            f"{breakeven_yield_loss:.1f} lbs/acre",
            help="A yield loss of this amount would result in a claim payout equal to your premium."
        )

        st.info(
            "**Interpretation:** These metrics translate your premium into tangible business terms. The **Total Coverage** represents your financial safety net. The **Coverage per Dollar Spent** acts as an ROI on your risk management investment. The **Breakeven Yield Loss** provides a clear threshold for when your insurance policy begins to pay for itself in a given season.", icon="💡"
        )
        st.markdown("<hr style='border-color: #233554;'>", unsafe_allow_html=True)

        # --- Risk Deep Dive ---
        st.subheader("Risk Factor Analysis")
        st.info(
            "**How to Read This Chart:** **Red bars** represent factors that increased your risk and premium. **Blue bars** represent factors that decreased it. The length of the bar indicates the magnitude of the impact. The 'base value' is the average model prediction.", icon="💡"
        )
        shap_plot_object = shap.force_plot(artifacts['shap_explainer'].expected_value, results['shap_values'].values, results['final_input'])
        st_shap(shap_plot_object, 200)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- SECTION 3: DYNAMIC PREMIUM ADJUSTMENT ---
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.header("Step 3: Track Your Premium Throughout the Season")
        st.write("As the season unfolds, update the sliders below with **actual observed weather data** to dynamically adjust your premium, reflecting the true risk as it evolves.")

        adjusted_inputs = st.session_state.inputs.copy()
        last_premium = results['dynamic_premium']

        stages = {
            "Dormancy (Jan-Feb)": [('Total_Chill_Day_Units', "Actual Chill Units", 0, 100)],
            "Bloom (Mid-March)": [('Total_Frost_Days', "Cumulative Frost Days", 0, 50), ('Total_Precip_in', "Cumulative Rain (in)", 0.0, 50.0)],
            "Growth & Hull Split (August)": [('Total_Heat_Stress_Days', "Cumulative Heat Stress", 0, 150), ('Avg_VPD_kPa', "Avg. Air Dryness (VPD, kPa)", 0.1, 2.0)],
            "Harvest (October)": [('Total_Precip_in', "FINAL Cumulative Rain (in)", 0.0, 50.0)]
        }
        stage_cols = st.columns(len(stages))

        for i, (stage_name, controls) in enumerate(stages.items()):
            with stage_cols[i]:
                st.subheader(stage_name)
                for key, label, min_val, max_val in controls:
                    slider_key = f"adj_{key}_{i}"
                    default_value = adjusted_inputs[key]
                    if isinstance(default_value, float):
                        adjusted_inputs[key] = st.slider(label, min_val, max_val, float(default_value), key=slider_key)
                    else:
                        adjusted_inputs[key] = st.slider(label, min_val, max_val, int(default_value), key=slider_key)

                stage_results = get_premium_and_scores(artifacts, tuple(adjusted_inputs.items()))
                st.metric(f"Premium after {stage_name.split(' ')[0]}", f"${stage_results['dynamic_premium']:.2f}", f"${stage_results['dynamic_premium'] - last_premium:.2f}", delta_color="inverse")
                last_premium = stage_results['dynamic_premium']

        st.markdown("<hr style='border-color: #233554;'>", unsafe_allow_html=True)
        st.subheader("Final Adjusted Premium Summary")
        final_premium = last_premium
        initial_premium = results['dynamic_premium']
        total_adjustment = final_premium - initial_premium

        summary_col1, summary_col2 = st.columns(2)
        summary_col1.metric("Initial Forecasted Premium", f"${initial_premium:.2f} / acre")
        summary_col2.metric("FINAL Season-Adjusted Premium", f"${final_premium:.2f} / acre", f"Total Adjustment: ${total_adjustment:+.2f}", delta_color="inverse")

        if abs(total_adjustment) < 0.01:
            st.success("**On Track:** Your final premium has remained consistent with the initial forecast, indicating stable conditions.")
        elif total_adjustment < 0:
            st.success(f"**Favorable Season:** Better-than-expected conditions have lowered your final premium by ${-total_adjustment:.2f}/acre!")
        else:
            st.warning(f"**Challenging Season:** Tougher conditions have led to a necessary premium increase of ${total_adjustment:.2f}/acre to cover the elevated risk.")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()