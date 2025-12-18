import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('loan_default_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def probability_to_risk_level(probability, low_threshold=0.3, high_threshold=0.6):
    if probability < low_threshold:
        return 'Low', '🟢'
    elif probability < high_threshold:
        return 'Medium', '🟡'
    else:
        return 'High', '🔴'

def get_risk_color(risk_level):
    colors = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
    return colors.get(risk_level, '#6c757d')

# Main app
def main():
    st.title("🏦 Loan Default Risk Prediction System")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ Model file 'loan_default_model.pkl' not found. Please ensure the model is trained and saved.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Single Prediction", "Batch Prediction", "Model Info"])
    
    if page == "Single Prediction":
        single_prediction_page(model)
    elif page == "Batch Prediction":
        batch_prediction_page(model)
    else:
        model_info_page()

def single_prediction_page(model):
    st.header("Single Loan Application Prediction")
    st.markdown("Enter the applicant's information below to assess default risk.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    
    with col2:
        st.subheader("Employment & Credit")
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=60)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=20, value=3)
        has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
    
    with col3:
        st.subheader("Loan Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=10000, step=1000)
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
        loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=36)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1)
        dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])
    
    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'LoanAmount': [loan_amount],
            'CreditScore': [credit_score],
            'MonthsEmployed': [months_employed],
            'NumCreditLines': [num_credit_lines],
            'InterestRate': [interest_rate],
            'LoanTerm': [loan_term],
            'DTIRatio': [dti_ratio],
            'Education': [education],
            'EmploymentType': [employment_type],
            'MaritalStatus': [marital_status],
            'HasMortgage': [has_mortgage],
            'HasDependents': [has_dependents],
            'LoanPurpose': [loan_purpose],
            'HasCoSigner': [has_cosigner]
        })
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0, 1]
            risk_level, emoji = probability_to_risk_level(probability)
            
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Display results in columns
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("Default Prediction", "Yes" if prediction == 1 else "No")
            
            with res_col2:
                st.metric("Default Probability", f"{probability:.1%}")
            
            with res_col3:
                st.metric("Risk Level", f"{emoji} {risk_level}")
            
            # Gauge chart for probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Risk Score", 'font': {'size': 24}},
                number={'suffix': "%", 'font': {'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': get_risk_color(risk_level)},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 60], 'color': '#fff3cd'},
                        {'range': [60, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.subheader("Recommendation")
            if risk_level == 'Low':
                st.success("✅ **APPROVE** - Low risk applicant. Proceed with standard terms.")
            elif risk_level == 'Medium':
                st.warning("⚠️ **REVIEW REQUIRED** - Moderate risk")
            else:
                st.error("❌ **HIGH RISK** - Recommend rejection or substantial risk mitigation")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

def batch_prediction_page(model):
    st.header("Batch Loan Predictions")
    st.markdown("Upload a CSV file with multiple loan applications for batch processing.")
    
    # Download template
    st.subheader("📥 Download Template")
    template_data = {
        'Age': [35, 28],
        'Income': [50000, 60000],
        'LoanAmount': [10000, 15000],
        'CreditScore': [650, 700],
        'MonthsEmployed': [60, 48],
        'NumCreditLines': [3, 4],
        'InterestRate': [10.0, 8.5],
        'LoanTerm': [36, 48],
        'DTIRatio': [0.3, 0.25],
        'Education': ["Bachelor's", "Master's"],
        'EmploymentType': ["Full-time", "Full-time"],
        'MaritalStatus': ["Single", "Married"],
        'HasMortgage': ["No", "Yes"],
        'HasDependents': ["No", "Yes"],
        'LoanPurpose': ["Auto", "Home"],
        'HasCoSigner': ["No", "Yes"]
    }
    template_df = pd.DataFrame(template_data)
    
    csv = template_df.to_csv(index=False)
    st.download_button(
        label="Download CSV Template",
        data=csv,
        file_name="loan_application_template.csv",
        mime="text/csv"
    )
    
    # File upload
    st.subheader("📤 Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} loan applications")
            
            st.subheader("Preview Data")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🔍 Run Batch Predictions", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Make predictions
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)[:, 1]
                    risk_levels = [probability_to_risk_level(p)[0] for p in probabilities]
                    
                    # Create results dataframe
                    results_df = df.copy()
                    results_df['Default_Prediction'] = predictions
                    results_df['Default_Probability'] = probabilities
                    results_df['Risk_Level'] = risk_levels
                    
                    st.success("✅ Predictions completed!")
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    sum_col1, sum_col2, sum_col3 = st.columns(3)
                    
                    with sum_col1:
                        st.metric("Total Applications", len(results_df))
                        st.metric("Predicted Defaults", int(predictions.sum()))
                    
                    with sum_col2:
                        st.metric("Average Default Probability", f"{probabilities.mean():.1%}")
                        st.metric("High Risk Applications", (pd.Series(risk_levels) == 'High').sum())
                    
                    with sum_col3:
                        st.metric("Medium Risk Applications", (pd.Series(risk_levels) == 'Medium').sum())
                        st.metric("Low Risk Applications", (pd.Series(risk_levels) == 'Low').sum())
                    
                    # Visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Risk level distribution
                        risk_counts = pd.Series(risk_levels).value_counts()
                        fig1 = px.bar(
                            x=risk_counts.index,
                            y=risk_counts.values,
                            color=risk_counts.index,
                            color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
                            labels={'x': 'Risk Level', 'y': 'Count'},
                            title='Risk Level Distribution'
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with viz_col2:
                        # Probability distribution
                        fig2 = px.histogram(
                            probabilities,
                            nbins=30,
                            labels={'value': 'Default Probability', 'count': 'Frequency'},
                            title='Default Probability Distribution'
                        )
                        fig2.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Low/Medium")
                        fig2.add_vline(x=0.6, line_dash="dash", line_color="red", annotation_text="Medium/High")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Display results
                    st.subheader("Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv_results,
                        file_name="loan_predictions_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

def model_info_page():
    st.header("Model Information")
    
    st.subheader("📊 About the Model")
    st.markdown("""
    This loan default prediction system uses a **Logistic Regression** model trained on historical loan data.
    The model predicts the probability of a borrower defaulting on their loan based on various factors.
    
    ### Features Used:
    
    **Numerical Features:**
    - Age
    - Income
    - Loan Amount
    - Credit Score
    - Months Employed
    - Number of Credit Lines
    - Interest Rate
    - Loan Term
    - Debt-to-Income Ratio
    
    **Categorical Features:**
    - Education Level
    - Employment Type
    - Marital Status
    - Has Mortgage
    - Has Dependents
    - Loan Purpose
    - Has Co-Signer
    """)
    
    st.subheader("🎯 Risk Level Classification")
    st.markdown("""
    The model outputs a probability score (0-100%) which is converted to risk levels:
    
    - 🟢 **Low Risk** (< 30%): Approve with standard terms
    - 🟡 **Medium Risk** (30-60%): Review required, consider risk mitigation
    - 🔴 **High Risk** (> 60%): Recommend rejection or substantial safeguards
    """)
    
    st.subheader("⚙️ Model Architecture")
    st.markdown("""
    - **Algorithm**: Logistic Regression with balanced class weights
    - **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
    - **Pipeline**: Scikit-learn Pipeline for streamlined predictions
    """)
    
    st.subheader("📝 How to Use")
    st.markdown("""
    1. **Single Prediction**: Enter individual applicant details for instant risk assessment
    2. **Batch Prediction**: Upload a CSV file with multiple applications for bulk processing
    3. **Download Results**: Export predictions with risk scores and recommendations
    """)

if __name__ == "__main__":
    main()