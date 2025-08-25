import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import io
from datetime import datetime
import base64

# Set page configuration
st.set_page_config(
    page_title="Insurance Claim Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.3rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the XGBoost model from file"""
    try:
        model = xgb.XGBClassifier()
        model.load_model('xgb_claim_model.json')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure 'xgb_claim_model.json' is in the same directory as this app.")
        return None

def preprocess_data(df):
    """
    Preprocess the uploaded data according to the specified transformations
    """
    try:
        df_processed = df.copy()
        
        # Required columns for preprocessing
        required_cols = ['status', 'spartek', 'First_reg', 'SDBEITR5', 'vtr_dau', 'kosten_verw', 'kosten_prov']
        missing_cols = [col for col in required_cols if col not in df_processed.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Apply transformations
        st.write("🔄 Applying data transformations...")
        
        # 1. status_id transformation
        df_processed['status_id'] = pd.factorize(df_processed['status'])[0] + 1
        st.write("✅ Created status_id from status column")
        
        # 2. spartek transformation
        df_processed['spartek'] = pd.factorize(df_processed['spartek'])[0] + 1
        st.write("✅ Transformed spartek column")
        
        # 3. Car_age_indays calculation
        df_processed['First_reg'] = pd.to_datetime(df_processed['First_reg'])
        df_processed['Car_age_indays'] = (pd.Timestamp.today() - df_processed['First_reg']).dt.days
        st.write("✅ Calculated Car_age_indays")
        
        # 4. estimated_total_paid calculation
        df_processed['estimated_total_paid'] = (df_processed['SDBEITR5'] / (5 * 365)) * df_processed['vtr_dau']
        st.write("✅ Calculated estimated_total_paid")
        
        # 5. Select only required features for prediction
        feature_columns = ['vtr_dau', 'kosten_verw', 'kosten_prov', 'spartek', 'status_id', 'Car_age_indays', 'estimated_total_paid']
        
        # Check if all feature columns exist
        missing_features = [col for col in feature_columns if col not in df_processed.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns after preprocessing: {missing_features}")
        
        features_df = df_processed[feature_columns]
        
        st.write("✅ Selected model features")
        
        return df_processed, features_df
        
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

def create_download_link(df, filename, file_format):
    """Create download link for dataframe"""
    if file_format == 'CSV':
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.csv">📥 Download {filename}.csv</a>'
    else:  # Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">📥 Download {filename}.xlsx</a>'
    
    return href

def main():
    # Main header
    st.markdown('<h1 class="main-header">🏠 Insurance Claim Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar information
    with st.sidebar:
        st.header("📋 Model Information")
        st.markdown("""
        **Required Input Columns:**
        - `status` (will be converted to status_id)
        - `spartek` (will be transformed)
        - `First_reg` (for car age calculation)
        - `SDBEITR5` (for payment estimation)
        - `vtr_dau`
        - `kosten_verw`
        - `kosten_prov`
        
        **Model Features:**
        - vtr_dau
        - kosten_verw
        - kosten_prov
        - spartek
        - status_id
        - Car_age_indays
        - estimated_total_paid
        """)
    
    # Step 1: File Upload
    st.markdown('<div class="step-header">Step 1: Upload Your Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload a CSV or Excel file containing your insurance data. Make sure it includes all required columns listed in the sidebar.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with insurance data"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.markdown('<div class="success-box">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
            
            # Display basic info about uploaded data
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Show data preview
            st.markdown('<div class="step-header">Data Preview</div>', unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
            # Step 2: Load Model and Preprocess
            st.markdown('<div class="step-header">Step 2: Model Loading & Data Preprocessing</div>', unsafe_allow_html=True)
            
            if st.button("🚀 Process Data & Make Predictions", type="primary"):
                with st.spinner("Loading model and processing data..."):
                    # Load model
                    model = load_model()
                    
                    if model is not None:
                        try:
                            # Preprocess data
                            df_processed, features_df = preprocess_data(df)
                            
                            # Make predictions
                            st.write("🔮 Making predictions...")
                            predictions = model.predict_proba(features_df)[:, 1]  # Get probability of positive class
                            
                            # Add predictions to original data
                            df_with_predictions = df.copy()
                            df_with_predictions['claim_probability'] = predictions
                            df_with_predictions['claim_prediction'] = (predictions > 0.5).astype(int)
                            
                            st.markdown('<div class="success-box">✅ Predictions completed successfully!</div>', unsafe_allow_html=True)
                            
                            # Step 3: Display Results
                            st.markdown('<div class="step-header">Step 3: Prediction Results</div>', unsafe_allow_html=True)
                            
                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Records", len(df_with_predictions))
                            with col2:
                                st.metric("Predicted Claims", int(df_with_predictions['claim_prediction'].sum()))
                            with col3:
                                st.metric("Claim Rate", f"{df_with_predictions['claim_prediction'].mean():.2%}")
                            with col4:
                                avg_prob = df_with_predictions['claim_probability'].mean()
                                st.metric("Avg. Probability", f"{avg_prob:.3f}")
                            
                            # Results table
                            st.subheader("📊 Detailed Results")
                            
                            # Display results with highlights
                            display_df = df_with_predictions.copy()
                            
                            # Sort by probability for better visualization
                            display_df = display_df.sort_values('claim_probability', ascending=False)
                            
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                column_config={
                                    "claim_probability": st.column_config.ProgressColumn(
                                        "Claim Probability",
                                        help="Probability of insurance claim",
                                        min_value=0,
                                        max_value=1,
                                        format="%.3f"
                                    ),
                                    "claim_prediction": st.column_config.CheckboxColumn(
                                        "Will Claim?",
                                        help="Predicted claim (>50% probability)",
                                    )
                                }
                            )
                            
                            # Step 4: Download Results
                            st.markdown('<div class="step-header">Step 4: Download Results</div>', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📥 Download Options")
                                download_format = st.selectbox("Choose format", ["CSV", "Excel"])
                                filename = st.text_input("Filename", value="insurance_predictions")
                            
                            with col2:
                                st.subheader("📈 Quick Stats")
                                high_risk = (df_with_predictions['claim_probability'] > 0.7).sum()
                                medium_risk = ((df_with_predictions['claim_probability'] > 0.3) & 
                                             (df_with_predictions['claim_probability'] <= 0.7)).sum()
                                low_risk = (df_with_predictions['claim_probability'] <= 0.3).sum()
                                
                                st.write(f"🔴 High Risk (>70%): {high_risk}")
                                st.write(f"🟡 Medium Risk (30-70%): {medium_risk}")
                                st.write(f"🟢 Low Risk (≤30%): {low_risk}")
                            
                            # Create download link
                            download_link = create_download_link(df_with_predictions, filename, download_format)
                            st.markdown(download_link, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.markdown(f'<div class="error-box">❌ Error during processing: {str(e)}</div>', unsafe_allow_html=True)
                            st.error("Please check your data format and ensure all required columns are present.")
            
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Error reading file: {str(e)}</div>', unsafe_allow_html=True)
            st.error("Please ensure your file is a valid CSV or Excel file.")
    
    else:
        st.markdown('<div class="info-box">👆 Please upload a file to get started.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()