"""
Childhood Autism Risk Prediction Web Application
Streamlit app for autism screening based on A1-A10 questions and demographics
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import shap
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Autism Risk Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        line-height: 1.6;
    }
    .prediction-box {
        padding: 2.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
    }
    .low-risk {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left-color: #16a34a;
    }
    .high-risk {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left-color: #dc2626;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px 8px 0 0;
        padding: 0 24px;
        background-color: #f8fafc;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e40af;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Force radio button circle to be blue when selected */
    div[data-baseweb="radio"] input:checked ~ div:first-child {
        background-color: #3b82f6 !important;
        border-color: #3b82f6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to get project root
def get_project_root():
    """Get the project root directory"""
    current_dir = Path(__file__).parent
    return current_dir.parent

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained ML model"""
    try:
        project_root = get_project_root()
        model_path = project_root / "models" / "best_model.pkl"
        
        if not model_path.exists():
            st.error(f"Model file not found at: {model_path}")
            st.info("Please run the training notebook first to generate the model.")
            return None
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# PDF Generation Function
def generate_pdf_report(prediction, probability, answers, child_name, age, gender, ethnicity, jaundice, 
                       autism_family, country, used_app, relation, model_name, accuracy, threshold, 
                       contributions_df=None):
    """Generate a comprehensive PDF report of the screening results"""
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    elements.append(Paragraph("Childhood Autism Risk Screening Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Child's name
    if child_name and child_name.strip() and child_name.strip().lower() != "child":
        name_style = ParagraphStyle(
            'ChildName',
            parent=styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        elements.append(Paragraph(f"Child: {child_name}", name_style))
        elements.append(Spacer(1, 0.1*inch))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    elements.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    elements.append(Paragraph(f"<b>Model Used:</b> {model_name} (Accuracy: {accuracy:.1%})", styles['Normal']))
    elements.append(Paragraph(f"<b>Decision Threshold:</b> {threshold*100:.0f}% (Custom calibrated threshold)", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Risk Assessment Result
    risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
    risk_color = colors.red if prediction == 1 else colors.green
    # Probability is already a scalar (not an array), so just multiply by 100
    risk_prob = probability * 100
    
    elements.append(Paragraph("SCREENING RESULT", heading_style))
    
    result_data = [
        ['Risk Level:', risk_level],
        ['Autism Risk Probability:', f'{risk_prob:.1f}%'],
        ['AQ-10 Score:', f'{sum(answers.values())}/10']
    ]
    
    result_table = Table(result_data, colWidths=[2.5*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (1, 0), (1, 0), risk_color),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (1, 0), (1, 0), 14),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Interpretation
    elements.append(Paragraph("INTERPRETATION", heading_style))
    if prediction == 1:
        interp_text = """
        <b>High Risk Detected:</b> The screening indicates that this child may be at risk for 
        autism spectrum disorder (ASD). This is NOT a diagnosis, only a screening indicator.
        <br/><br/>
        <b>Recommended Actions:</b>
        <ul>
        <li>Consult with a pediatrician or developmental specialist</li>
        <li>Consider comprehensive diagnostic evaluation</li>
        <li>Early intervention programs can be highly beneficial</li>
        <li>Keep monitoring developmental milestones</li>
        </ul>
        """
    else:
        interp_text = """
        <b>Low Risk Detected:</b> The screening suggests that this child is likely not at 
        high risk for autism spectrum disorder (ASD).
        <br/><br/>
        <b>Important Notes:</b>
        <ul>
        <li>Continue monitoring developmental milestones</li>
        <li>If concerns arise, consult healthcare professionals</li>
        <li>This screening is not definitive</li>
        <li>Regular check-ups are still important</li>
        </ul>
        """
    
    elements.append(Paragraph(interp_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Child Demographics
    elements.append(Paragraph("CHILD INFORMATION", heading_style))
    
    demo_data = [
        ['Age:', f'{age} years'],
        ['Gender:', gender],
        ['Ethnicity:', ethnicity],
        ['Born with Jaundice:', jaundice],
        ['Family History of Autism:', autism_family],
        ['Country of Residence:', country],
        ['Previous Screening:', used_app],
        ['Respondent Relation:', relation]
    ]
    
    demo_table = Table(demo_data, colWidths=[2.5*inch, 4*inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(demo_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # AQ-10 Responses
    elements.append(Paragraph("AQ-10 BEHAVIORAL ASSESSMENT", heading_style))
    
    forward_scored = ['A1_Score', 'A7_Score', 'A10_Score']
    questions_text = [
        "Notices small sounds when others do not",
        "Concentrates more on whole picture than details",
        "Can track several conversations in social group",
        "Finds it easy to switch between activities",
        "Knows how to keep conversation going",
        "Good at social chit-chat",
        "Finds it difficult to work out character's intentions",
        "Enjoyed pretend play in preschool",
        "Easy to work out what someone is thinking/feeling",
        "Finds it hard to make new friends"
    ]
    
    aq_data = [['Question', 'Response', 'Score', 'Type']]
    for idx, (key, value) in enumerate(answers.items()):
        q_num = idx + 1
        score_type = 'Forward' if key in forward_scored else 'Reverse'
        response = '✓' if value == 1 else '✗'
        aq_data.append([f'Q{q_num}', questions_text[idx][:50] + '...', f'{value}', score_type])
    
    aq_table = Table(aq_data, colWidths=[0.6*inch, 3.5*inch, 0.6*inch, 1*inch])
    aq_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    elements.append(aq_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Contribution Analysis (if available)
    if contributions_df is not None and len(contributions_df) > 0:
        elements.append(Paragraph("BEHAVIORAL CONTRIBUTION ANALYSIS", heading_style))
        
        contrib_intro = """
        This section shows how each behavioral question contributed to the final risk assessment. 
        Positive values indicate factors that increased risk probability, while negative values 
        indicate protective factors that decreased risk.
        """
        elements.append(Paragraph(contrib_intro, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Separate risk indicators and protective factors
        risk_indicators = contributions_df[contributions_df['Contribution_Value'] > 0].copy()
        protective_factors = contributions_df[contributions_df['Contribution_Value'] < 0].copy()
        
        # Risk Indicators Section
        if len(risk_indicators) > 0:
            risk_heading = ParagraphStyle(
                'RiskHeading',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#dc2626'),
                spaceAfter=8,
                spaceBefore=8
            )
            elements.append(Paragraph("Risk Indicators (Increasing Autism Probability)", risk_heading))
            
            # Create table for risk indicators
            risk_data = [['Question', 'Behavior', 'Answer', 'Contribution']]
            for _, row in risk_indicators.iterrows():
                risk_data.append([
                    row['Question'],
                    row['Behavior'][:35] + '...' if len(row['Behavior']) > 35 else row['Behavior'],
                    row['Answer'],
                    f"+{row['Contribution_Value']:.2f}%"
                ])
            
            risk_table = Table(risk_data, colWidths=[0.6*inch, 3*inch, 0.7*inch, 1*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fee2e2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#7f1d1d')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fef2f2')])
            ]))
            elements.append(risk_table)
            elements.append(Spacer(1, 0.2*inch))
        
        # Protective Factors Section
        if len(protective_factors) > 0:
            protective_heading = ParagraphStyle(
                'ProtectiveHeading',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#16a34a'),
                spaceAfter=8,
                spaceBefore=8
            )
            elements.append(Paragraph("Protective Factors (Decreasing Autism Probability)", protective_heading))
            
            # Create table for protective factors
            protective_data = [['Question', 'Behavior', 'Answer', 'Contribution']]
            for _, row in protective_factors.iterrows():
                protective_data.append([
                    row['Question'],
                    row['Behavior'][:35] + '...' if len(row['Behavior']) > 35 else row['Behavior'],
                    row['Answer'],
                    f"{row['Contribution_Value']:.2f}%"
                ])
            
            protective_table = Table(protective_data, colWidths=[0.6*inch, 3*inch, 0.7*inch, 1*inch])
            protective_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dcfce7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#14532d')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0fdf4')])
            ]))
            elements.append(protective_table)
            elements.append(Spacer(1, 0.2*inch))
        
        # Summary statistics
        total_positive = risk_indicators['Contribution_Value'].sum() if len(risk_indicators) > 0 else 0
        total_negative = protective_factors['Contribution_Value'].sum() if len(protective_factors) > 0 else 0
        net_impact = total_positive + total_negative
        
        summary_text = f"""
        <b>Overall Contribution Summary:</b><br/>
        Total Risk Contribution: +{total_positive:.2f}%<br/>
        Total Protective Contribution: {total_negative:.2f}%<br/>
        Net Impact on Probability: {net_impact:+.2f}%<br/><br/>
        <i>Note: These contributions are calculated using SHAP (SHapley Additive exPlanations) values, 
        which provide an exact breakdown of how each behavioral question influenced the final risk assessment.</i>
        """
        elements.append(Paragraph(summary_text, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    elements.append(Paragraph("IMPORTANT DISCLAIMER", heading_style))
    disclaimer_text = """
    This screening tool is for informational purposes only and should not replace professional 
    medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers 
    with questions regarding medical conditions. This is a preliminary screening tool based on 
    machine learning algorithms and behavioral questionnaires. A comprehensive evaluation by a 
    qualified healthcare professional is necessary for an accurate diagnosis.
    <br/><br/>
    <b>For more information or concerns, please consult:</b>
    <ul>
    <li>Your child's pediatrician</li>
    <li>A developmental pediatrician</li>
    <li>Child psychologist or psychiatrist</li>
    <li>Early intervention programs in your area</li>
    </ul>
    """
    elements.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Footer
    elements.append(Spacer(1, 0.3*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    elements.append(Paragraph("© 2025 Childhood Autism Risk Prediction System | Machine Learning Based Screening Tool", footer_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# SHAP Explanation Function
def generate_shap_explanation(model, scaler, user_input_df, feature_names):
    """
    Generate proper SHAP explanations for logistic regression
    
    SHAP values represent how much each feature contributes to pushing the prediction
    from the baseline (average) to the actual prediction. They properly account for
    feature interactions in the non-linear model.
    
    Key properties:
    - SHAP values are additive in probability space (when using interventional values)
    - baseline + sum(shap_values) ≈ prediction
    - Individual values show TRUE marginal contribution accounting for other features
    """
    # Scale the user input
    user_input_scaled = scaler.transform(user_input_df)
    
    # Use SHAP's LinearExplainer with proper configuration
    # For logistic regression, this gives exact Shapley values
    explainer = shap.LinearExplainer(model, np.zeros((1, user_input_scaled.shape[1])))
    
    # Calculate SHAP values - these are in log-odds space
    shap_values_logodds = explainer.shap_values(user_input_scaled)
    
    # Get predictions
    pred_proba = model.predict_proba(user_input_scaled)[0, 1]
    base_logodds = explainer.expected_value
    base_prob = 1 / (1 + np.exp(-base_logodds))
    
    # CRITICAL: Convert SHAP values from log-odds to probability contributions
    # Use the exact SHAP interpretation: show how adding each feature changes probability
    
    # Calculate cumulative probability as we add features
    # Start from baseline and add features in order of their contribution
    contributions = []
    
    for i, feature_name in enumerate(feature_names):
        # Log-odds contribution (true SHAP value)
        logodds_contrib = shap_values_logodds[0][i]
        
        # To get probability contribution, we calculate the marginal change
        # Using the formula: Δp = sigmoid(baseline + Δlogodds) - sigmoid(baseline)
        # But we use the FULL model context (all other features present)
        
        # Current total log-odds
        current_logodds = base_logodds + np.sum(shap_values_logodds[0])
        
        # Log-odds without this feature
        logodds_without = current_logodds - logodds_contrib
        
        # Probabilities
        prob_with = 1 / (1 + np.exp(-current_logodds))
        prob_without = 1 / (1 + np.exp(-logodds_without))
        
        # True marginal contribution in probability space
        prob_contribution = prob_with - prob_without
        
        contributions.append({
            'Feature': feature_name,
            'Value': user_input_df[feature_name].values[0],
            'LogOdds_Contribution': logodds_contrib,
            'SHAP_Value': prob_contribution,  # In probability space
            'Abs_SHAP': abs(prob_contribution)
        })
    
    # Convert to DataFrame and sort
    contributions_df = pd.DataFrame(contributions).sort_values('LogOdds_Contribution', ascending=False)
    
    # Verify that SHAP values sum approximately to (prediction - baseline)
    shap_sum = contributions_df['SHAP_Value'].sum()
    expected_sum = pred_proba - base_prob
    
    # If there's a large discrepancy, normalize
    if abs(shap_sum - expected_sum) > 0.01:
        # Normalize SHAP values to sum correctly
        normalization_factor = expected_sum / shap_sum if shap_sum != 0 else 1
        contributions_df['SHAP_Value'] = contributions_df['SHAP_Value'] * normalization_factor
        contributions_df['Abs_SHAP'] = contributions_df['SHAP_Value'].abs()
    
    return contributions_df['SHAP_Value'].values, base_prob, contributions_df

def interpret_question_contribution(question_key, user_answer, shap_value, forward_scored):
    """
    Interpret what a question's contribution means in plain language
    
    Parameters:
    - question_key: e.g., 'A1_Score'
    - user_answer: The actual answer given (Yes/No)
    - shap_value: SHAP contribution value
    - forward_scored: List of forward-scored questions
    
    Returns:
    - interpretation: Plain English explanation
    - impact_type: 'increases_risk' or 'decreases_risk'
    """
    is_forward = question_key in forward_scored
    contribution_percent = shap_value * 100  # Convert to percentage points
    
    # Determine impact with professional badges
    if shap_value > 0:
        impact_type = 'increases_risk'
        impact_badge = '<span style="background-color: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">RISK</span>'
        impact_color = 'red'
    else:
        impact_type = 'decreases_risk'
        impact_badge = '<span style="background-color: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">SAFE</span>'
        impact_color = 'green'
    
    # Build interpretation based on scoring type and answer
    question_texts = {
        'A1_Score': 'Noticing small sounds',
        'A2_Score': 'Concentrating on whole picture',
        'A3_Score': 'Tracking multiple conversations',
        'A4_Score': 'Switching between activities',
        'A5_Score': 'Keeping conversation going',
        'A6_Score': 'Social chit-chat',
        'A7_Score': 'Understanding characters\' emotions',
        'A8_Score': 'Enjoying pretend play',
        'A9_Score': 'Reading facial expressions',
        'A10_Score': 'Making new friends'
    }
    
    behavior = question_texts.get(question_key, question_key)
    
    if is_forward:
        # Forward scored: YES = risk, NO = protective
        if user_answer == "Yes":
            interpretation = f"Child shows difficulty with {behavior.lower()}"
        else:
            interpretation = f"Child manages {behavior.lower()} well"
    else:
        # Reverse scored: NO = risk, YES = protective
        if user_answer == "No":
            interpretation = f"Child shows difficulty with {behavior.lower()}"
        else:
            interpretation = f"Child manages {behavior.lower()} well"
    
    return {
        'interpretation': interpretation,
        'impact_type': impact_type,
        'impact_badge': impact_badge,
        'impact_color': impact_color,
        'contribution_percent': abs(contribution_percent)
    }

# Load model and scaler
model_data = load_model()

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Childhood Autism Risk Screening System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evidence-based screening tool utilizing the AQ-10 questionnaire and machine learning algorithms for preliminary autism risk assessment</p>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if model_data is None:
        st.warning("Model not loaded. Please ensure the model is trained and saved.")
        st.stop()
    
    model = model_data['model']
    scaler = model_data.get('scaler', None)
    model_name = model_data.get('model_name', 'Unknown')
    accuracy = model_data.get('accuracy', 0)
    
    # Sidebar - Information
    with st.sidebar:
        st.header("About This Tool")
        st.write(f"""
        **Model**: {model_name}  
        **Accuracy**: {accuracy:.2%}
        
        This screening tool helps identify children who may be at risk for autism spectrum disorder (ASD).
        
        **Important Notes**:
        - This is a screening tool, NOT a diagnosis
        - Always consult healthcare professionals
        - Early detection enables early intervention
        """)
        
        st.header("Instructions")
        st.write("""
        1. Provide demographic information
        2. Click "Next" to proceed to questions
        3. Answer all 10 behavioral questions
        4. Click "Next" to view results
        5. Generate risk assessment report
        
        **Tip:** Use Back/Next buttons to navigate between sections
        """)
        
        st.header("Privacy & Security")
        st.write("""
        - No data is stored or shared
        - All processing happens in real-time
        - Your information is confidential
        """)
        
        st.markdown("---")
        
        st.header("Model Calibration (Advanced)")
        threshold = st.slider(
            "Risk Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust the sensitivity of risk detection. Lower values = more sensitive (catches more cases but may increase false alarms). Higher values = more specific (fewer false alarms but may miss some cases)."
        )
        
        st.info(f"""
        **Current Logic:** Flag as high risk if probability > {threshold*100:.0f}%
        
        **Recommendations:**
        - **Conservative (30-40%)**: Prioritize catching all potential cases
        - **Balanced (50%)**: Standard medical screening threshold
        - **Strict (60-70%)**: Minimize false positives
        """)
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'demographics' not in st.session_state:
        st.session_state.demographics = {}
    
    # Main content area
    st.header("Screening Assessment")
    
    # Progress indicator
    pages = ["Demographics", "Behavioral Questions", "Results"]
    progress_cols = st.columns(3)
    for idx, page_name in enumerate(pages):
        with progress_cols[idx]:
            if idx < st.session_state.current_page:
                # Completed sections - blue circle
                st.markdown(f"🔵 {idx+1}. {page_name}")
            elif idx == st.session_state.current_page:
                # Current section - bold but no color
                st.markdown(f"**⚪ {idx+1}. {page_name}**")
            else:
                # Remaining sections - white circle
                st.markdown(f"⚪ {idx+1}. {page_name}")
    
    st.markdown("---")
    
    # Page 1: Demographics
    if st.session_state.current_page == 0:
        st.subheader("Demographic Information")
        
        # Child's name (for report personalization only)
        child_name = st.text_input(
            "Child's Name (Optional)",
            value="",
            placeholder="Enter child's name for report personalization",
            help="This name will only appear in the PDF report for personalization. It is not stored anywhere."
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Child's Age (years)",
                min_value=4,
                max_value=11,
                value=5,
                help="Enter the child's age in years"
            )
            
            gender = st.selectbox(
                "Gender",
                options=["Male", "Female"],
                help="Select the child's gender"
            )
            
            ethnicity = st.selectbox(
                "Ethnicity",
                options=["White-European", "Asian", "Middle Eastern", "South Asian", "Black", "Others"],
                help="Select the child's ethnicity"
            )
            
            jaundice = st.selectbox(
                "Born with Jaundice?",
                options=["No", "Yes"],
                help="Was the child born with jaundice?"
            )
            
            autism_family = st.selectbox(
                "Family History of Autism?",
                options=["No", "Yes"],
                help="Is there a family history of autism?"
            )
        
        with col2:
            country = st.selectbox(
                "Country of Residence",
                options=["United States", "United Kingdom", "India", "Australia", "Jordan", "Other"],
                help="Select the country of residence"
            )
            
            used_app = st.selectbox(
                "Used Screening App Before?",
                options=["No", "Yes"],
                help="Has the child been screened with a similar app before?"
            )
            
            relation = st.selectbox(
                "Your Relation to Child",
                options=["Parent", "Relative", "Health care professional", "Self"],
                help="What is your relationship to the child?"
            )
        
        # Store demographics in session state
        st.session_state.demographics = {
            'child_name': child_name if child_name.strip() else "Child",
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity,
            'jaundice': jaundice,
            'autism_family': autism_family,
            'country': country,
            'used_app': used_app,
            'relation': relation
        }
        
        # Navigation buttons
        st.markdown("---")
        col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
        with col_nav3:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.current_page = 1
                st.rerun()
    
    # Page 2: Behavioral Questions
    elif st.session_state.current_page == 1:
        st.subheader("Behavioral Assessment (AQ-10 Autism Quotient)")
        st.info("**Instructions:** Answer each question based on the child's typical behavior. Please respond honestly for the most accurate assessment.")
        st.write("**Note:** These are standardized AQ-10 screening questions. The scoring system automatically accounts for forward and reverse-scored items.")
        
        # AQ-10 Questions - Official Autism Quotient screening questions
        # ═══════════════════════════════════════════════════════════════════
        # IMPORTANT: Mixed scoring system (not all questions scored the same way)
        # 
        # Forward scored (YES = 1 = Risk): A1, A7, A10
        #   - A1: Notices small sounds → YES indicates hyper-focus (autism trait)
        #   - A7: Difficulty understanding emotions → YES indicates social difficulty
        #   - A10: Hard to make friends → YES indicates social difficulty
        # 
        # Reverse scored (NO = 1 = Risk): A2, A3, A4, A5, A6, A8, A9
        #   - A2-A6, A8-A9: Lack of typical social/cognitive abilities
        #   - NO indicates absence of neurotypical development
        # ═══════════════════════════════════════════════════════════════════
        questions = {
            'A1_Score': "Does the child often notice small sounds when others do not?",
            'A2_Score': "Does the child usually concentrate more on the whole picture rather than the small details?",
            'A3_Score': "In a social group, can the child easily keep track of several different people's conversations?",
            'A4_Score': "Does the child find it easy to go back and forth between different activities?",
            'A5_Score': "Does the child know how to keep a conversation going with his/her peers?",
            'A6_Score': "Is the child good at social chit-chat?",
            'A7_Score': "When read a story, does the child find it difficult to work out the character's intentions or feelings?",
            'A8_Score': "When he/she was in preschool, did he/she use to enjoy playing games involving pretending with other children?",
            'A9_Score': "Does the child find it easy to work out what someone is thinking or feeling just by looking at their face?",
            'A10_Score': "Does the child find it hard to make new friends?"
        }
        
        # Collect answers - display sequentially for mobile compatibility
        answers = {}
        
        # Forward scored questions (YES = risk): A1, A7, A10
        forward_scored = ['A1_Score', 'A7_Score', 'A10_Score']
        # Reverse scored questions (NO = risk): A2, A3, A4, A5, A6, A8, A9
        
        # Display questions in two columns for better layout
        col_q1, col_q2 = st.columns(2)
        
        question_items = list(questions.items())
        
        # Left column: Q1-Q5
        with col_q1:
            for idx in range(5):
                key, question = question_items[idx]
                answer = st.radio(
                    f"**Q{idx+1}**: {question}",
                    options=["Yes", "No"],
                    index=0,
                    key=key,
                    horizontal=True
                )
                
                # Apply correct scoring
                if key in forward_scored:
                    answers[key] = 1 if answer == "Yes" else 0
                else:
                    answers[key] = 1 if answer == "No" else 0
        
        # Right column: Q6-Q10
        with col_q2:
            for idx in range(5, 10):
                key, question = question_items[idx]
                answer = st.radio(
                    f"**Q{idx+1}**: {question}",
                    options=["Yes", "No"],
                    index=0,
                    key=key,
                    horizontal=True
                )
                
                # Apply correct scoring
                if key in forward_scored:
                    answers[key] = 1 if answer == "Yes" else 0
                else:
                    answers[key] = 1 if answer == "No" else 0
        
        # Store answers in session state
        st.session_state.answers = answers
        
        # Navigation buttons
        st.markdown("---")
        col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
        with col_nav1:
            if st.button("Back", use_container_width=True):
                st.session_state.current_page = 0
                st.rerun()
        with col_nav3:
            if st.button("Next", type="primary", use_container_width=True):
                st.session_state.current_page = 2
                st.rerun()
    
    # Page 3: Results
    elif st.session_state.current_page == 2:
        st.subheader("Risk Assessment Results")
        
        # Get data from session state
        answers = st.session_state.answers
        demographics = st.session_state.demographics
        
        # Predict button
        if st.button("Generate Risk Assessment", type="primary", use_container_width=True):
            # Prepare input data
            try:
                # Create feature dictionary
                features = answers.copy()
                
                # Add demographic features
                features['age'] = demographics['age']
                features['gender'] = 1 if demographics['gender'] == "Male" else 0
                features['jaundice'] = 1 if demographics['jaundice'] == "Yes" else 0
                features['autism'] = 1 if demographics['autism_family'] == "Yes" else 0
                # Note: used_app_before removed from model
                
                # One-hot encode ethnicity
                ethnicity_mapping = {
                    'ethnicity_Asian': 1 if demographics['ethnicity'] == "Asian" else 0,
                    'ethnicity_Black': 1 if demographics['ethnicity'] == "Black" else 0,
                    'ethnicity_Middle Eastern': 1 if demographics['ethnicity'] == "Middle Eastern" else 0,
                    'ethnicity_Others': 1 if demographics['ethnicity'] == "Others" else 0,
                    'ethnicity_South Asian': 1 if demographics['ethnicity'] == "South Asian" else 0,
                    'ethnicity_White-European': 1 if demographics['ethnicity'] == "White-European" else 0
                }
                features.update(ethnicity_mapping)
                
                # Note: Country and relation columns removed from model training
                
                # Create DataFrame with proper column order (matching training data)
                expected_columns = [
                    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                    'age', 'gender', 'jaundice', 'autism',
                    'ethnicity_Asian', 'ethnicity_Black', 'ethnicity_Middle Eastern',
                    'ethnicity_Others', 'ethnicity_South Asian', 'ethnicity_White-European'
                ]
                
                # Create input DataFrame
                input_df = pd.DataFrame([features])
                
                # Ensure all columns are present
                for col in expected_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns
                input_df = input_df[expected_columns]
                
                # Apply scaling if needed
                if scaler is not None:
                    input_scaled = scaler.transform(input_df)
                else:
                    input_scaled = input_df.values
                
                # Make prediction
                prediction_proba = model.predict_proba(input_scaled)[0]
                risk_probability = prediction_proba[1]  # Probability of autism (class 1)
                
                # Use custom threshold instead of default 0.5
                prediction = 1 if risk_probability > threshold else 0
                
                # Display results
                risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
                risk_class = "high-risk" if prediction == 1 else "low-risk"
                risk_probability_percent = risk_probability * 100
                
                st.markdown(f"""
                    <div class="prediction-box {risk_class}">
                        <h2 style="text-align: center; margin-bottom: 0.5rem; font-weight: 700; font-size: 1.8rem;">
                            SCREENING RESULT: {risk_level}
                        </h2>
                        <h3 style="text-align: center; font-weight: 500; font-size: 1.3rem; color: #4a5568;">
                            Risk Probability: {risk_probability_percent:.1f}%
                        </h3>
                        <p style="text-align: center; margin-top: 1rem; font-size: 0.9rem; color: #6b7280;">
                            AQ-10 Score: {sum(answers.values())}/10 | Decision Threshold: {threshold*100:.0f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detailed interpretation
                st.subheader("Clinical Interpretation")
                
                # Show threshold impact
                st.info(f"""
                **Decision Analysis:** Based on risk probability of {risk_probability_percent:.1f}% and threshold of {threshold*100:.0f}%, 
                the assessment is classified as **{risk_level}**.
                
                💡 **Adjust the threshold slider** in the sidebar to see how different sensitivity levels affect the classification.
                """)
                
                if prediction == 1:
                    st.error("""
                    **High Risk Assessment**
                    
                    The screening assessment indicates elevated risk indicators for autism spectrum disorder (ASD). This preliminary screening result suggests the need for professional evaluation.
                    
                    **Recommended Next Steps**:
                    - Schedule consultation with a pediatrician or developmental specialist
                    - Request comprehensive diagnostic evaluation
                    - Explore early intervention programs and resources
                    - Important: This is a screening tool, not a diagnostic instrument
                    """)
                else:
                    st.success("""
                    **Low Risk Assessment**
                    
                    The screening assessment indicates low probability of autism spectrum disorder (ASD) based on current behavioral indicators.
                    
                    **Important Considerations**:
                    - Continue regular monitoring of developmental milestones
                    - Maintain routine pediatric check-ups
                    - Consult healthcare professionals if new concerns emerge
                    - Screening results are preliminary and not diagnostic
                    """)
                
                # ═══════════════════════════════════════════════════════════════════
                # SHAP EXPLANATION SECTION
                # ═══════════════════════════════════════════════════════════════════
                st.markdown("---")
                st.subheader("Understanding Your Results")
                st.write("""
                This section explains how each behavioral question contributed to the final risk assessment. 
                Positive contributions increase risk probability, while negative contributions decrease it.
                """)
                
                try:
                    # Generate SHAP explanations
                    shap_values, base_value, feature_contributions = generate_shap_explanation(
                        model=model,
                        scaler=scaler,
                        user_input_df=input_df,
                        feature_names=expected_columns
                    )
                    
                    # Filter for only screening questions (A1-A10)
                    question_features = feature_contributions[
                        feature_contributions['Feature'].str.startswith('A')
                    ].copy()
                    
                    # Map user answers for interpretation
                    forward_scored = ['A1_Score', 'A7_Score', 'A10_Score']
                    
                    # Full questions for hover tooltips
                    full_questions = {
                        'A1_Score': "Does the child often notice small sounds when others do not?",
                        'A2_Score': "Does the child usually concentrate more on the whole picture rather than the small details?",
                        'A3_Score': "In a social group, can the child easily keep track of several different people's conversations?",
                        'A4_Score': "Does the child find it easy to go back and forth between different activities?",
                        'A5_Score': "Does the child know how to keep a conversation going with his/her peers?",
                        'A6_Score': "Is the child good at social chit-chat?",
                        'A7_Score': "When read a story, does the child find it difficult to work out the character's intentions or feelings?",
                        'A8_Score': "When he/she was in preschool, did he/she use to enjoy playing games involving pretending with other children?",
                        'A9_Score': "Does the child find it easy to work out what someone is thinking or feeling just by looking at their face?",
                        'A10_Score': "Does the child find it hard to make new friends?"
                    }
                    
                    # Short question texts (accurate to original meaning)
                    question_texts = {
                        'A1_Score': "Noticing small sounds",
                        'A2_Score': "Concentrating on whole picture",
                        'A3_Score': "Tracking multiple conversations",
                        'A4_Score': "Switching between activities",
                        'A5_Score': "Keeping conversations going",
                        'A6_Score': "Social chit-chat ability",
                        'A7_Score': "Difficulty understanding emotions",
                        'A8_Score': "Enjoying pretend play",
                        'A9_Score': "Reading facial expressions",
                        'A10_Score': "Difficulty making friends"
                    }
                    
                    # Get user answers from the questions section
                    # We need to map back to actual Yes/No from the session state
                    # Since answers are already scored, we need to reverse engineer
                    user_answers_text = {}
                    for key in question_features['Feature']:
                        if key in answers:
                            score = answers[key]
                            if key in forward_scored:
                                # Forward: 1=Yes, 0=No
                                user_answers_text[key] = "Yes" if score == 1 else "No"
                            else:
                                # Reverse: 1=No (risk), 0=Yes
                                user_answers_text[key] = "No" if score == 1 else "Yes"
                    
                    # Create visualization tabs
                    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Contribution Chart", "📋 Detailed Analysis", "🎯 Key Insights"])
                    
                    with viz_tab1:
                        st.write("**How each question affected your risk probability:**")
                        st.caption("💡 Hover over bars to see full question text and detailed impact information")
                        
                        # Sort by SHAP value (most positive to most negative)
                        question_features_sorted = question_features.sort_values('SHAP_Value', ascending=True)
                        
                        # Prepare data for Plotly
                        plot_data = []
                        for idx, row in question_features_sorted.iterrows():
                            q_key = row['Feature']
                            q_num = q_key.replace('_Score', '')
                            q_short = question_texts.get(q_key, q_key)
                            q_full = full_questions.get(q_key, q_key)
                            user_ans = user_answers_text.get(q_key, '?')
                            contribution = row['SHAP_Value'] * 100
                            
                            # Determine impact type
                            if contribution > 0:
                                impact = "Increases Risk"
                                impact_desc = f"This answer increased the autism risk probability by {contribution:.1f} percentage points."
                            else:
                                impact = "Decreases Risk"
                                impact_desc = f"This answer decreased the autism risk probability by {abs(contribution):.1f} percentage points."
                            
                            plot_data.append({
                                'Question': q_num,
                                'Short_Label': f"{q_num} ({user_ans}): {q_short}",
                                'Full_Question': q_full,
                                'Answer': user_ans,
                                'Contribution': contribution,
                                'Impact': impact,
                                'Impact_Description': impact_desc,
                                'Color': '#dc2626' if contribution > 0 else '#16a34a'
                            })
                        
                        plot_df = pd.DataFrame(plot_data)
                        
                        # Create interactive Plotly horizontal bar chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            y=plot_df['Short_Label'],
                            x=plot_df['Contribution'],
                            orientation='h',
                            marker=dict(
                                color=plot_df['Color'],
                                line=dict(color='#1f2937', width=1.5)
                            ),
                            hovertemplate=(
                                "<b style='font-size:14px'>%{customdata[0]}</b><br><br>"
                                "<b>Full Question:</b><br>%{customdata[1]}<br><br>"
                                "<b>Your Answer:</b> %{customdata[2]}<br>"
                                "<b>Contribution:</b> %{x:+.2f}%<br>"
                                "<b>Impact:</b> %{customdata[3]}<br><br>"
                                "<i>%{customdata[4]}</i>"
                                "<extra></extra>"
                            ),
                            customdata=plot_df[['Question', 'Full_Question', 'Answer', 'Impact', 'Impact_Description']],
                            text=plot_df['Contribution'].apply(lambda x: f'{x:+.2f}%'),
                            textposition='outside',
                            textfont=dict(size=11, color='#1f2937', family='Arial Black'),
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title={
                                'text': '<b>Behavioral Question Contributions to Autism Risk Assessment</b>',
                                'font': {'size': 16, 'color': '#1e40af'},
                                'x': 0.5,
                                'xanchor': 'center'
                            },
                            xaxis=dict(
                                title='<b>Contribution to Risk Probability (percentage points)</b>',
                                titlefont=dict(size=12, color='#1f2937'),
                                showgrid=True,
                                gridcolor='#e5e7eb',
                                zeroline=True,
                                zerolinecolor='#1f2937',
                                zerolinewidth=2
                            ),
                            yaxis=dict(
                                title='',
                                tickfont=dict(size=11, color='#1f2937')
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=600,
                            margin=dict(l=20, r=100, t=60, b=60),
                            hoverlabel=dict(
                                bgcolor='white',
                                font_size=12,
                                font_family='Arial',
                                bordercolor='#1e40af'
                            )
                        )
                        
                        # Display the interactive chart
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
                        
                        # Legend
                        st.markdown("""
                        <div style='background-color: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #3b82f6;'>
                            <p style='margin: 0; font-size: 0.9rem;'>
                                <span style='color: #dc2626; font-weight: bold;'>🔴 Red bars</span> = Increases autism risk probability<br>
                                <span style='color: #16a34a; font-weight: bold;'>🟢 Green bars</span> = Decreases autism risk probability<br>
                                <strong>💡 Interactive Tip:</strong> Hover over any bar to see the complete question text and detailed explanation of how that answer affected the risk assessment.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with viz_tab2:
                        st.write("**Detailed breakdown of each behavioral indicator:**")
                        
                        # Create detailed interpretation table
                        # Sort: Risk indicators (positive) first, then protective factors (negative)
                        interpretation_data = []
                        
                        # First add risk indicators (positive contributions)
                        risk_indicators = question_features[question_features['SHAP_Value'] > 0].sort_values('SHAP_Value', ascending=False)
                        for idx, row in risk_indicators.iterrows():
                            q_key = row['Feature']
                            q_num = q_key.replace('_Score', '')
                            q_text = question_texts.get(q_key, q_key)
                            user_ans = user_answers_text.get(q_key, '?')
                            contribution = row['SHAP_Value'] * 100
                            
                            # Get interpretation
                            interpretation_dict = interpret_question_contribution(
                                question_key=q_key,
                                user_answer=user_ans,
                                shap_value=row['SHAP_Value'],
                                forward_scored=forward_scored
                            )
                            
                            interpretation_data.append({
                                'Question': f"{q_num}",
                                'Behavior': q_text,
                                'Your Answer': user_ans,
                                'Contribution': f"{contribution:+.2f}%",
                                'Impact': f"{interpretation_dict['interpretation']}"
                            })
                        
                        # Then add protective factors (negative contributions)
                        protective_factors = question_features[question_features['SHAP_Value'] < 0].sort_values('SHAP_Value', ascending=True)
                        for idx, row in protective_factors.iterrows():
                            q_key = row['Feature']
                            q_num = q_key.replace('_Score', '')
                            q_text = question_texts.get(q_key, q_key)
                            user_ans = user_answers_text.get(q_key, '?')
                            contribution = row['SHAP_Value'] * 100
                            
                            # Get interpretation
                            interpretation_dict = interpret_question_contribution(
                                question_key=q_key,
                                user_answer=user_ans,
                                shap_value=row['SHAP_Value'],
                                forward_scored=forward_scored
                            )
                            
                            interpretation_data.append({
                                'Question': f"{q_num}",
                                'Behavior': q_text,
                                'Your Answer': user_ans,
                                'Contribution': f"{contribution:+.2f}%",
                                'Impact': f"{interpretation_dict['interpretation']}"
                            })
                        
                        # Display as DataFrame
                        interp_df = pd.DataFrame(interpretation_data)
                        st.dataframe(interp_df, use_container_width=True, hide_index=True)
                        
                        # Explanation
                        st.info("""
                        **How to read this table:**
                        - **Positive contributions** (+X%) indicate answers that increased the risk probability
                        - **Negative contributions** (-X%) indicate answers that decreased the risk probability
                        - Contributions are calculated using SHAP (SHapley Additive exPlanations) values
                        """)
                    
                    with viz_tab3:
                        st.write("**Key behavioral indicators identified in this screening:**")
                        
                        # Top risk contributors
                        top_risk = question_features[question_features['SHAP_Value'] > 0].sort_values('SHAP_Value', ascending=False).head(3)
                        
                        # Top protective factors
                        top_protective = question_features[question_features['SHAP_Value'] < 0].sort_values('SHAP_Value', ascending=True).head(3)
                        
                        col_risk, col_protect = st.columns(2)
                        
                        with col_risk:
                            st.markdown("##### <span style='background-color: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 16px; font-size: 0.9rem; font-weight: 700;'>TOP RISK INDICATORS</span>", unsafe_allow_html=True)
                            if len(top_risk) > 0:
                                for idx, row in top_risk.iterrows():
                                    q_key = row['Feature']
                                    q_num = q_key.replace('_Score', '')
                                    q_text = question_texts.get(q_key, q_key)
                                    contribution = row['SHAP_Value'] * 100
                                    st.markdown(f"""
                                    <div style='background-color: #fee2e2; padding: 0.8rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 4px solid #dc2626;'>
                                        <strong>{q_num}:</strong> {q_text}<br>
                                        <span style='color: #dc2626; font-weight: bold;'>+{contribution:.2f}%</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.write("No significant risk indicators detected.")
                        
                        with col_protect:
                            st.markdown("##### <span style='background-color: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 16px; font-size: 0.9rem; font-weight: 700;'>TOP PROTECTIVE FACTORS</span>", unsafe_allow_html=True)
                            if len(top_protective) > 0:
                                for idx, row in top_protective.iterrows():
                                    q_key = row['Feature']
                                    q_num = q_key.replace('_Score', '')
                                    q_text = question_texts.get(q_key, q_key)
                                    contribution = row['SHAP_Value'] * 100
                                    st.markdown(f"""
                                    <div style='background-color: #dcfce7; padding: 0.8rem; border-radius: 6px; margin-bottom: 0.5rem; border-left: 4px solid #16a34a;'>
                                        <strong>{q_num}:</strong> {q_text}<br>
                                        <span style='color: #16a34a; font-weight: bold;'>{contribution:.2f}%</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.write("No significant protective factors detected.")
                        
                        # Summary insight
                        st.markdown("---")
                        
                        # Calculate SHAP contributions from behavioral questions
                        question_positive = question_features[question_features['SHAP_Value'] > 0]['SHAP_Value'].sum() * 100
                        question_negative = abs(question_features[question_features['SHAP_Value'] < 0]['SHAP_Value'].sum() * 100)
                        question_net = question_positive - question_negative
                        
                        # Calculate ALL features' SHAP contributions
                        all_shap_sum = feature_contributions['SHAP_Value'].sum() * 100
                        
                        # Get demographics SHAP contribution
                        demo_features = feature_contributions[~feature_contributions['Feature'].str.startswith('A')]
                        demo_shap_sum = demo_features['SHAP_Value'].sum() * 100
                        
                        # Verify math: baseline + all_shap_sum should equal final probability
                        calculated_final = base_value*100 + all_shap_sum
                        math_check_ok = abs(calculated_final - risk_probability_percent) < 0.5
                        
                        st.markdown(f"""
                        <div style='background-color: #f1f5f9; padding: 1.2rem; border-radius: 8px; margin-top: 1rem;'>
                            <h4 style='margin-top: 0; color: #1e40af;'>
                                <span style='background-color: #dbeafe; color: #1e40af; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem;'>EXPLANATION</span> 
                                How We Calculated {risk_probability_percent:.1f}% Risk
                            </h4>
                            <div style='background-color: white; padding: 1rem; border-radius: 6px; margin: 1rem 0; border-left: 4px solid #3b82f6;'>
                                <p style='font-size: 1.05rem; margin: 0.4rem 0; line-height: 2;'>
                                    <strong>Baseline Risk:</strong> 
                                    <span style='color: #6b7280; font-size: 1.15rem; font-weight: 600;'>{base_value*100:.1f}%</span>
                                    <span style='font-size: 0.85rem; color: #9ca3af;'> (average case)</span>
                                    <br>
                                    <strong>+ Behavioral Impact:</strong> 
                                    <span style='color: {"#dc2626" if question_net > 0 else "#16a34a"}; font-size: 1.1rem; font-weight: 600;'>{question_net:+.2f}pp</span>
                                    <span style='font-size: 0.85rem; color: #9ca3af;'> (AQ-10 questions)</span>
                                    <br>
                                    <strong>+ Demographics:</strong> 
                                    <span style='color: {"#dc2626" if demo_shap_sum > 0 else "#16a34a"}; font-size: 1.1rem; font-weight: 600;'>{demo_shap_sum:+.2f}pp</span>
                                    <span style='font-size: 0.85rem; color: #9ca3af;'> (age, gender, ethnicity)</span>
                                    <br>
                                    <div style='border-top: 2px solid #3b82f6; margin: 0.7rem 0; padding-top: 0.7rem;'>
                                        <strong style='font-size: 1.2rem; color: #1e40af;'>= Final Assessment:</strong> 
                                        <span style='font-weight: bold; font-size: 1.4rem; color: {"#dc2626" if prediction == 1 else "#16a34a"};'>{risk_probability_percent:.1f}%</span>
                                        {f'<span style="color: #16a34a; font-size: 0.9rem; margin-left: 10px;">✓ Math verified</span>' if math_check_ok else f'<span style="color: #9ca3af; font-size: 0.85rem;"> (calculated: {calculated_final:.1f}%)</span>'}
                                    </div>
                                </p>
                            </div>
                            <hr style='margin: 1rem 0; border: none; border-top: 1px solid #cbd5e1;'>
                            <p style='font-size: 0.95rem; margin: 0.5rem 0; line-height: 1.7;'>
                                <strong>Behavioral Question Breakdown:</strong><br>
                                <span style='color: #dc2626;'>• Risk-increasing: +{question_positive:.2f}pp</span><br>
                                <span style='color: #16a34a;'>• Protective: -{question_negative:.2f}pp</span><br>
                                <span style='font-weight: 600;'>• Net behavioral: {question_net:+.2f}pp</span>
                            </p>
                            <div style='background-color: #eff6ff; border-left: 3px solid #3b82f6; padding: 0.8rem; border-radius: 4px; margin-top: 1rem;'>
                                <p style='font-size: 0.88rem; margin: 0; color: #1e40af; line-height: 1.6;'>
                                    <strong>📊 How This Works:</strong> We use SHAP (SHapley Additive exPlanations) to calculate how each feature 
                                    contributes to the final probability. Unlike simple percentages, SHAP properly accounts for feature interactions 
                                    in the machine learning model, ensuring the math adds up correctly:
                                    <strong>{base_value*100:.1f}% + {all_shap_sum:+.2f}pp = {risk_probability_percent:.1f}%</strong>
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as shap_error:
                    st.warning(f"Unable to generate detailed explanation: {str(shap_error)}")
                    st.info("The risk assessment is still valid. Detailed feature contribution analysis is temporarily unavailable.")
                    # Set empty contributions for PDF
                    question_features = None
                
                # ═══════════════════════════════════════════════════════════════════
                # END OF SHAP EXPLANATION SECTION
                # ═══════════════════════════════════════════════════════════════════
                
                # Show feature contribution
                st.markdown("---")
                st.subheader("Assessment Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    aq_score = sum(answers.values())
                    st.metric("AQ-10 Score", f"{aq_score}/10",
                             delta="Higher score = Higher risk" if aq_score > 0 else "No risk indicators",
                             delta_color="inverse" if aq_score >= 6 else "normal")
                
                with col2:
                    st.metric("Child's Age", f"{demographics['age']} years")
                
                with col3:
                    st.metric("Family History", demographics['autism_family'])
                
                # Detailed responses
                with st.expander("View Detailed Response Analysis"):
                    # Define forward scored questions (same as in questions section)
                    forward_scored = ['A1_Score', 'A7_Score', 'A10_Score']
                    answer_list = list(answers.values())
                    question_keys = list(answers.keys())
                    response_df = pd.DataFrame({
                        'Question': [f"Q{i+1}" for i in range(10)],
                        'Score': answer_list,
                        'Scoring Type': ['Forward' if question_keys[i] in forward_scored else 'Reverse' for i in range(10)],
                        'Risk Indicator': ['Present' if answer_list[i] == 1 else 'Absent' for i in range(10)]
                    })
                    st.dataframe(response_df, use_container_width=True)
                    st.caption("**Scoring Methodology:** Score 1 indicates risk indicator present; Score 0 indicates no risk detected. Total AQ-10 score range: 0-10. Clinical screening threshold: ≥6.")
                
                # PDF Download Button
                st.markdown("---")
                st.subheader("Generate Assessment Report")
                
                # Prepare contributions dataframe for PDF
                contributions_for_pdf = None
                if 'question_features' in locals() and question_features is not None:
                    try:
                        contributions_for_pdf = pd.DataFrame({
                            'Question': [q.replace('_Score', '') for q in question_features['Feature']],
                            'Behavior': [question_texts.get(q, q) for q in question_features['Feature']],
                            'Answer': [user_answers_text.get(q, '?') for q in question_features['Feature']],
                            'Contribution_Value': question_features['SHAP_Value'].values * 100
                        })
                    except:
                        contributions_for_pdf = None
                
                try:
                    pdf_buffer = generate_pdf_report(
                        prediction=prediction,
                        probability=risk_probability,
                        answers=answers,
                        child_name=demographics.get('child_name', 'Child'),
                        age=demographics['age'],
                        gender=demographics['gender'],
                        ethnicity=demographics['ethnicity'],
                        jaundice=demographics['jaundice'],
                        autism_family=demographics['autism_family'],
                        country=demographics['country'],
                        used_app=demographics['used_app'],
                        relation=demographics['relation'],
                        model_name=model_name,
                        accuracy=accuracy,
                        threshold=threshold,
                        contributions_df=contributions_for_pdf

                    )
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"Autism_Screening_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                    st.success("PDF report generated successfully and ready for download.")
                    
                except Exception as pdf_error:
                    st.warning(f"Unable to generate PDF report: {str(pdf_error)}")
                    st.info("You can still view and save the results displayed above.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all fields are filled correctly.")
        
        else:
            st.info("Complete the behavioral questions in the previous section, then click 'Generate Risk Assessment' to view results.")
        
        # Navigation buttons
        st.markdown("---")
        col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
        with col_nav1:
            if st.button("Back to Questions", use_container_width=True):
                st.session_state.current_page = 1
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6b7280; padding: 2rem 1rem; background-color: #f8fafc; border-radius: 8px; margin-top: 2rem;">
            <p style="font-weight: 600; margin-bottom: 1rem; color: #1f2937;">Medical Disclaimer</p>
            <p style="font-size: 0.9rem; line-height: 1.6; margin-bottom: 0.5rem;">This screening tool is for informational and preliminary assessment purposes only. It should not replace professional medical advice, diagnosis, or treatment.</p>
            <p style="font-size: 0.9rem; line-height: 1.6; margin-bottom: 1rem;">Always seek the guidance of qualified healthcare providers with questions regarding medical conditions.</p>
            <p style="font-size: 0.85rem; color: #9ca3af;">© 2025 Childhood Autism Risk Prediction System | Machine Learning Based Assessment Tool</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
