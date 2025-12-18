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
def generate_pdf_report(prediction, probability, answers, age, gender, ethnicity, jaundice, 
                       autism_family, country, used_app, relation, model_name, accuracy):
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
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    elements.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
    elements.append(Paragraph(f"<b>Model Used:</b> {model_name} (Accuracy: {accuracy:.1%})", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Risk Assessment Result
    risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
    risk_color = colors.red if prediction == 1 else colors.green
    risk_prob = probability[1] * 100
    
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
        
        # Collect answers in columns for better layout
        col1, col2 = st.columns(2)
        answers = {}
        
        # Forward scored questions (YES = risk): A1, A7, A10
        forward_scored = ['A1_Score', 'A7_Score', 'A10_Score']
        # Reverse scored questions (NO = risk): A2, A3, A4, A5, A6, A8, A9
        
        for idx, (key, question) in enumerate(questions.items()):
            with col1 if idx % 2 == 0 else col2:
                answer = st.radio(
                    f"**Q{idx+1}**: {question}",
                    options=["Yes", "No"],
                    index=0,  # Default to "Yes"
                    key=key,
                    horizontal=True
                )
                
                # Apply correct scoring based on question type
                if key in forward_scored:
                    # Forward scored: YES = 1 (risk), NO = 0
                    answers[key] = 1 if answer == "Yes" else 0
                else:
                    # Reverse scored: NO = 1 (risk), YES = 0
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
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Display results
                risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
                risk_class = "high-risk" if prediction == 1 else "low-risk"
                risk_probability = prediction_proba[1] * 100
                
                st.markdown(f"""
                    <div class="prediction-box {risk_class}">
                        <h2 style="text-align: center; margin-bottom: 0.5rem; font-weight: 700; font-size: 1.8rem;">
                            SCREENING RESULT: {risk_level}
                        </h2>
                        <h3 style="text-align: center; font-weight: 500; font-size: 1.3rem; color: #4a5568;">
                            Risk Probability: {risk_probability:.1f}%
                        </h3>
                        <p style="text-align: center; margin-top: 1rem; font-size: 0.9rem; color: #6b7280;">
                            AQ-10 Score: {sum(answers.values())}/10
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detailed interpretation
                st.subheader("Clinical Interpretation")
                
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
                
                # Show feature contribution
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
                
                try:
                    pdf_buffer = generate_pdf_report(
                        prediction=prediction,
                        probability=prediction_proba,
                        answers=answers,
                        age=demographics['age'],
                        gender=demographics['gender'],
                        ethnicity=demographics['ethnicity'],
                        jaundice=demographics['jaundice'],
                        autism_family=demographics['autism_family'],
                        country=demographics['country'],
                        used_app=demographics['used_app'],
                        relation=demographics['relation'],
                        model_name=model_name,
                        accuracy=accuracy
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
