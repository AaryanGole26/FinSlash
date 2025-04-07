import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Streamlit config
st.set_page_config(page_title="FinSlash - Loan Approval", layout="wide", initial_sidebar_state="expanded")

# JavaScript to detect theme and apply dynamic styles
st.markdown("""
    <script>
    const setThemeStyles = () => {
        const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        document.documentElement.style.setProperty('--text-color', isDarkMode ? '#ffffff' : '#2c3e50');
        document.documentElement.style.setProperty('--subheader-color', isDarkMode ? '#e0e0e0' : '#34495e');
        document.documentElement.style.setProperty('--background-content', isDarkMode ? 'rgba(30, 30, 30, 0.9)' : 'rgba(255, 255, 255, 0.9)');
        document.documentElement.style.setProperty('--footer-color', isDarkMode ? '#b0b0b0' : '#7f8c8d');
    };
    setThemeStyles();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', setThemeStyles);
    </script>
    <style>
    :root {
        --text-color: #2c3e50;
        --subheader-color: #34495e;
        --background-content: rgba(255, 255, 255, 0.9);
        --footer-color: #7f8c8d;
    }
    .stTitle {color: var(--text-color); font-family: 'Arial', sans-serif; font-weight: bold;}
    .stSubheader {color: var(--subheader-color); font-family: 'Arial', sans-serif;}
    .stMarkdown {font-family: 'Arial', sans-serif; color: var(--text-color);}
    .stSelectbox {background-color: #ecf0f1; border-radius: 5px; padding: 5px;}
    .stFileUploader {background-color: #ecf0f1; border-radius: 5px; padding: 10px;}
    .prediction-approved {color: #27ae60; font-size: 24px; font-weight: bold;}
    .prediction-rejected {color: #c0392b; font-size: 24px; font-weight: bold;}
    .sidebar .sidebar-content {background-color: #34495e; color: white;}
    .footer {text-align: center; color: var(--footer-color); font-size: 12px; padding: 20px;}
    .content-wrapper {
        background-color: var(--background-content);
        padding: 25px;
        border-radius: 15px;
        margin: 20px;
    }
    /* Ensure sidebar text remains readable */
    section[data-testid="stSidebar"] {
        background-image: url("https://w0.peakpx.com/wallpaper/981/932/HD-wallpaper-personal-loan-emi-calculator-personal-loan-personal-loan-india-fianance-loans-banking.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        color: white !important;
    }
    section[data-testid="stSidebar"] .css-1cpxqw2, /* Sidebar title */
    section[data-testid="stSidebar"] .css-16huue1, /* Sidebar radio options */
    section[data-testid="stSidebar"] .css-1fj2g2n, /* Radio text */
    section[data-testid="stSidebar"] .css-1d391kg, /* Selected radio text */
    section[data-testid="stSidebar"] label, /* Labels in sidebar */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6,
    section[data-testid="stSidebar"] p {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
    }
    section[data-testid="stSidebar"] .stRadio > div {
        background-color: rgba(0, 0, 0, 0.3);
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Us", "Dashboard"])

# Load training dataset
@st.cache_data
def load_data():
    df = pd.read_csv('datasets/train.csv')
    return df

df = load_data()

if 'Dependents' in df.columns:
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(float)

# Dashboard Page
if page == "Dashboard":
    st.markdown("<div class='content-wrapper'>", unsafe_allow_html=True)

    st.title("\U0001F3E6 Your AI-powered Loan Approval Dashboard")

    # Preprocess training data
    df.fillna({
        'Gender': df['Gender'].mode()[0],
        'Married': df['Married'].mode()[0],
        'Dependents': df['Dependents'].mode()[0],
        'Self_Employed': df['Self_Employed'].mode()[0],
        'Credit_History': df['Credit_History'].mode()[0],
        'Loan_Amount_Term': df['Loan_Amount_Term'].mode()[0],
        'LoanAmount': df['LoanAmount'].median(),
        'ApplicantIncome': df['ApplicantIncome'].median(),
        'CoapplicantIncome': df['CoapplicantIncome'].median()
    }, inplace=True)

    if df.isnull().sum().sum() > 0:
        st.error("Training data still contains NaNs after preprocessing!")
        st.write(df.isnull().sum())
        st.stop()

    df_encoded = df.copy()
    le = LabelEncoder()
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    X = df_encoded.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df_encoded['Loan_Status']

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    st.markdown("### Upload Candidate Data", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload `candidates.csv` to predict loan status", type=["csv"])

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)

        if 'Dependents' in test_df.columns:
            test_df['Dependents'] = test_df['Dependents'].replace('3+', 3).astype(float)

        test_df.fillna({
            'Gender': test_df['Gender'].mode()[0],
            'Married': test_df['Married'].mode()[0],
            'Dependents': test_df['Dependents'].mode()[0],
            'Self_Employed': test_df['Self_Employed'].mode()[0],
            'Credit_History': test_df['Credit_History'].mode()[0],
            'Loan_Amount_Term': test_df['Loan_Amount_Term'].mode()[0],
            'LoanAmount': test_df['LoanAmount'].median(),
            'ApplicantIncome': test_df['ApplicantIncome'].median(),
            'CoapplicantIncome': test_df['CoapplicantIncome'].median()
        }, inplace=True)

        if test_df.isnull().sum().sum() > 0:
            st.error("Test data still contains NaNs after preprocessing!")
            st.write(test_df.isnull().sum())
            st.stop()

        encode_map = {
            'Gender': {'Male': 1, 'Female': 0},
            'Married': {'Yes': 1, 'No': 0},
            'Education': {'Graduate': 1, 'Not Graduate': 0},
            'Self_Employed': {'Yes': 1, 'No': 0},
            'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
        }

        for col, mapping in encode_map.items():
            if col in test_df.columns:
                test_df[col] = test_df[col].map(mapping)

        X_test = test_df[X.columns]
        test_df['Prediction'] = model.predict(X_test)
        test_df['Prediction'] = test_df['Prediction'].map({1: '✅ Approved', 0: '❌ Rejected'})

        sorted_loan_ids = sorted(test_df['Loan_ID'])
        st.markdown("#### Select a Candidate", unsafe_allow_html=True)
        selected_id = st.selectbox("Loan ID", sorted_loan_ids, label_visibility="collapsed")
        selected = test_df[test_df['Loan_ID'] == selected_id].iloc[0]

        pred_class = "prediction-approved" if "Approved" in selected['Prediction'] else "prediction-rejected"
        st.markdown(f"### Prediction: <span class='{pred_class}'>{selected['Prediction']}</span>", unsafe_allow_html=True)

        st.markdown("#### Applicant Details", unsafe_allow_html=True)
        applicant_details = {
            'Gender': 'Male' if selected['Gender'] == 1 else 'Female',
            'Married': 'Yes' if selected['Married'] == 1 else 'No',
            'Education': 'Graduate' if selected['Education'] == 1 else 'Not Graduate',
            'Self_Employed': 'Yes' if selected['Self_Employed'] == 1 else 'No',
            'ApplicantIncome': f"${int(selected['ApplicantIncome']):,}",
            'CoapplicantIncome': f"${int(selected['CoapplicantIncome']):,}",
            'LoanAmount': f"${int(selected['LoanAmount']):,}",
            'Loan_Amount_Term': f"{int(selected['Loan_Amount_Term'])} months",
            'Credit_History': "Good" if int(selected['Credit_History']) == 1 else "Bad",
            'Property_Area': 'Urban' if selected['Property_Area'] == 2 else 'Semiurban' if selected['Property_Area'] == 1 else 'Rural'
        }
        st.dataframe(pd.DataFrame([applicant_details]), use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(5, 3), facecolor='none')
            ax1.bar(['Applicant', 'Coapplicant'], 
                    [selected['ApplicantIncome'], selected['CoapplicantIncome']], 
                    color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=1)
            ax1.set_title('Income Comparison', fontsize=12, color='var(--text-color)', pad=10)
            ax1.set_ylabel('Income ($)', fontsize=10, color='var(--text-color)')
            ax1.tick_params(axis='both', labelsize=8, colors='var(--text-color)')
            plt.tight_layout()
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 3), facecolor='none')
            property_areas = ['Rural', 'Semiurban', 'Urban']
            values = [1 if selected['Property_Area'] == 0 else 0,
                      1 if selected['Property_Area'] == 1 else 0,
                      1 if selected['Property_Area'] == 2 else 0]
            colors = ['#e74c3c', '#f1c40f', '#3498db']
            non_zero_indices = [i for i, v in enumerate(values) if v > 0]
            ax2.pie([values[i] for i in non_zero_indices], 
                    labels=[property_areas[i] for i in non_zero_indices], 
                    autopct='%1.1f%%', startangle=90, colors=[colors[i] for i in non_zero_indices], 
                    textprops={'fontsize': 10, 'color': 'var(--text-color)'}, 
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1})
            ax2.set_title('Property Area', fontsize=12, color='var(--text-color)', pad=10)
            ax2.axis('equal')
            st.pyplot(fig2)

        with col3:
            fig3, ax3 = plt.subplots(figsize=(5, 3), facecolor='none')
            months = range(1, int(selected['Loan_Amount_Term']) + 1)
            loan_repayment = [selected['LoanAmount'] / selected['Loan_Amount_Term']] * len(months)
            ax3.plot(months, loan_repayment, marker='o', color='#9b59b6', linewidth=2, markersize=5)
            ax3.set_title('Repayment Schedule', fontsize=12, color='var(--text-color)', pad=10)
            ax3.set_xlabel('Months', fontsize=10, color='var(--text-color)')
            ax3.set_ylabel('Monthly ($)', fontsize=10, color='var(--text-color)')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.tick_params(axis='both', labelsize=8, colors='var(--text-color)')
            plt.tight_layout()
            st.pyplot(fig3)

    st.markdown("</div>", unsafe_allow_html=True)

# About Us Page
elif page == "About Us":
    st.title("\U0001F4B0 FinSlash")
    st.subheader("Simplifying Loan Approvals with AI")

    st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #ecf0f1; border-radius: 10px;'>
            <h3 style='color: var(--text-color);'>Welcome to FinSlash</h3>
            <p style='color: var(--subheader-color); font-size: 16px;'> 
                We’re transforming the way loan approvals work—making them faster, smarter, and fairer with cutting-edge AI.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### \U0001F680 Who We Are")
    st.markdown("""
        **FinSlash** is brought to you by the **3JT Team**, a passionate group of innovators at the intersection of finance and technology.
        Our goal? To empower financial institutions and individuals with tools that make loan decisions transparent, efficient, and data-driven.
    """, unsafe_allow_html=True)

    st.markdown("### \U0001F4CA What We Offer")
    st.markdown("""
        - **Accurate Predictions**
        - **User-Friendly Interface**
        - **Transparency**
        - **Scalability**
    """, unsafe_allow_html=True)

    st.markdown("### \U0001F52D Our Vision")
    st.markdown("""
        We dream of a world where financial decisions are instant, unbiased, and accessible to all.
        By blending AI with financial expertise, we’re paving the way for a future of smarter lending and greater inclusion.
    """, unsafe_allow_html=True)

    st.markdown("### \U0001F465 Meet the 3JT Team")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            **Aaryan Gole** 🧙‍♂️  <br>
            AI Analyst and Backend Developer <br>
            📧 aaryan.224767101@vcet.edu.in <br>
            📞 +91 93097 44137
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            **Swarup Kakade** 💼  <br>
            Data Scientist and Backend Developer <br>
            📧 swarup.224837102@vcet.edu.in <br>
            📞 +91 93216 37437
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            **Nitish Jha** 💻  <br>
            Frontend Developer and UI/UX Designer <br>
            📧 nitish.224827101@vcet.edu.in <br>
            📞 +91 73852 08766
        """, unsafe_allow_html=True)

    st.markdown("### \U0001F4E9 Get in Touch")
    st.markdown("""
        Got ideas, questions, or just want to say hi?  <br>
        📩 Reach out at: **aryn.gole@gmail.com**  <br>
        🔗 Follow us on [X](https://x.com/finslash)
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center; margin-top: 20px; color: var(--footer-color);'>
            <i>“Lending made simple. Decisions made smart.”</i>
        </div>
    """, unsafe_allow_html=True)

# Footer (on all pages)
st.markdown("""
    <div class='footer'>
        © 2025 FinSlash | Powered by Streamlit | Built with ❤️ by the 3JT Team ⚔️
    </div>
""", unsafe_allow_html=True)
