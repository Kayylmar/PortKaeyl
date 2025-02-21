import streamlit as st
import pandas as pd
from utils.styling import load_css
import base64


def get_pdf_download_link(pdf_path, filename):
    """Generate download link for PDF file"""
    with open(pdf_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-button">📥 Download CV</a>'
    return href

st.set_page_config(page_title="Profile", page_icon="👤", layout="wide")

load_css()


# Custom CSS
st.markdown("""
    <style>
    .css-1v0mbdj.etr89bj1 {
        text-align: center;
    }
    .profile-img {
        border-radius: 50%;
        margin: 0 auto;
        display: block;
    }
    .social-links {
        text-align: center;
        padding: 1rem 0;
    }
    .experience-card {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        border-left: 3px solid #0366d6;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    # Profile Image
    st.image("static/img/profile.jpg", width=200, output_format="auto")
    
    # Contact Information
    st.markdown("""
    ### Contact
    - 📧 syauqid1@gmail.com
    - 📱 0858 6018 7715
    """)
    
    # Social Links
    st.markdown("""
    ### Social Links
    - [GitHub](https://github.com/Kayylmar)
    - [LinkedIn](https://www.linkedin.com/in/kaylan-damar-syauqi/)
    """)

with col2:
    st.title("Kaylan Damar Syauqi")
    st.subheader("Web Developer")
    st.markdown(get_pdf_download_link("cv/cv_bayuzen_ahmad.pdf", "cv_bayuzen_ahmad.pdf"), unsafe_allow_html=True)
    
    st.markdown("""
    ### Summary
    Seasoned Data Scientist with 4+ years of expertise in machine learning solutions, data engineering, 
    and analytics. Proven track record in leading cross-functional teams and developing end-to-end data 
    pipelines using Python, R, and cloud platforms.
    """)

# Experience Section
st.header("Professional Experience")

experiences = [
    {
        "role": "Web Developer",
        "company": "TELKOMSEL",
        "period": "MAR 2024 – PRESENT",
        "points": [
            "Engineered web analytics ecosystem with Selenium",
            "Implemented fraud detection system using Isolation Forest",
            "Created RFM segmentation model"
        ]
    },
]

for exp in experiences:
    with st.expander(f"{exp['company']} - {exp['role']}", expanded=True):
        st.markdown(f"**Period:** {exp['period']}")
        for point in exp['points']:
            st.markdown(f"- {point}")

# Skills Section
st.header("Skills & Expertise")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Technical Skills")
    technical_skills = [
        "Machine Learning",
        "Data Analysis",
        "Python Programming",
        "SQL",
        "Data Visualization"
    ]
    for skill in technical_skills:
        st.markdown(f"- {skill}")

with col2:
    st.subheader("Tools & Technologies")
    tools = [
        "Python",
        "R Studio",
        "Tableau",
        "Power BI",
        "PySpark"
    ]
    for tool in tools:
        st.markdown(f"- {tool}")

# Education Section
st.header("Education")
st.markdown("""
#### Airlangga University
- **Degree:** Bachelor of Information System
- **GPA:** 3.38/4.00
- **Period:** AUG 2021 - APR 2025
""")

# Certifications
st.header("Certifications")
certifications = [
    "Certified Associate Data Scientist - BNSP Indonesia",
    "Full Stack Data Science - iNeuron",
    "Artificial Engineering: Computer Vision - AI Indonesia"
]

for cert in certifications:
    st.markdown(f"- {cert}")