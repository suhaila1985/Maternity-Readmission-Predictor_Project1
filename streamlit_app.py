"""
Maternity Patient Readmission Risk Prediction Dashboard
Stable Production-Safe Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Maternity Readmission Risk",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# 1️⃣ LOAD & TRAIN MODEL (STABLE VERSION)
# =====================================================
@st.cache_resource
def load_and_train_model():

    try:
        df = pd.read_csv("maternity_data.csv")
    except:
        st.info("Using demo data (maternity_data.csv not found)")
        np.random.seed(42)
        df = pd.DataFrame({
            'PatientID': range(1000),
            'Age': np.random.uniform(18,45,1000),
            'LaborDuration': np.random.uniform(1,16,1000),
            'LengthofStaydays': np.random.uniform(2,15,1000),
            'Location': np.random.choice(['Urban','Rural'],1000),
            'Complications': np.random.choice(['No','Yes'],1000,p=[0.7,0.3]),
            'Readmitted': np.random.choice(['No','Yes'],1000,p=[0.75,0.25])
        })

    # Cleaning
    df = df[(df['Age']>=18)&(df['Age']<=45)]
    df['Readmitted'] = (df['Readmitted']=="Yes").astype(int)
    df['Location_Encoded'] = (df['Location']=="Rural").astype(int)
    df['Complications_Encoded'] = (df['Complications']=="Yes").astype(int)

    features = ['Age','LaborDuration','LengthofStaydays',
                'Location_Encoded','Complications_Encoded']

    X = df[features]
    y = df['Readmitted']

    # Ensure both classes exist
    if len(y.unique()) < 2:
        st.error("Only one class found in dataset. Cannot train model.")
        st.stop()

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=5
    )

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    # Safe probability extraction
    class_index = list(model.classes_).index(1)
    y_proba = model.predict_proba(X_test)[:,class_index]

    accuracy = accuracy_score(y_test,y_pred)

    try:
        auc = roc_auc_score(y_test,y_proba)
    except:
        auc = None

    cm = confusion_matrix(y_test,y_pred)

    return model, accuracy, auc, cm, features


model, accuracy, auc, cm, feature_cols = load_and_train_model()

# =====================================================
# 2️⃣ SIDEBAR INPUT
# =====================================================
with st.sidebar:
    st.header("Patient Input")

    age = st.slider("Age",18,45,30)
    labor = st.slider("Labor Duration (hrs)",1.0,16.0,8.0)
    los = st.slider("Length of Stay (days)",2.0,16.0,7.0)
    location = st.selectbox("Location",['Urban','Rural'])
    complications = st.selectbox("Complications",['No','Yes'])

location_encoded = 1 if location=="Rural" else 0
comp_encoded = 1 if complications=="Yes" else 0

patient = np.array([[age,labor,los,location_encoded,comp_encoded]])

# Safe probability extraction
class_index = list(model.classes_).index(1)
risk_probability = model.predict_proba(patient)[0][class_index]
prediction = model.predict(patient)[0]

# =====================================================
# 3️⃣ RISK LOGIC
# =====================================================
threshold = 0.5  # adjustable

if risk_probability >= 0.65:
    risk_level = "🔴 HIGH RISK"
    color = "red"
elif risk_probability >= 0.4:
    risk_level = "🟡 MODERATE RISK"
    color = "orange"
else:
    risk_level = "🟢 LOW RISK"
    color = "green"

# =====================================================
# 4️⃣ MAIN DISPLAY
# =====================================================
st.title("🏥 Maternity Readmission Risk Prediction")

col1,col2,col3 = st.columns(3)

col1.metric("Risk Score",f"{risk_probability:.1%}")
col2.markdown(f"<h3 style='color:{color}'>{risk_level}</h3>",unsafe_allow_html=True)
col3.metric("Model Accuracy",f"{accuracy:.1%}")

st.divider()

# =====================================================
# 5️⃣ CONFUSION MATRIX
# =====================================================
st.subheader("Model Performance")

if auc:
    st.write(f"AUC Score: {auc:.3f}")
else:
    st.write("AUC Score: Not available (single-class issue)")

fig, ax = plt.subplots()
ax.matshow(cm)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j,i,str(cm[i,j]),va='center',ha='center')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.divider()

st.info("⚠️ This tool is for research/demo purposes only. Clinical validation required.")
