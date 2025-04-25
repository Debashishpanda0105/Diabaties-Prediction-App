import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import io

logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)

@st.cache_data
def load_data():
    file_path = r"C:\Users\debup\OneDrive\Desktop\Diabetes\data\diabetes.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"🚫 File not found at: `{file_path}`")
        return None
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns.tolist(), df

def main():
    st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")
    st.title("🩺 Diabetes Prediction App")

    data = load_data()
    if data is None:
        return
    (X_train, X_test, y_train, y_test), feature_names, full_df = data

    # 🔹 Overview Dashboard
    st.markdown("### 📊 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size", len(full_df))
    col2.metric("Features", len(feature_names))
    col3.metric("Train/Test Split", "80% / 20%")

    # Tabs Layout
    tab1, tab2, tab3 = st.tabs(["📋 Patient Input", "📈 Model Output", "📊 Reports"])

    with tab1:
        st.sidebar.header("🧬 Input Patient Health Data")
        user_input = []
        for feature in feature_names:
            min_val = float(X_train[feature].min())
            max_val = float(X_train[feature].max())
            default_val = float(X_train[feature].mean())
            val = st.sidebar.slider(f"{feature}", min_val, max_val, default_val)
            user_input.append(val)

        input_data = pd.DataFrame([user_input], columns=feature_names)

        st.markdown("### 👤 Patient Data Summary")
        st.write("**Review the entered data for the patient:**")
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

    with tab2:
        # Model and Prediction Section
        k = st.sidebar.slider("🔢 Choose number of neighbors (K)", 1, 15, 5)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        input_scaled = scaler.transform(input_data)

        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][prediction]

        st.metric("✅ Model Accuracy", f"{accuracy * 100:.2f}%")
        st.metric("🧪 Prediction", "Diabetic" if prediction == 1 else "Not Diabetic", delta=f"{proba*100:.2f}% Confidence")

        # Result
        st.markdown("---")
        if prediction == 1:
            st.error("🚨 **The patient is likely Diabetic.** Please consult a healthcare provider.")
            st.markdown("### 🩺 Recommended Tips & Precautions")
            st.markdown("""
            - 🥗 **Maintain a balanced diet** rich in fiber and low in refined sugars.
            - 🚶 **Exercise regularly** (30 minutes a day, 5 days a week).
            - 💧 **Stay hydrated** — drink plenty of water.
            - 💊 **Take medications** as prescribed — never skip doses.
            - 🧂 **Limit sodium intake** to keep blood pressure in check.
            - 🧘 **Manage stress** through yoga, meditation, or mindfulness.
            - 🦶 **Check your feet daily** for cuts, swelling, or blisters.
            - 🩸 **Monitor blood sugar levels** regularly and keep a log.
            - 🚭 **Avoid smoking and limit alcohol** — both increase complications.
            - 🩻 **Schedule regular checkups** for eyes, kidneys, and heart.
            ⚠️ **Always follow up with a certified medical professional for a personalized care plan.**
            """)
        else:
            st.success("🟢 **The patient is not Diabetic.** Keep up the healthy lifestyle!")
            st.markdown("### 🙌 Prevention Tips to Stay Healthy")
            st.markdown("""
            - 🥦 Eat plenty of vegetables, fruits, and whole grains.
            - 🏃 Engage in physical activity regularly.
            - ⚖️ Maintain a healthy weight.
            - 🛌 Get quality sleep (7–9 hours).
            - 🧘 Reduce stress to avoid hormonal imbalance.
            """)

    with tab3:
        # Classification Report
        st.markdown("### 📊 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        # Confusion Matrix
        st.markdown("### 🔎 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Distribution of Feature Example
        st.markdown("### 📌 Feature Distribution by Outcome")
        feat_selected = st.selectbox("Select feature to visualize:", feature_names)
        fig_dist = px.histogram(full_df, x=feat_selected, color="Outcome", barmode="overlay", nbins=30)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Correlation Heatmap
        st.markdown("### 📈 Feature Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(full_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

        # Download Report
        st.markdown("### 📥 Download Report")
        prediction_text = "Diabetic" if prediction == 1 else "Not Diabetic"
        report_text = f"""
        Diabetes Prediction Report

        Prediction: {prediction_text}
        Confidence: {proba * 100:.2f}%
        Model Accuracy: {accuracy * 100:.2f}%
        Selected K: {k}

        Input Data:
        {input_data.to_string(index=False)}
        """

        buffer = io.BytesIO()
        buffer.write(report_text.encode())
        buffer.seek(0)
        st.download_button("📄 Download Prediction Report", buffer, file_name="diabetes_report.txt")

    # Footer
    st.markdown("---")
    st.markdown("Made with by **Debashish Panda*")

if __name__ == "__main__":
    main()
