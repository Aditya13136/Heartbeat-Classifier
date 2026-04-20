import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import zipfile
import os
import tempfile
import pickle

from tensorflow.keras.models import load_model

# =========================
# Load Model + Scaler
# =========================
model = load_model("ecg_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))  # MUST be trained scaler

window_size = 200
class_names = ['Normal', 'Abnormal']

# =========================
# UI
# =========================
st.title("💓 ECG Heartbeat Classifier (WFDB ZIP Upload)")
st.write("Upload a ZIP file containing .dat + .hea + .atr files")

# =========================
# Upload ZIP
# =========================
uploaded_zip = st.file_uploader("Upload ECG ZIP file", type=["zip"])

if uploaded_zip is not None:

    with tempfile.TemporaryDirectory() as tmpdir:

        # Save ZIP
        zip_path = os.path.join(tmpdir, "ecg.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Find record
        hea_files = [f for f in os.listdir(tmpdir) if f.endswith(".hea")]

        if len(hea_files) == 0:
            st.error("No .hea file found in ZIP")
        else:
            record_name = hea_files[0].replace(".hea", "")
            record_path = os.path.join(tmpdir, record_name)

            # =========================
            # Load ECG Signal
            # =========================
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal[:, 0]

            st.subheader("Raw ECG Signal")
            fig, ax = plt.subplots()
            ax.plot(signal[:2000])
            ax.set_title("ECG Signal (First 2000 samples)")
            st.pyplot(fig)

            # =========================
            # Check signal length
            # =========================
            if len(signal) < window_size:
                st.error("Signal too short for prediction")
            else:

                # =========================
                # FIXED PREPROCESSING
                # =========================

                beat = signal[:window_size]      # (200,)
                beat = beat.reshape(1, -1)       # (1, 200)

                beat_scaled = scaler.transform(beat)  # (1, 200)

                input_data = beat_scaled.reshape(1, window_size, 1)

                # =========================
                # Prediction
                # =========================
                prediction = model.predict(input_data)
                pred_class = np.argmax(prediction)
                confidence = np.max(prediction)

                st.subheader(f"Prediction: {class_names[pred_class]}")
                st.write(f"Confidence: {confidence * 100:.2f}%")

                # =========================
                # Plot processed signal
                # =========================
                fig, ax = plt.subplots()
                ax.plot(beat_scaled.flatten())
                ax.set_title("Processed ECG Beat")
                st.pyplot(fig)

                # =========================
                # Optional annotations
                # =========================
                try:
                    ann = wfdb.rdann(record_path, "atr")
                    st.write(f"Annotations found: {len(ann.sample)} beats")
                except:
                    st.write("No annotations found or skipped.")