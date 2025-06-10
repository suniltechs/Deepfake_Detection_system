import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
import matplotlib.pyplot as plt
from transformers import pipeline
import pandas as pd
import uuid
from transformers import ViTFeatureExtractor, TFAutoModelForImageClassification
import os
from dotenv import load_dotenv

# Streamlit configuration
st.set_page_config(
    page_title="DeepGuard : Unmask the Truth", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🕵️"
)

# Custom CSS with improved dark/light mode
st.markdown("""
<style>
    :root {
        --primary: #4f46e5;
        --primary-light: #6366f1;
        --primary-dark: #4338ca;
        --secondary: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --info: #3b82f6;
        --light: #f8fafc;
        --dark: #0f172a;
        --gray: #64748b;
        --gray-light: #e2e8f0;
        --card-bg: #ffffff;
        --card-border: #e2e8f0;
        --body-bg: #f8fafc;
        --text: #334155;
        --text-muted: #64748b;
    }

    [data-theme="dark"] {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --primary-dark: #4f46e5;
        --secondary: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --info: #3b82f6;
        --light: #1e293b;
        --dark: #f8fafc;
        --gray: #94a3b8;
        --gray-light: #334155;
        --card-bg: #1e293b;
        --card-border: #334155;
        --body-bg: #0f172a;
        --text: #e2e8f0;
        --text-muted: #94a3b8;
    }

    html, body, .stApp {
        background-color: var(--body-bg) !important;
        color: var(--text) !important;
        transition: all 0.3s ease;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text) !important;
    }

    /* Header */
    .header {
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .subheader {
        color: var(--primary) !important;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Cards */
    .card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton>button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* File uploader */
    .stFileUploader>div>div>div {
        border: 2px dashed var(--gray-light) !important;
        background-color: var(--card-bg) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
    }

    .stFileUploader>div>div>div:hover {
        border-color: var(--primary) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--card-bg) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        margin: 0 !important;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: white !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text) !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: var(--text-muted) !important;
    }

    /* Progress bar */
    .stProgress>div>div>div {
        background-color: var(--primary) !important;
    }

    /* Sidebar */
    .stSidebar {
        background-color: var(--card-bg) !important;
    }

    /* Dark mode toggle */
    .toggle-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
    }

    .toggle-label {
        font-weight: 500;
        color: var(--text);
    }

    .toggle-switch {
        position: relative;
        display: inline-block;
        width: 50px;
        height: 24px;
    }

    .toggle-switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }

    .slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: var(--gray-light);
        transition: .4s;
        border-radius: 24px;
    }

    .slider:before {
        position: absolute;
        content: "";
        height: 16px;
        width: 16px;
        left: 4px;
        bottom: 4px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
    }

    input:checked + .slider {
        background-color: var(--primary);
    }

    input:checked + .slider:before {
        transform: translateX(26px);
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }

    .badge-fake {
        background-color: var(--danger);
    }

    .badge-real {
        background-color: var(--secondary);
    }

    /* Custom upload area */
    .upload-area {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        border: 2px dashed var(--gray-light);
        border-radius: 12px;
        background-color: var(--card-bg);
        transition: all 0.3s ease;
        cursor: pointer;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .upload-area:hover {
        border-color: var(--primary);
    }

    .upload-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        color: var(--primary);
    }

    /* Results container */
    .results-container {
        margin-top: 1.5rem;
    }

    /* Alert boxes */
    .stAlert {
        border-radius: 8px !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--gray-light);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
            
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--card-bg);
        color: var(--text);
        text-align: center;
        padding: 1rem 0.5rem;
        border-top: 1px solid var(--card-border);
        z-index: 100;
        font-size: 0.95rem;
    }

    .footer a {
        color: var(--primary);
        text-decoration: none;
        margin: 0 8px;
    }

    .footer a:hover {
        color: var(--primary-dark);
        text-decoration: underline;
    }

    .social-icon {
        font-size: 1.3rem;
        vertical-align: middle;
        margin-left: 4px;
    }

</style>
""", unsafe_allow_html=True)

# JavaScript for dark/light mode toggle
st.markdown("""
<script>
    // Function to set theme
    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
    }
    
    // Check for saved theme or use system preference
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme) {
        setTheme(savedTheme);
        document.getElementById('theme-toggle').checked = (savedTheme === 'dark');
    } else if (systemPrefersDark) {
        setTheme('dark');
        document.getElementById('theme-toggle').checked = true;
    } else {
        setTheme('light');
        document.getElementById('theme-toggle').checked = false;
    }
    
    // Toggle theme when switch is clicked
    document.getElementById('theme-toggle').addEventListener('change', function() {
        if (this.checked) {
            setTheme('dark');
        } else {
            setTheme('light');
        }
    });
</script>
""", unsafe_allow_html=True)

# Sidebar with theme toggle and settings
with st.sidebar:
    st.markdown("""
    <div class="toggle-container">
        <span class="toggle-label">🌙 Dark Mode</span>
        <label class="toggle-switch">
            <input type="checkbox" id="theme-toggle">
            <span class="slider"></span>
        </label>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Settings")
    mode = st.radio(
        "Input Type",
        ["Image", "Video"],
        key="mode",
        help="Choose to analyze an image or video for deepfake detection."
    )
    
    with st.expander("Advanced Options", expanded=False):
        if mode == "Video":
            default_model = "google/vit-base-patch16-224"
            model_id = st.text_input(
                "Hugging Face Model ID",
                value=default_model,
                help="Enter the Hugging Face model ID for video analysis."
            )
        threshold = st.slider(
            "Fake Score Threshold",
            0.1, 0.9, 0.5 if mode == "Image" else 0.1, 0.05,
            help="Set the threshold for classifying content as fake."
        )

# 1️⃣ Load image model (MesoNet)
@st.cache_resource
def load_image_model():
    try:
        model = load_model('mesonet_deepfake_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading image model: {e}")
        return None

# 2️⃣ Load Hugging Face pipeline for video
@st.cache_resource
def load_video_model(model_id, token=None):
    try:
        model = TFAutoModelForImageClassification.from_pretrained(model_id, token=token)
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_id, token=token)
        return model, feature_extractor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# 3️⃣ Preprocess image for MesoNet
def preprocess_image(image, target_size=(256, 256)):
    img = image.resize(target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# 4️⃣ Preprocess video frame for HF model
def preprocess_frame(frame):
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])
    frame_rgb = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(frame_rgb)

# 5️⃣ Detect and crop face
def preprocess_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_crop = frame[max(0, y):y+h, max(0, x):x+w]
        face_resized = cv2.resize(face_crop, (224, 224))
        return face_resized, (x, y, w, h)
    return None, None

# 6️⃣ Predict for image (MesoNet)
def predict_image(model, image, threshold=0.5):
    processed = preprocess_image(image)
    pred = model.predict(processed, verbose=0)
    fake_score = pred[0][0]
    real_score = pred[0][1]
    label = "Fake" if fake_score > threshold else "Real"
    confidence = fake_score * 100 if label == "Fake" else real_score * 100
    return label, confidence, fake_score, real_score

# Helper function for ViT prediction
def vit_predict(model, feature_extractor, image):
    inputs = feature_extractor(images=image, return_tensors="tf")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = tf.argmax(logits, axis=-1).numpy()[0]
    confidence = tf.nn.softmax(logits, axis=-1).numpy()[0][predicted_class]
    return predicted_class, confidence

# Main Streamlit app
def main():
    # Hero section
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="header">DeepGuard</h1>
        <p style="font-size: 1.1rem; color: var(--text-muted); max-width: 800px; margin: 0 auto;">
            Advanced AI-powered deepfake detection of manipulated media with state-of-the-art neural networks.
            Upload images or videos to analyze for deepfake content.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if mode == "Image":
        st.markdown('### 📷 Image Analysis', unsafe_allow_html=True)
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div  class="upload-area">
                    <div class="upload-icon">📁</div>
                    <h3>Upload Image</h3>
                    <p style="color: var(--text-muted);">Supported formats: JPG, JPEG, PNG</p>
                </div>
                """, unsafe_allow_html=True)
                
                img_file = st.file_uploader(
                    " ",
                    type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed"
                )

            image_model = load_image_model()
            if image_model is None:
                st.warning("Image model could not be loaded. Ensure 'mesonet_deepfake_model.h5' is in this folder.")
                return

            if img_file:
                img = Image.open(img_file)
                with col1:
                    st.image(img, caption="Uploaded Image", use_container_width=True)

                with st.spinner("🔍 Analyzing image content..."):
                    pred_label, pred_conf, fake_score, real_score = predict_image(image_model, img, threshold)

                # Enhanced results display
                with st.container():
                    st.markdown("""
                    <div class="card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h3 style="margin: 0;">Detection Results</h3>
                            <span class="badge %s">%s (%.1f%%)</span>
                        </div>
                    """ % (
                        "badge-fake" if pred_label == "Fake" else "badge-real",
                        pred_label,
                        pred_conf
                    ), unsafe_allow_html=True)
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Fake Score", f"{fake_score:.3f}", delta=None)
                    with cols[1]:
                        st.metric("Real Score", f"{real_score:.3f}", delta=None)
                    with cols[2]:
                        st.metric("Threshold", f"{threshold:.2f}")
                    with cols[3]:
                        st.metric("Model", "MesoNet")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                if pred_label == "Real":
                    st.success("✅ Authentic content detected with high confidence.")
                else:
                    st.error("⚠️ Potential deepfake detected. Please verify this content.")

                # Visualization
                with st.container():
                    st.markdown("### Confidence Scores")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(['Fake', 'Real'], [fake_score, real_score], color=['#ef4444', '#10b981'])
                    ax.axvline(x=threshold, color='#3b82f6', linestyle='--', label='Threshold')
                    ax.set_xlim(0, 1)
                    ax.set_title('Confidence Scores', pad=20)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

    else:  # Video mode
        st.markdown('### 🎥 Video Analysis', unsafe_allow_html=True)
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div class="upload-area">
                    <div class="upload-icon">🎬</div>
                    <h3>Upload Video</h3>
                    <p style="color: var(--text-muted);">Supported formats: MP4, AVI, MOV</p>
                </div>
                """, unsafe_allow_html=True)
                
                vid_file = st.file_uploader(
                    " ",
                    type=["mp4", "avi", "mov"],
                    label_visibility="collapsed"
                )

            load_dotenv() 
            token = os.getenv("HUGGINGFACE_TOKEN")
            video_model, feature_extractor = load_video_model(model_id, token)

            if video_model is None:
                st.warning(f"Could not load model '{model_id}'. Check model name or token on HuggingFace.")
                return

            if vid_file:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tmp.write(vid_file.read())
                tmp.close()

                cap = cv2.VideoCapture(tmp.name)
                stframe = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                current = 0
                preds, frame_scores = [], []
                window = 10
                faces_detected = 0
                skipped = 0
                debug_dir = "debug_frames"
                os.makedirs(debug_dir, exist_ok=True)

                # Video info card
                with st.container():
                    st.markdown("""
                    <div class="card">
                        <h3 style="margin-top: 0;">Video Information</h3>
                    """, unsafe_allow_html=True)
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Total Frames", frame_count)
                    with cols[1]:
                        st.metric("FPS", f"{fps:.1f}")
                    with cols[2]:
                        st.metric("Duration", f"{duration:.1f}s")
                    st.markdown('</div>', unsafe_allow_html=True)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if current % 8 == 0:
                        face, bbox = preprocess_face(frame)
                        if face is not None:
                            pil = preprocess_frame(face)
                            predicted_class, confidence = vit_predict(video_model, feature_extractor, pil)
                            label = video_model.config.id2label[predicted_class]

                            if confidence > 0.6:
                                faces_detected += 1
                                preds.append(confidence)
                                frame_scores.append({'Frame': current, 'FAKE_Score': confidence, 'Time (s)': current/fps})

                                smooth = np.mean(preds[-window:]) if len(preds) >= window else confidence
                                label = "FAKE" if smooth > threshold else "REAL"
                                color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
                                x, y, w, h = bbox
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                cv2.putText(
                                    frame,
                                    f"{label} ({smooth:.2f})",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    color,
                                    2
                                )
                                cv2.imwrite(os.path.join(debug_dir, f"frame_{current}.jpg"), frame)
                            else:
                                skipped += 1

                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    current += 1
                    progress = min(current / frame_count, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {current}/{frame_count} ({int(progress*100)}%)")

                cap.release()
                os.unlink(tmp.name)

                if preds:
                    avg = np.mean(preds)
                    dyn = np.percentile(preds, 90)
                    final = "FAKE" if avg > threshold else "REAL"

                    # Enhanced results display
                    with st.container():
                        st.markdown("""
                        <div class="card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <h3 style="margin: 0;">Video Analysis Summary</h3>
                                <span class="badge %s">%s</span>
                            </div>
                        """ % (
                            "badge-fake" if final == "FAKE" else "badge-real",
                            final
                        ), unsafe_allow_html=True)
                        
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Faces Detected", f"{faces_detected}", f"{(faces_detected/(frame_count//8)*100):.1f}%")
                        with cols[1]:
                            st.metric("Avg Fake Score", f"{avg:.3f}")
                        with cols[2]:
                            st.metric("90th Percentile", f"{dyn:.3f}")
                        with cols[3]:
                            st.metric("Threshold", f"{threshold:.2f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)

                    if faces_detected < (frame_count // 8) * 0.5:
                        st.warning("⚠️ Low face detection rate. Try a video with clearer faces.")

                    # Visualization
                    df = pd.DataFrame(frame_scores)
                    
                    tab1, tab2 = st.tabs(["Frame Analysis", "Download Data"])
                    
                    with tab1:
                        st.markdown("### Fake Score Over Time")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(df['Time (s)'], df['FAKE_Score'], color='#ef4444', linewidth=2, label='Fake Score')
                        ax.axhline(y=threshold, color='#3b82f6', linestyle='--', label=f"Threshold ({threshold})")
                        ax.fill_between(df['Time (s)'], df['FAKE_Score'], threshold, 
                                      where=(df['FAKE_Score'] > threshold), 
                                      color='#ef4444', alpha=0.3, interpolate=True)
                        ax.fill_between(df['Time (s)'], df['FAKE_Score'], threshold, 
                                      where=(df['FAKE_Score'] <= threshold), 
                                      color='#10b981', alpha=0.3, interpolate=True)
                        ax.set_xlabel("Time (seconds)")
                        ax.set_ylabel("Fake Score")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                        # Histogram
                        st.markdown("### Distribution of Fake Scores")
                        fig2, ax2 = plt.subplots(figsize=(10, 4))
                        ax2.hist(df['FAKE_Score'], bins=20, color='#ef4444', edgecolor='#0f172a', alpha=0.7)
                        ax2.axvline(x=threshold, color='#3b82f6', linestyle='--', label=f"Threshold ({threshold})")
                        ax2.set_xlabel("Fake Score")
                        ax2.set_ylabel("Frequency")
                        ax2.legend()
                        ax2.grid(True, alpha=0.3)
                        st.pyplot(fig2)
                    
                    with tab2:
                        st.dataframe(df.sort_values('FAKE_Score', ascending=False).head(10))
                        st.download_button(
                            label="📥 Download Full Analysis Data",
                            data=df.to_csv(index=False),
                            file_name="deepfake_analysis_results.csv",
                            mime="text/csv"
                        )

                else:
                    st.warning("⚠️ No faces found. Use a video with clear frontal faces.")

                st.success("✅ Video processing completed!")
                status_text.empty()

    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <div class="footer">
        <div>
            Developed by <strong>Sunil Sowrirajan</strong> &copy; 2025 <span id="year"></span> |
            <a href="linkedin.com/in/sunil-sowrirajan-40548826b" target="_blank" title="LinkedIn">
                <i class="fab fa-linkedin social-icon"></i>
            </a>
            <a href="https://github.com/suniltechs" target="_blank" title="GitHub">
                <i class="fab fa-github social-icon"></i>
            </a>
        </div>
    </div>
    <script>
        document.getElementById("year").textContent = new Date().getFullYear();
    </script>
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    main()
