import numpy as np
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import io
import base64
from streamlit.components.v1 import html

# --- Configuration ---
IMG_SIZE = 28

@st.cache_resource
def build_and_train_model():
    """Build, train, and return the Random Forest model on MNIST."""
    # Load MNIST from sklearn
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)

    # Normalize
    X = X / 255.0

    # Use a subset for faster training on cloud (20K samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=30,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    test_acc = model.score(X_test, y_test)
    return model, test_acc

def preprocess_uploaded_image(uploaded_file):
    """Convert uploaded image to model-ready input."""
    img = Image.open(uploaded_file)
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = ImageOps.invert(img)
    img_array = np.array(img).astype(np.float64) / 255.0
    img_flat = img_array.reshape(1, -1)  # Flatten to (1, 784)
    return img_flat, img

def preprocess_canvas_data(canvas_data_url):
    """Convert canvas base64 image data to model-ready input."""
    # Remove the data URL prefix
    header, encoded = canvas_data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = ImageOps.invert(img)
    img_array = np.array(img).astype(np.float64) / 255.0
    img_flat = img_array.reshape(1, -1)
    return img_flat, img

def get_canvas_html():
    """Return HTML/JS for an embedded drawing canvas."""
    return """
    <style>
        .canvas-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
        }
        #drawCanvas {
            border: 2px solid #ffd200;
            border-radius: 12px;
            cursor: crosshair;
            background: #000;
            touch-action: none;
        }
        .canvas-buttons {
            display: flex;
            gap: 10px;
        }
        .canvas-btn {
            padding: 8px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-clear {
            background: #3a3a5c;
            color: #e2e8f0;
        }
        .btn-clear:hover { background: #4a4a6c; }
        .btn-predict {
            background: linear-gradient(135deg, #f7971e, #ffd200);
            color: #1a1a2e;
        }
        .btn-predict:hover { transform: scale(1.05); }
    </style>
    <div class="canvas-container">
        <canvas id="drawCanvas" width="280" height="280"></canvas>
        <div class="canvas-buttons">
            <button class="canvas-btn btn-clear" onclick="clearCanvas()">🗑️ Clear</button>
            <button class="canvas-btn btn-predict" onclick="sendCanvas()">🔍 Predict</button>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('drawCanvas');
        const ctx = canvas.getContext('2d');
        let drawing = false;

        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return {
                x: (clientX - rect.left) * (canvas.width / rect.width),
                y: (clientY - rect.top) * (canvas.height / rect.height)
            };
        }

        canvas.addEventListener('mousedown', (e) => { drawing = true; ctx.beginPath(); const p = getPos(e); ctx.moveTo(p.x, p.y); });
        canvas.addEventListener('mousemove', (e) => { if (!drawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
        canvas.addEventListener('mouseup', () => { drawing = false; });
        canvas.addEventListener('mouseleave', () => { drawing = false; });

        canvas.addEventListener('touchstart', (e) => { e.preventDefault(); drawing = true; ctx.beginPath(); const p = getPos(e); ctx.moveTo(p.x, p.y); });
        canvas.addEventListener('touchmove', (e) => { e.preventDefault(); if (!drawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
        canvas.addEventListener('touchend', (e) => { e.preventDefault(); drawing = false; });

        function clearCanvas() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }

        function sendCanvas() {
            const dataURL = canvas.toDataURL('image/png');
            // Send to Streamlit via query params trick
            const encoded = encodeURIComponent(dataURL);
            window.parent.postMessage({type: 'streamlit:setComponentValue', value: dataURL}, '*');

            // Also store in a hidden input for form submission
            const existing = window.parent.document.querySelectorAll('input[data-testid="stTextInput"]');
        }
    </script>
    """

def main():
    st.set_page_config(
        page_title="Handwritten Digit Recognition",
        page_icon="✏️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Custom CSS ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: #1a1a2e;
        box-shadow: 0 8px 32px rgba(247, 151, 30, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.8;
        font-size: 1.05rem;
    }

    .prediction-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2a2a3d 100%);
        border-left: 4px solid #ffd200;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        text-align: center;
    }

    .prediction-digit {
        font-size: 5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }

    .prediction-label {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }

    .confidence-bar-container {
        background: #1e1e2e;
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.4rem 0;
    }

    .confidence-label {
        color: #e2e8f0;
        font-size: 0.9rem;
        font-weight: 500;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }

    .confidence-bar-bg {
        background: #2a2a3d;
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
    }

    .status-badge {
        display: inline-block;
        background: linear-gradient(135deg, #34d399, #059669);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .accuracy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }

    .sample-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ About")
        st.markdown("""
        **Handwritten Digit Recognition** uses a
        Random Forest classifier trained on the
        MNIST dataset to recognize digits 0-9.

        **Tech Stack:**
        - 🧠 scikit-learn Random Forest
        - 📊 MNIST Dataset (70K images)
        - 📤 Image Upload Support
        - 🎨 Built-in Drawing Canvas
        """)
        st.divider()
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        - Draw **thick, centered** digits
        - Use a **white digit on dark background**
        - Upload **clear, high-contrast** images
        - **Square images** work best
        """)
        st.divider()
        st.caption("Built with ❤️ using Streamlit & scikit-learn")

    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>✏️ Handwritten Digit Recognition</h1>
        <p>Upload an image of a digit (0-9) and let the ML model predict it</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("🧠 Training model on MNIST — this may take a minute on first run..."):
        model, test_acc = build_and_train_model()

    st.markdown(
        f'<span class="status-badge">✓ Model Ready</span>'
        f'<span class="accuracy-badge">Test Accuracy: {test_acc:.1%}</span>',
        unsafe_allow_html=True
    )
    st.write("")

    # --- Prediction display function ---
    def show_prediction(probabilities, col):
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[predicted_digit] * 100

        with col:
            st.markdown("#### 🎯 Prediction Result")
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted Digit</div>
                <div class="prediction-digit">{predicted_digit}</div>
                <div style="color: #94a3b8; margin-top: 0.5rem;">
                    Confidence: <strong style="color: #ffd200;">{confidence:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 All Predictions")
            top_indices = np.argsort(probabilities)[::-1]
            for i, idx in enumerate(top_indices):
                prob = probabilities[idx] * 100
                if prob < 0.1:
                    continue
                if i == 0:
                    bar_color = "linear-gradient(90deg, #f7971e, #ffd200)"
                elif i == 1:
                    bar_color = "linear-gradient(90deg, #667eea, #764ba2)"
                else:
                    bar_color = "linear-gradient(90deg, #4a5568, #718096)"
                st.markdown(f"""
                <div class="confidence-bar-container">
                    <div class="confidence-label">
                        <span>Digit {idx}</span>
                        <span>{prob:.1f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: {prob}%; background: {bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # --- Upload Image ---
    st.markdown("### 📤 Upload a Digit Image")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image of a handwritten digit...",
            type=["png", "jpg", "jpeg", "bmp"],
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", width=280)

            if st.button("🔍 Predict Digit", key="predict_upload", use_container_width=True):
                img_flat, processed_img = preprocess_uploaded_image(uploaded_file)
                probabilities = model.predict_proba(img_flat)[0]
                show_prediction(probabilities, col2)

    # --- Draw Section ---
    st.markdown("---")
    st.markdown("### ✏️ Or Draw a Digit")
    st.info("💡 Draw a digit in the black box below using your mouse, then click **Predict**.")

    # Use Streamlit's built-in canvas approach
    col3, col4 = st.columns([1, 1], gap="large")

    with col3:
        # Embedded HTML Canvas
        html(get_canvas_html(), height=350)

        st.markdown("")
        st.markdown("**After drawing**, save the canvas as an image and upload it above, or use the sample images below.")

    # --- Sample Test Images ---
    st.markdown("---")
    st.markdown("### 🧪 Quick Test with Sample Digits")
    st.markdown("Click a button below to test the model with a sample from the MNIST dataset:")

    sample_cols = st.columns(5)

    # Generate sample digits from the trained model's data
    @st.cache_data
    def get_sample_digits():
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        samples = {}
        for digit in range(10):
            idx = np.where(y == digit)[0][0]
            samples[digit] = X[idx]
        return samples

    samples = get_sample_digits()

    # First row: digits 0-4
    for i, col in enumerate(sample_cols):
        with col:
            digit = i
            img_array = samples[digit].reshape(28, 28)
            img = Image.fromarray((img_array).astype(np.uint8), mode='L')
            st.image(img, caption=f"Digit {digit}", width=100)
            if st.button(f"Test {digit}", key=f"sample_{digit}", use_container_width=True):
                img_flat = samples[digit].reshape(1, -1) / 255.0
                probabilities = model.predict_proba(img_flat)[0]
                st.session_state['last_prediction'] = (probabilities, digit)

    # Second row: digits 5-9
    sample_cols2 = st.columns(5)
    for i, col in enumerate(sample_cols2):
        with col:
            digit = i + 5
            img_array = samples[digit].reshape(28, 28)
            img = Image.fromarray((img_array).astype(np.uint8), mode='L')
            st.image(img, caption=f"Digit {digit}", width=100)
            if st.button(f"Test {digit}", key=f"sample_{digit}", use_container_width=True):
                img_flat = samples[digit].reshape(1, -1) / 255.0
                probabilities = model.predict_proba(img_flat)[0]
                st.session_state['last_prediction'] = (probabilities, digit)

    # Show sample prediction result
    if 'last_prediction' in st.session_state:
        probabilities, true_digit = st.session_state['last_prediction']
        predicted = np.argmax(probabilities)
        confidence = probabilities[predicted] * 100

        st.markdown("---")
        st.markdown("### 🎯 Sample Prediction Result")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted Digit</div>
                <div class="prediction-digit">{predicted}</div>
                <div style="color: #94a3b8; margin-top: 0.5rem;">
                    Confidence: <strong style="color: #ffd200;">{confidence:.1f}%</strong>
                </div>
                <div style="color: {'#34d399' if predicted == true_digit else '#ef4444'}; margin-top: 0.5rem; font-weight: 600;">
                    {'✓ Correct!' if predicted == true_digit else f'✗ True digit: {true_digit}'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### 📊 Confidence Scores")
            top_indices = np.argsort(probabilities)[::-1]
            for i, idx in enumerate(top_indices):
                prob = probabilities[idx] * 100
                if prob < 0.5:
                    continue
                if i == 0:
                    bar_color = "linear-gradient(90deg, #f7971e, #ffd200)"
                elif i == 1:
                    bar_color = "linear-gradient(90deg, #667eea, #764ba2)"
                else:
                    bar_color = "linear-gradient(90deg, #4a5568, #718096)"
                st.markdown(f"""
                <div class="confidence-bar-container">
                    <div class="confidence-label">
                        <span>Digit {idx}</span>
                        <span>{prob:.1f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: {prob}%; background: {bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()