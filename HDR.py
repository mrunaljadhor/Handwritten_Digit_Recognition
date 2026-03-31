import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# --- Configuration ---
IMG_SIZE = 28

@st.cache_resource
def build_and_train_model():
    """Build, train, and return the CNN model on MNIST."""
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3, validation_split=0.1, verbose=0)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    return model, test_acc

def preprocess_canvas_image(canvas_result):
    """Convert canvas drawing to model-ready input."""
    if canvas_result.image_data is not None:
        # Get the image from canvas (RGBA)
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        # Convert to grayscale
        img = img.convert('L')
        # Resize to 28x28
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        # Invert (canvas has white drawing on black, MNIST is white on black)
        img = ImageOps.invert(img)
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array, img
    return None, None

def preprocess_uploaded_image(uploaded_file):
    """Convert uploaded image to model-ready input."""
    img = Image.open(uploaded_file)
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = ImageOps.invert(img)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array, img

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
        transition: width 0.5s ease;
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
    </style>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### ⚙️ About")
        st.markdown("""
        **Handwritten Digit Recognition** uses a 
        Convolutional Neural Network (CNN) trained 
        on the MNIST dataset to recognize digits 0-9.

        **Tech Stack:**
        - 🧠 TensorFlow / Keras CNN
        - 📊 MNIST Dataset (60K training images)
        - ✏️ Interactive Drawing Canvas
        - 📤 Image Upload Support
        """)
        st.divider()
        st.markdown("### 💡 Tips")
        st.markdown("""
        - **Draw thick digits** in the center of the canvas
        - Use the **full canvas area** for better results
        - **Clear** the canvas before drawing a new digit
        - Upload a **clear, high-contrast** image for best accuracy
        """)
        st.divider()
        st.caption("Built with ❤️ using Streamlit & TensorFlow")

    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>✏️ Handwritten Digit Recognition</h1>
        <p>Draw or upload a digit (0-9) and let the CNN predict it</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("🧠 Training CNN model on MNIST — this may take a minute on first run..."):
        model, test_acc = build_and_train_model()

    st.markdown(
        f'<span class="status-badge">✓ Model Ready</span>'
        f'<span class="accuracy-badge">Test Accuracy: {test_acc:.1%}</span>',
        unsafe_allow_html=True
    )
    st.write("")

    # --- Input method ---
    tab1, tab2 = st.tabs(["✏️ Draw a Digit", "📤 Upload an Image"])

    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("#### Draw your digit below:")
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )

            if st.button("🔍 Predict", key="predict_draw", use_container_width=True):
                if canvas_result.image_data is not None:
                    img_array, processed_img = preprocess_canvas_image(canvas_result)
                    if img_array is not None:
                        predictions = model.predict(img_array, verbose=0)
                        predicted_digit = np.argmax(predictions)
                        confidence = predictions[0][predicted_digit] * 100

                        with col2:
                            st.markdown("#### Prediction Result")
                            st.markdown(f"""
                            <div class="prediction-card">
                                <div class="prediction-label">Predicted Digit</div>
                                <div class="prediction-digit">{predicted_digit}</div>
                                <div style="color: #94a3b8; margin-top: 0.5rem;">
                                    Confidence: <strong style="color: #ffd200;">{confidence:.1f}%</strong>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Show top 3 predictions
                            st.markdown("#### 📊 Top Predictions")
                            top_indices = np.argsort(predictions[0])[::-1][:5]
                            for i, idx in enumerate(top_indices):
                                prob = predictions[0][idx] * 100
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
                else:
                    st.warning("Please draw a digit on the canvas first!")

    with tab2:
        col3, col4 = st.columns([1, 1], gap="large")

        with col3:
            st.markdown("#### Upload a digit image:")
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["png", "jpg", "jpeg", "bmp"],
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                # Show uploaded image
                st.image(uploaded_file, caption="Uploaded Image", width=280)

                if st.button("🔍 Predict", key="predict_upload", use_container_width=True):
                    img_array, processed_img = preprocess_uploaded_image(uploaded_file)
                    predictions = model.predict(img_array, verbose=0)
                    predicted_digit = np.argmax(predictions)
                    confidence = predictions[0][predicted_digit] * 100

                    with col4:
                        st.markdown("#### Prediction Result")
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-label">Predicted Digit</div>
                            <div class="prediction-digit">{predicted_digit}</div>
                            <div style="color: #94a3b8; margin-top: 0.5rem;">
                                Confidence: <strong style="color: #ffd200;">{confidence:.1f}%</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Show top 3 predictions
                        st.markdown("#### 📊 Top Predictions")
                        top_indices = np.argsort(predictions[0])[::-1][:5]
                        for i, idx in enumerate(top_indices):
                            prob = predictions[0][idx] * 100
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