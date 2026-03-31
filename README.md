# ✏️ Handwritten Digit Recognition

A deep learning web app that recognizes handwritten digits (0-9) using a Convolutional Neural Network trained on the MNIST dataset.

## Features

- **Draw to Predict** — Interactive canvas to draw digits directly in the browser
- **Upload to Predict** — Upload an image of a handwritten digit
- **Real-time CNN** — Model trains on MNIST at startup and caches for fast inference
- **Confidence Visualization** — See top-5 prediction probabilities

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Dataset | MNIST (60K training, 10K test) |
| Drawing | streamlit-drawable-canvas |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run HDR.py
```

## Live Demo

Deployed on [Streamlit Community Cloud](https://share.streamlit.io/).

## License

MIT
