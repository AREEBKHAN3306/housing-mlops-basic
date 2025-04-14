import pandas as pd
import numpy as np
import gradio as gr
import pickle

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define prediction function
def predict_price(area, bedrooms, bathrooms):
    input_data = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)
    return f"Predicted House Price: ${prediction[0]:,.2f}"

# Define Gradio Interface
inputs = [
    gr.Number(label="Area (sqft)"),
    gr.Number(label="Bedrooms"),
    gr.Number(label="Bathrooms")
]

iface = gr.Interface(
    fn=predict_price,
    inputs=inputs,
    outputs="text",
    title="Housing Price Prediction App",
    description="Enter house details to predict the price."
)

iface.launch()
