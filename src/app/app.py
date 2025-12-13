import streamlit as st
import requests
import os

def get_prediction(input_text):
    # URL for the inference endpoint. 
    # Using 'model_inference_endpoint' as hostname for Docker Compose environment.
    # Fallback to localhost for local development.
    url = os.getenv("INFERENCE_ENDPOINT_URL", "http://model_inference_endpoint:8000/get-prediction/")
    
    try:
        response = requests.post(url, json={"input_texts": input_text})
        response.raise_for_status()
        # Assuming the endpoint returns the prediction directly or in a JSON wrapper
        return response.json()
    except requests.exceptions.RequestException as e:
        # If the docker service name fails, try localhost as a fallback (useful for local dev without docker networking setup)
        if "model_inference_endpoint" in url:
             try:
                url = "http://localhost:8000/get-prediction/"
                response = requests.post(url, json={"input_texts": input_text})
                response.raise_for_status()
                return response.json()
             except requests.exceptions.RequestException:
                 pass # Return original error if fallback also fails
        return f"Error: {e}"

# Streamlit page configuration
st.set_page_config(page_title="Tweet Classifier", layout="wide")

# Streamlit UI components
st.title("Classify your tweet")

# User inputs the tweet
tweet_input = st.text_input("Enter your tweet", "")

# Button to trigger prediction
if st.button("Classify Tweet"):
    # Get prediction
    prediction = get_prediction(tweet_input)
    
    # Display the prediction
    st.write("Prediction:", prediction)

