import streamlit as st
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/extract-entities"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ADEGuard",
    page_icon="🛡️",
    layout="centered"
)

# --- Main UI Components ---
st.title("🛡️ ADEGuard: Adverse Drug Event Analyzer")
st.markdown("Enter a symptom narrative below to extract potential Adverse Drug Events (ADEs) and DRUG names.")

# Text area for user input
symptom_text = st.text_area(
    "Symptom Narrative",
    height=150,
    placeholder="e.g., The patient reported a severe headache and nausea approximately 2 hours after receiving the Pfizer vaccine."
)

# Button to trigger analysis
if st.button("Analyze Text"):
    if not symptom_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            # --- API Call ---
            with st.spinner("Analyzing..."):
                payload = {"text": symptom_text}
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)

                results = response.json()
                entities = results.get("entities", [])

            # --- Display Results ---
            st.success("Analysis Complete!")

            if entities:
                st.subheader("Extracted Entities:")
                
                # Create a visually appealing display for entities
                ade_color = "#FFB3B3" # Light red
                drug_color = "#B3D9FF" # Light blue

                for entity in entities:
                    entity_type = entity['type']
                    color = ade_color if entity_type == 'ADE' else drug_color
                    
                    # Using columns for better layout
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Entity:** `{entity['word']}`")
                    with col2:
                        st.markdown(f"<span style='background-color:{color}; padding: 2px 6px; border-radius: 5px;'>{entity_type}</span>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"**Confidence:** {entity['confidence']:.2%}")
                    st.divider()
            else:
                st.info("No entities were found in the provided text.")

        except requests.exceptions.RequestException as e:
            logging.error(f"API connection error: {e}")
            st.error("Could not connect to the API backend. Please ensure the FastAPI server is running.")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            st.error("An unexpected error occurred during analysis.")

# --- Instructions to Run ---
st.markdown("---")
st.info("""
    **How to run this application:**
    1.  **Start the Backend:** In one terminal, run `uvicorn api_backend:app --reload`.
    2.  **Start the Frontend:** In a second terminal, run `streamlit run streamlit_app.py`.
""")
