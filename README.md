ADEGuard: AI-Powered Adverse Drug Event DetectionAuthor: Mohit DhawaneStatus: Completed🚀 Live Interactive DemosExperience the project results through these live, interactive web applications:View the Interactive Project Hub »View the Interactive Evaluation Report »(Note: You will need to replace "your-username.github.io/your-repo-name" with your actual GitHub Pages URL after deploying.)1. Project OverviewADEGuard is an end-to-end MLOps pipeline designed to transform unstructured, noisy VAERS (Vaccine Adverse Event Reporting System) symptom notes into clean, structured, and actionable insights. The core of the system is a fine-tuned BioBERT model that achieves an F1-Score of over 97% in identifying ADE (Adverse Drug Event) and DRUG entities from free-text narratives.2. System ArchitectureThe project follows a modern, modular pipeline designed for scalability and maintainability:Data Refinery (LangGraph): A stateful workflow ingests raw data, performs cleaning and pre-processing, and uses semantic chunking to isolate individual adverse events.Programmatic Labeling (LLM): A Large Language Model, guided by a sophisticated prompt with a "clinical expert" persona, is used to programmatically generate a large, high-quality "Silver Data" training set.Model Training (Hugging Face): A pre-trained BioBERT model is fine-tuned on the "Silver Data" to create a specialist NER model for the biomedical domain.API Backend (FastAPI): The trained model is served via a high-performance, asynchronous REST API, making it available as a reusable service.UI Frontend (Streamlit): An interactive web dashboard allows non-technical users to input symptom text and receive real-time entity extraction results.3. Technology StackCore Language: PythonML / NLP: Hugging Face Transformers, BioBERT, Sentence-Transformers, SeqevalData Processing: Pandas, LangGraphBackend: FastAPI, UvicornFrontend: StreamlitAnnotation: Label Studio (for Gold Data), Programmatic Labeling (for Silver Data)4. How to Run This ProjectFollow these steps to set up and run the full ADEGuard pipeline on your local machine.Step 1: Clone the Repositorygit clone [https://github.com/MohitDhawane/ADEGuard-AI-Powered-Adverse-Drug-Event-Detection](https://github.com/MohitDhawane/ADEGuard-AI-Powered-Adverse-Drug-Event-Detection)
cd your-repository-name
Step 2: Set Up a Virtual Environment & Install Dependencies# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt
Step 3: Run the Full PipelineExecute the scripts in the following order to process the data and train the model:
# 1. Consolidate and filter raw data
python data_processing.py

# 2. Perform basic cleaning
python data_cleaning.py

# 3. Generate refined text chunks (this can take a while)
python langgraph_refinery.py

# 4. Process LLM output to create silver_data.json


# 5. Transform the silver data for training
python transform_ner_data.py

# 7. Train the NER model
python train_ner_model.py
Step 4: Launch the ApplicationYou need two separate terminals to run the backend and frontend.Terminal 1: Start the FastAPI Backenduvicorn api_backend:app --reload
The API will be available at http://127.0.0.1:8000.Terminal 2: Start the Streamlit Frontendstreamlit run streamlit_app.py



