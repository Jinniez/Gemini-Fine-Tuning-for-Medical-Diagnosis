import os
import pandas as pd
import google.generativeai as genai
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def initialize_genai_model(api_key, model_name="gemini-1.5-flash"):
    """Initialize and return the generative model."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model

def process_medical_queries(model, test_data):
    """Process medical queries by adding instructions and obtaining model responses."""
    instruction = "Suppose that you are a medical diagnosis assistant, skilled in recognizing diseases and prescribing treatments. If a patient has the following condition, give me the most probable diagnosis and the related probability and urgency and rationale for the diagnosis besides suitable treatments. Start your response with Diagnosis:"
    results = []
    stop_words = set(stopwords.words('english'))

    for index, row in test_data.iterrows():
        query = f"{instruction} {row['Patient Conditions']}"
        response = model.generate_content(query)
        texts = [part.text for part in response.parts]
        first_diagnosis = " ".join(texts).split('.')[0]

        correct_keywords = set(row['Correct Diagnosis'].lower().split()) - stop_words
        output_keywords = set(first_diagnosis.lower().split()) - stop_words
        correct_count = len(correct_keywords & output_keywords)
        percentage_correct = (correct_count / len(correct_keywords)) * 100 if correct_keywords else 0
        is_correct = percentage_correct >= 50

        results.append((index, row['Correct Diagnosis'], first_diagnosis, is_correct))
        print(f"Processed {index + 1}/{len(test_data)} cases.")

    return results

def calculate_accuracy(results):
    """Calculate the accuracy of the diagnosis predictions."""
    total_cases = len(results)
    correct_cases = sum(1 for _, _, _, is_correct in results if is_correct)
    accuracy = (correct_cases / total_cases) * 100
    return accuracy

def read_test_data(file_path, sheet_name=None):
    """Read test data from an Excel file, optionally specifying a sheet."""
    return pd.read_excel(file_path, sheet_name=sheet_name)

# API key comes from the environment — never hardcode it.
# Set it first: export GEMINI_API_KEY="your-key-here"
api_key = os.environ.get("GEMINI_API_KEY")
model_name = "gemini-1.5-flash"

# Initialize the model
model = initialize_genai_model(api_key, model_name)

# Read Test Data from Excel
file_path = 'Test_Data.xlsx'
sheet_name = 'NewSamples'  # Specify the sheet name here
test_data = read_test_data(file_path, sheet_name)

# Process the test data with the model
diagnoses_results = process_medical_queries(model, test_data)

# Calculate and print the accuracy
accuracy = calculate_accuracy(diagnoses_results)
print(f"Accuracy of the diagnoses: {accuracy:.2f}%")
