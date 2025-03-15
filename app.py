
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(resume_text, job_description):
    """
    Compute the cosine similarity between a resume and a job description.
    """
    resume_embedding = model.encode([resume_text])
    job_embedding = model.encode([job_description])
    similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
    return similarity

def rank_resumes(resumes_df, job_description):
    """
    Rank resumes based on their similarity to a job description.
    """
    ranked_resumes = []
    for _, row in resumes_df.iterrows():
        resume_id = row["ID"]
        resume_text = row["content"]
        similarity_score = compute_similarity(resume_text, job_description)
        ranked_resumes.append((resume_id, resume_text, similarity_score))
    ranked_resumes.sort(key=lambda x: x[2], reverse=True)
    return ranked_resumes

def validate_csv(file):
    """
    Validate the uploaded CSV file to ensure it has the required columns (ID and content).
    """
    try:
        # Read the uploaded CSV file
        resumes_df = pd.read_csv(file.name)  # Use file.name to get the file path
        
        # Check if the required columns exist
        if "ID" not in resumes_df.columns or "content" not in resumes_df.columns:
            return False, "The CSV file must contain 'ID' and 'content' columns."
        
        return True, resumes_df
    except Exception as e:
        return False, f"Error reading CSV file: {str(e)}"

def main(csv_file, job_description):
    """
    Main function to process uploaded CSV and rank resumes.
    """
    if csv_file is None:
        return "Please upload a CSV file."

    # Validate the CSV file
    is_valid, validation_result = validate_csv(csv_file)
    if not is_valid:
        return validation_result  # Return the error message

    # If the file is valid, rank the resumes
    resumes_df = validation_result
    ranked_resumes = rank_resumes(resumes_df, job_description)
    return "\n".join([f"Resume ID: {resume_id} | Resume: {resume_text[:50]}... | Similarity Score: {score:.2f}" 
                      for resume_id, resume_text, score in ranked_resumes])

iface = gr.Interface(
    fn=main,
    inputs=[
        gr.File(label="Upload Resumes CSV (ID, content)"), 
        gr.Textbox(lines=5, placeholder="Enter job description...", label="Job Description")
    ],
    outputs="text",
    title="Resume Screening System",
    description="Upload a CSV file with resumes (ID, content) and enter a job description to rank resumes based on similarity."
)

iface.launch()
