import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel

# Load the model components
tfidf_vectorizer = joblib.load('Streamlit/tfidf_vectorizer.pkl')
jobs_train = pd.read_csv('Streamlit/jobs_train.csv')
tfidf_matrix_train = tfidf_vectorizer.transform(jobs_train['skills'])

# Function to get recommendations
def get_recommendations(skills, tfidf_matrix, tfidf_vectorizer, jobs_df, top_n=4):
    skills_tfidf = tfidf_vectorizer.transform([skills])
    cosine_similarities = linear_kernel(skills_tfidf, tfidf_matrix).flatten()
    related_job_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return jobs_df.iloc[related_job_indices]

# Streamlit app
st.title("Job Recommendation System")

st.write("Enter your skills to get job recommendations:")

# User input for skills
skills_input = st.text_area("Skills")

if st.button('Recommend Jobs'):
    if skills_input:
        recommendations = get_recommendations(skills_input, tfidf_matrix_train, tfidf_vectorizer, jobs_train)
        st.write("Top job recommendations based on your skills:\n")
        for index, row in recommendations.iterrows():
            st.write(f"Job ID: {row['jobId']}")
            st.write(f"Job Title: {row['jobTitle']}")
            
            # Clean and display Skills Required
            skills_list = row['skills'].replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace("[", "").replace("]", "").split(',')
            skills_str = ', '.join([skill.strip() for skill in skills_list])
            st.write(f"Skills Required: {skills_str}")
            
            st.write(f"Description: {row['description']}")
            st.write(f"Job Type: {row['jobType']}")
            
            # Clean and display Location
            location_list = row['location'].replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace("[", "").replace("]", "").split(',')
            location_str = ', '.join([loc.strip() for loc in location_list])
            st.write(f"Location: {location_str}")
            
            st.write("-" * 40 + "\n")
    else:
        st.write("Please enter your skills to get recommendations.")