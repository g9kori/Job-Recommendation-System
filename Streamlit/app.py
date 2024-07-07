import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel

# Load the model components
tfidf_vectorizer = joblib.load('Streamlit/tfidf_vectorizer.pkl')
jobs_train = pd.read_csv('Streamlit/jobs_train.csv')
tfidf_matrix_train = tfidf_vectorizer.transform(jobs_train['skills'])

# Extract all unique skills from the jobs dataset
all_skills = set()
for skills in jobs_train['skills']:
    for skill in skills.replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace("[", "").replace("]", "").split(','):
        all_skills.add(skill.strip().lower())

# Function to get recommendations
def get_recommendations(skills, tfidf_matrix, tfidf_vectorizer, jobs_df, top_n=4):
    skills_tfidf = tfidf_vectorizer.transform([skills])
    cosine_similarities = linear_kernel(skills_tfidf, tfidf_matrix).flatten()
    related_job_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return jobs_df.iloc[related_job_indices]

# Function to check for invalid skills
def check_invalid_skills(input_skills, all_skills):
    invalid_skills = []
    for skill in input_skills:
        # Remove leading and trailing spaces, and convert to lowercase
        clean_skill = skill.strip().lower()
        if clean_skill not in all_skills:
            invalid_skills.append(skill.strip())
    return invalid_skills

# Function to check for invalid skills considering phrases
def check_invalid_skills(input_skills, all_skills):
    invalid_skills = []
    for skill in input_skills:
        # Remove leading and trailing spaces, and convert to lowercase
        clean_skill = skill.strip().lower()
        
        # Check if the exact phrase is in the all_skills set
        if clean_skill not in all_skills:
            # Check if any part of the skill phrase exists in all_skills
            words = clean_skill.split()
            if not any(word in all_skills for word in words):
                invalid_skills.append(skill.strip())
    return invalid_skills

# Streamlit app
st.title("Job Recommendation System")

st.write("Enter your skills to get job recommendations:")

# User input for skills
skills_input = st.text_area("Skills")

if st.button('Recommend Jobs'):
    if skills_input:
        # Split by commas and remove extra spaces around commas
        input_skills = [skill.strip() for skill in skills_input.split(',')]
        invalid_skills = check_invalid_skills(input_skills, all_skills)

        if invalid_skills:
            st.write(f"Error: Invalid skills: {', '.join(invalid_skills)}")
        else:
            # Join valid skills with commas for display and recommendation
            joined_skills = ', '.join(input_skills)
            recommendations = get_recommendations(joined_skills, tfidf_matrix_train, tfidf_vectorizer, jobs_train)
            st.write("### Top job recommendations based on your skills:\n")
            for index, row in recommendations.iterrows():
                st.write(f"#### Job ID: {row['jobId']}")
                st.write(f"**Job Title:** {row['jobTitle']}")
                
                # Clean and display Skills Required
                skills_list = row['skills'].replace("{", "").replace("}", "").replace("'", "").replace('"', "").replace("[", "").replace("]", "").split(',')
                skills_str = ', '.join([skill.strip() for skill in skills_list])
                st.write(f"**Skills Required:** {skills_str}")
                
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Job Type:** {row['jobType']}")
                
                # Clean and display Location
                location_list = row['location'].replace("{", "").replace("}", "").split(',')
                location_str = ', '.join([location.strip() for location in location_list])
                st.write(f"**Location:** {location_str}")
                
                st.write("-" * 40 + "\n")
    else:
        st.write("Please enter your skills to get recommendations.")