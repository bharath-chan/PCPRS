import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
@st.cache
def load_data():
    data = pd.read_csv('dataforanalysis_final.csv')  # Replace 'your_data.csv' with your actual file path
    return data

data = load_data()

# Data preprocessing
# Handle missing values
data.fillna('No data', inplace=True)

# Combine relevant features into one text column for each course
data['course_features'] = data['Course1'] + ' ' + data['Course2'] + ' ' + data['Course3']

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['course_features'])

# Streamlit web application
st.title('Course Recommendation System')

gender = st.selectbox('Gender', data['Gender'].unique())
father_occupation = st.selectbox('Father\'s Occupation', data['Father_occupation'].unique())
mother_occupation = st.selectbox('Mother\'s Occupation', data['Mother_occupation'].unique())
intermediate_percentage = st.selectbox('Intermediate Percentage', data['Inter'].unique())
eamcet_rank = st.selectbox('EAMCET Rank', data['Rank'].unique())
branch = st.selectbox('Branch', data['branch'].unique())
goal = st.selectbox('Goal', data['goal'].unique())

input_features = [gender, father_occupation, mother_occupation, intermediate_percentage, eamcet_rank, branch, goal]

if st.button('Recommend Courses'):
    input_text = ' '.join(input_features)
    input_tfidf = tfidf_vectorizer.transform([input_text])

    # Compute cosine similarity between input and all courses
    similarities = cosine_similarity(input_tfidf, tfidf_matrix)

    # Get indices of top recommended courses
    top_indices = similarities.argsort()[0][-3:][::-1]

    st.write('Recommended Courses:')
    for i, idx in enumerate(top_indices, start=1):
        st.write(f'{i}. {data.iloc[idx]["Course1"]}, {data.iloc[idx]["Course2"]}, {data.iloc[idx]["Course3"]}')
