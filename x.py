import streamlit as st
import pickle
import re

# Load the pre-trained model (e.g., SVM, Logistic Regression, etc.)
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TfidfVectorizer that was used to fit the model
with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Define a custom list of stopwords
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now'
])

# Function to clean and preprocess text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphabetical characters using regex
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    cleaned_tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(cleaned_tokens)

# Define the Streamlit application layout
st.title('Mental Health Issue Prediction')

# Create a text input box for user to type in
user_input = st.text_area('Enter Text Here:')

# Prediction functionality
if st.button('Predict'):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess_text(user_input)
        
        # Vectorize the text using the pre-fitted TfidfVectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Make the prediction using the model
        prediction = model.predict(text_vector)
        
        # Display the predicted mental health issue
        if isinstance(prediction[0], str):
            # If prediction is a string label
            st.write(f"Prediction: {prediction[0]}")
        else:
            # If prediction is an integer index
            mental_health_labels = ['Anxiety', 'Depression', 'PTSD & Trauma', 'Suicidal thoughts and self-harm']
            predicted_label = mental_health_labels[prediction[0]]
            st.write(f"Prediction: {predicted_label}")
    else:
        st.write("Please enter some text to predict.")
