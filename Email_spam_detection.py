import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv('spam.csv')

# Split the data into features (X) and labels (y)
X = data['text']
y = data['label']

# Convert text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X, y)

# Function to predict if an email is spam or not
def predict_email(text):
    # Preprocess the input text
    text_features = vectorizer.transform([text])
    
    # Predict the label
    prediction = model.predict(text_features)[0]
    
    # Return the result
    return "spam" if prediction == 'spam' else "ham"

# Test the function with a sample text
sample_text = "Congratulations, you've won"
prediction = predict_email(sample_text)
print(f"Prediction for '{sample_text}': {prediction}")


