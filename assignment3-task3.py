import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Datasets/train.csv')

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    processed_text = ' '.join(filtered_tokens)
    return processed_text

df['processed_text'] = df['text'].apply(preprocess_text)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Define a function to get the most similar question-answer pair
def get_most_similar_question(query):
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    similarity_scores = cosine_similarity(query_vector, X)
    most_similar_index = similarity_scores.argmax()
    answer = df.loc[most_similar_index, 'text']
    return answer

# Define a set of predefined questions
predefined_questions = [
    "What should I do in case of a flood?",
    "How to prepare for a wildfire?",
    "What are the symptoms of a heat stroke?",
    "How to stay safe during an earthquake?",
    "What are the signs of a tornado approaching?",
    "How to report a forest fire?",
    "What should I do in case of a car crash?",
    "How to protect myself from a hurricane?",
    "What are the precautions to take during a thunderstorm?",
    "How to perform CPR?"
]

# Initialize variables
questions = predefined_questions.copy()
choice = ''

# Run the question-answering bot until the user ends the program
while choice.lower() != 'q':
    if choice.lower() == 'r':
        questions = predefined_questions.copy()
    elif choice.lower() == 'm':
        questions.extend(input("Enter more questions (separated by commas): ").split(','))
    
    print("Choose a question number (1-10), C for a custom question, M for more questions, R to reset, or Q to quit:")
    for i, question in enumerate(questions, start=1):
        print(f"{i}. {question}")
    
    choice = input("Enter your choice: ")
    
    # Check if the user wants to quit
    if choice.lower() == 'q':
        break
    
    # Check if the user wants to reset
    if choice.lower() == 'r':
        continue
    
    # Check if the user wants to add more questions
    if choice.lower() == 'm':
        continue
    
    # Check if the user wants to ask a custom question
    if choice.lower() == 'c':
        custom_question = input("Enter your custom question: ")
        answer = get_most_similar_question(custom_question)
        print("Bot's Answer:", answer)
        continue
    
    # Check if the user entered a valid choice
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(questions):
        print("Invalid choice. Please try again.")
        continue
    
    # Get the chosen question
    chosen_question = questions[int(choice) - 1]
    
    # Get the answer for the chosen question
    answer = get_most_similar_question(chosen_question)
    
    print("Bot's Answer:", answer)
