import joblib
import warnings
warnings.filterwarnings("ignore")

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
emotion_number = joblib.load("emotion_mapping.pkl")

emotion_names = {v: k for k, v in emotion_number.items()}

text=input("Enter a text to predict the emotion: ")

prediction = model.predict(vectorizer.transform([text]))

print(f"The predicted emotion is: {emotion_names[prediction[0]]}")