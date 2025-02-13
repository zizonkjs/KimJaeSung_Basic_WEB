#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cgi
import cgitb
import joblib
import torch
import sys
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords

# CGI 오류 및 디버깅 활성화
cgitb.enable()

# 표준 출력의 인코딩을 UTF-8로 변환
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# 텍스트 전처리 함수 정의
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# SentimentClassifier 클래스 정의 (BatchNorm1d 포함)
class SentimentClassifier(torch.nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(1000, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)  # BatchNorm1d 추가
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)  # BatchNorm1d 추가
        self.fc3 = torch.nn.Linear(256, 128)
        self.bn3 = torch.nn.BatchNorm1d(128)  # BatchNorm1d 추가
        self.fc4 = torch.nn.Linear(128, 64)
        self.bn4 = torch.nn.BatchNorm1d(64)   # BatchNorm1d 추가
        self.fc5 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)  # 배치 정규화 적용
        x = self.relu(self.fc2(x))
        x = self.bn2(x)  # 배치 정규화 적용
        x = self.relu(self.fc3(x))
        x = self.bn3(x)  # 배치 정규화 적용
        x = self.relu(self.fc4(x))
        x = self.bn4(x)  # 배치 정규화 적용
        x = self.sigmoid(self.fc5(x))
        return x

# 모델 불러오기 (오류 핸들링 포함)
try:
    model = SentimentClassifier()
    model.load_state_dict(torch.load(r"C:\Users\kimjaesung\10.Basic_WEb\DAY_04\test\cgi-bin\model_weights.pth"))
    model.eval()
except Exception as e:
    print("Content-Type: text/html; charset=utf-8")
    print()
    print(f"<html><body><h1>Model loading error: {str(e)}</h1></body></html>")
    sys.exit(1)

# 벡터화 도구 불러오기 (오류 핸들링 포함)
try:
    tfidf_vectorizer = joblib.load(r"C:\Users\kimjaesung\10.Basic_WEb\DAY_04\test\cgi-bin\tfidf_vectorizer.pkl")
except Exception as e:
    print("Content-Type: text/html; charset=utf-8")
    print()
    print(f"<html><body><h1>Vectorizer loading error: {str(e)}</h1></body></html>")
    sys.exit(1)

# 감정 예측 함수 정의
def predict_sentiment(model, text, vectorizer, threshold=0.69):
    input_vector = vectorizer.transform([text]).toarray()
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output)
        predicted_label = (prediction >= threshold).item()
    return 'Positive' if predicted_label == 1 else 'Negative'

# HTML 페이지 출력 함수
def show_html(input_text="", result=""):
    print("Content-Type: text/html; charset=utf-8")
    print()
    print(f"""
    <html>
    <head><title>Sentiment Analysis</title></head>
    <body>
        <h1>Sentiment Analysis</h1>
        <form method="post">
            <textarea name="text" rows="10" cols="40">{input_text}</textarea>
            <p><input type="submit" value="Analyze"></p>
        </form>
        <p>Prediction: {result}</p>
    </body>
    </html>
    """)

# CGI 메인 함수
def main():
    form = cgi.FieldStorage()
    user_text = form.getfirst("text", "")
    
    if user_text:
        processed_text = preprocess_text(user_text)
        result = predict_sentiment(model, processed_text, tfidf_vectorizer)
        show_html(user_text, result)
    else:
        show_html()

if __name__ == "__main__":
    main()




