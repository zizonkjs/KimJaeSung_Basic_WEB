#!/usr/bin/env python3
import cgi
import os
import torch
from PIL import Image
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from Traffic_signs_data1 import CNNModel

# HTTP 헤더 출력
print("Content-Type: text/html; charset=utf-8")
print()

# 업로드된 파일 처리
form = cgi.FieldStorage()

# HTML 기본 구조 시작
print("""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Result of Traffic_signs</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 20px;
        }
        h1, h2 {
            color: #333;
        }
        img {
            margin-top: 20px;
            width: 200px;
        }
    </style>
</head>
<body>
    <h1>Result of Traffic_signs</h1>
""")

if 'file' in form:
    fileitem = form['file']

    if fileitem.filename:
        # 파일 저장 경로 설정
        upload_dir = r"C:\Users\kimjaesung\10.Basic_WEb\DAY_04\test\cgi-bin"  # 실제 파일을 저장할 경로 설정
        filename = os.path.basename(fileitem.filename)
        filepath = os.path.join(upload_dir, filename)

        # 파일 저장
        with open(filepath, 'wb') as f:
            f.write(fileitem.file.read())

        # 모델 불러오기
        model_path = r"C:\Users\kimjaesung\10.Basic_WEb\DAY_04\test\cgi-bin\traffic_sign_cnn_improved.pth"  # 모델 경로 설정
        model = CNNModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()

 
        # 이미지 전처리 함수 정의 (테스트 시에는 증강을 사용하지 않음)
        transform = transforms.Compose([
        transforms.ToTensor(),  # 텐서로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
])

        def predict(filepath):
            # 이미지 로딩과 예외 처리
            try:
                image = Image.open(filepath).convert('RGB')  # 이미지를 RGB 형식으로 변환
            except Exception as e:
                print(f"<h2>이미지를 로드하는 중 오류가 발생했습니다: {str(e)}</h2>")
                return None
            image = image.resize((32, 32))  # 모델 입력 크기로 리사이즈
            image = transform(image).unsqueeze(0)  # 배치 차원 추가
            output = model(image)
            _, predicted = torch.max(output, 1)
            return predicted.item()


        # 업로드된 이미지로부터 예측
        prediction = predict(filepath)

        # 예측 결과 출력
        label_names = [
        "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", 
        "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", 
        "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", 
        "No passing", "No passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection", 
        "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited", 
        "No entry", "General caution", "Dangerous curve to the left", "Dangerous curve to the right", 
        "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", 
        "Road work", "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", 
        "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits", 
        "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right", 
        "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory", 
        "End of no passing", "End of no passing by vehicles over 3.5 metric tons" ]
      # 라벨 리스트 (43개 클래스에 맞게 정의)
        label = label_names[prediction]

        print(f"<h2>This Traffic_signs is '{label}' ^.^</h2>")
        web_path = filepath.replace(r"C:\Users\kimjaesung\10.Basic_WEb\DAY_04\test\cgi-bin", "/cgi-bin").replace("\\", "/")

        # 이미지 태그 출력
        print(f"<img src='{web_path}' alt='Uploaded Image' />")
        

    else:
        print("<h2>파일 업로드 실패: 파일이 선택되지 않았습니다.</h2>")
else:
    print("<h2>파일 업로드 실패: 폼에 'file' 필드가 없습니다.</h2>")

# HTML 기본 구조 끝
print("""
</body>
</html>
""")
