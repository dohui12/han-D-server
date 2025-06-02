# 🖐️Han:D Sign Language Education APP 

2025-1 Deep Learning Term project

**Han:D**는 사용자의 손동작을 실시간으로 인식하여 수어 학습을 보조하는 교육 앱입니다.
딥러닝 모델을 기반으로 수어 영상을 분석하고 예측 결과를 반환하는 FastAPI 서버 부분 코드입니다.


## 📌 Features

- 사용자 영상에서 프레임 추출 및 keypoint 특징 추출
- LSTM 기반 딥러닝 모델을 통한 수어 클래스 분류
- 신뢰도가 threshold 값 이하일 경우 예외 처리
- Flutter 앱과 연동 가능한 REST API 제공


## 🧠 Model Overview

- **Input**: `(1, 60, 138)`  
  (60프레임, 각 프레임당 46개 관절의 3D 좌표)

- **Architecture**:
  LSTM (138 → 128)
→ BatchNorm1d(128)
→ LSTM (128 → 128)
→ Linear (128 → 64)
→ Dropout (p=0.3)
→ Linear (64 → num_classes)

- **Output**: `logits.shape = (1, n_classes)` → softmax 확률 분포


## 🚀 API Usage

- **Request**:
  `file`: `.mp4` 형식의 수어 영상
  
- **Response 예시**:
```json
{
"prediction": "class_3"
}
```

## 🧪 How to Run

1. 패키지 설치
```
pip install -r requirements.txt
```

2. 서버 실행
```
uvicorn app.main:app --reload
```

## 🧑‍💻 Contributors
| 이름  | 역할                |
| --- | ----------------- |
| 최정은 | 데이터셋 구성, 앱 기획     |
| 김도희 | 딥러닝 모델 설계 및 서버 구축 |
| 지유나 | 전처리 파이프라인 구현      |
