# 모델 및 클래스 설정
import torch

param = {
    "model_path": "models/best.pt",
    "time_steps": 60,
    "feat_dim": 138,
    "n_classes": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "class_names": ["class_0", "class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7", "class_8", "class_9",]
}