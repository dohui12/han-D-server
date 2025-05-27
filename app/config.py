# 모델 및 클래스 설정
import torch

param = {
    "model_path": "models/best_kpnet.pt",
    "time_steps": 60,
    "feat_dim": 138,
    "n_classes": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "class_names": ["class_0", "class_1", "class_2", "class_3", "class_4"]
}