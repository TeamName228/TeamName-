import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os
import logging
import time
import cv2
import numpy as np

# Настройка кодировки и логирования
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])

# Определение модели EnhancedSymbolCNN
class EnhancedSymbolCNN(nn.Module):
    def __init__(self, num_classes=369):
        super(EnhancedSymbolCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Пути
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, '..', 'model')
upload_dir = os.path.join(script_dir, '..', 'Uploads')
os.makedirs(upload_dir, exist_ok=True)

# Загрузка маппинга классов
try:
    with open(os.path.join(model_dir, 'class_to_symbol_normalized.json'), 'r', encoding='utf-8') as f:
        class_to_symbol = json.load(f)
    logging.info("Маппинг классов загружен успешно")
except FileNotFoundError:
    logging.error("Файл class_to_symbol_normalized.json не найден!")
    sys.exit(1)

# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = EnhancedSymbolCNN(num_classes=len(class_to_symbol)).to(device)
    model_path = os.path.join(model_dir, 'hasyv2_model_arg.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logging.info(f"Модель загружена на {device}")
except Exception as e:
    logging.error(f"Ошибка загрузки модели: {str(e)}")
    sys.exit(1)

# Предобработка изображения
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def preprocess_image(image):
    img = np.array(image)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    img = 255 - img  # Инверсия
    return Image.fromarray(img)

# Основная логика
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python recognize_symbol.py <путь_к_изображению>")
        sys.exit(1)
    image_path = sys.argv[1]
    try:
        img = Image.open(image_path).convert('L')
        img = preprocess_image(img)
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Сохранение обработанного изображения
        processed_img = transforms.ToPILImage()(img_tensor.squeeze(0).cpu())
        processed_img_path = os.path.join(upload_dir, f"processed_{int(time.time() * 1000)}.jpg")
        processed_img.save(processed_img_path)
        logging.info(f"Обработанное изображение сохранено: {processed_img_path}")

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            class_idx = predicted.item()
            latex_command = class_to_symbol.get(str(class_idx), "Неизвестный символ")
            # Простая замена некоторых LaTeX-команд на Unicode
            symbol_map = {
                '\\alpha': 'α',
                '\\beta': 'β',
                '\\gamma': 'γ',
                '\\delta': 'δ',
                '\\epsilon': 'ε',
                '\\theta': 'θ',
                '\\lambda': 'λ',
                '\\mu': 'μ',
                '\\pi': 'π',
                '\\sigma': 'σ',
                '\\phi': 'φ',
                '\\psi': 'ψ',
                '\\Sigma': 'Σ',
                '\\omega': 'ω',
                '\\leqslant': '⩽',
                '\\prod': '∏',
                '\\sum': '∑',
            }
            symbol = symbol_map.get(latex_command, latex_command)
            confidence = confidence.item()

            # Вывод в формате JSON
            result = {
                'symbol': symbol,
                'latex': latex_command,
                'confidence': f"{confidence:.2%}"
            }
            print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(f"Ошибка обработки изображения: {str(e)}")
        sys.exit(1)