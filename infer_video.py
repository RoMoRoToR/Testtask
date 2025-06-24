# infer_video.py
import cv2
from ultralytics import YOLO

# 1. Загружаем модель
model = YOLO("runs/train/food_exp_gpu5/weights/best.pt")

# 2. Открываем исходное видео и готовим VideoWriter
cap    = cv2.VideoCapture("video/2_1.MOV")
fps    = cap.get(cv2.CAP_PROP_FPS)
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Кодек на четыре символа, сохраняем в MOV-контейнер
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter("output_with_labels2.mov", fourcc, fps, (w, h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # дошли до конца видео

    # 3. Инференс одним кадром
    results = model(frame, conf=0.3)[0]

    # 4. Отрисовка боксов и подписей
    annotated = results.plot()

    # 5. Запись кадра в выходной файл
    out.write(annotated)

    frame_idx += 1

# 6. Освобождаем ресурсы
cap.release()
out.release()

print(f"Готово: сохранено {frame_idx} кадров в файл output_with_labels.mov")
