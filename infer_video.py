import cv2
from ultralytics import YOLO

model = YOLO("runs/train/food_exp_gpu5/weights/best.pt")

cap    = cv2.VideoCapture("video/2_1.MOV")
fps    = cap.get(cv2.CAP_PROP_FPS)
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter("output_with_labels2.mov", fourcc, fps, (w, h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame, conf=0.3)[0]


    annotated = results.plot()


    out.write(annotated)

    frame_idx += 1


cap.release()
out.release()

print(f"Готово: сохранено {frame_idx} кадров в файл output_with_labels.mov")
