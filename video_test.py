from ultralytics import YOLO


model = YOLO("runs/train/food_exp_gpu5/weights/best.pt")


project = "runs/inference"
name    = "video_test_py"


results = model.predict(
    source="video/4.MOV",
    conf=0.3,
    save=True,
    save_txt=False,
    project=project,
    name=name,
    show=False
)


output_path = results[0].path
print("Видео с предсказаниями сохранено в:", output_path)
