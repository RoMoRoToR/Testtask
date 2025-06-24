from ultralytics import YOLO

# 1) Загружаем модель
model = YOLO("runs/train/food_exp_gpu5/weights/best.pt")

# Папка, куда будут сохраняться результаты
project = "runs/inference"
name    = "video_test_py"

# 2) Запускаем предсказание на видео
results = model.predict(
    source="video/4.MOV",  # путь к видео
    conf=0.3,                      # порог confidence
    save=True,                     # сохранять выход
    save_txt=False,                # если нужно txt-метки — True
    project=project,
    name=name,
    show=False
)

# results – это список, берём первый элемент и у него читаем .path
output_path = results[0].path
print("Видео с предсказаниями сохранено в:", output_path)
