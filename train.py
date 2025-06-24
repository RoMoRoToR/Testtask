# train_gpu.py
import os
import multiprocessing
from ultralytics import YOLO

def main():
    # Параметры
    exp_name = "food_exp_gpu"   # базовое имя эксперимента
    project  = "runs/train"

    # 1) Обучаем модель
    model = YOLO("yolo11s.pt")
    _ = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project=project,
        name=exp_name
    )

    # 2) Находим папку эксперимента (самую свежую)
    exp_root = os.path.join(project)
    # Сортируем по времени модификации и берём последний
    runs = sorted(
        (d for d in os.listdir(exp_root) if d.startswith(exp_name)),
        key=lambda d: os.path.getmtime(os.path.join(exp_root, d))
    )
    if not runs:
        raise FileNotFoundError(f"Не найден ни один эксперимент в {exp_root} с именем {exp_name}*")
    last_run = runs[-1]
    best_weights = os.path.join(exp_root, last_run, "weights", "best.pt")
    print("Путь к лучшим весам:", best_weights)

    # 3) Валидация на лучших весах
    metrics = model.val(
        data="data.yaml",
        weights=best_weights,
        batch=16,
        device=0
    )
    print("Результаты валидации:\n", metrics)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
