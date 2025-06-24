import os
import multiprocessing
from ultralytics import YOLO


DATA_CFG       = "data.yaml"
BASE_MODEL     = "yolo11s.pt"
PROJECT        = "runs/train"
EXP_NAME       = "food_exp_gpu"
BATCH          = 16
IMG_SZ         = 640
DEVICE         = 0
EPOCHS_FREEZE  = 30
EPOCHS_FINE    = 70
FREEZE_LAYERS  = 10
EVOLVE         = True
AMP            = True

def main():
    print("[OPT] Пересчет анкоров под ваш датасет...")
    YOLO(BASE_MODEL).train(
        data=DATA_CFG,
        epochs=1,
        imgsz=IMG_SZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=f"{EXP_NAME}_anchor",
        mode="anchor"
    )

    hyp_cfg = None
    if EVOLVE:
        print("[OPT] Запуск гиперпараметрической эволюции...")
        YOLO(BASE_MODEL).train(
            data=DATA_CFG,
            epochs=1,
            imgsz=IMG_SZ,
            batch=BATCH,
            device=DEVICE,
            project=PROJECT,
            name=f"{EXP_NAME}_evolve",
            evolve=True
        )
        hyp_cfg = os.path.join(PROJECT, f"{EXP_NAME}_evolve", "hyp_evolved.yaml")

    print("[TRAIN] Этап 1: обучение с freeze слоёв...")
    model = YOLO(BASE_MODEL)
    model.train(
        data=DATA_CFG,
        epochs=EPOCHS_FREEZE,
        imgsz=IMG_SZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=f"{EXP_NAME}_freeze",
        freeze=FREEZE_LAYERS,
        amp=AMP,
        hyp=hyp_cfg
    )

    print("[TRAIN] Этап 2: полная дообучка с AutoAugment...")
    last_weights = os.path.join(PROJECT, f"{EXP_NAME}_freeze", "weights", "last.pt")
    model = YOLO(last_weights)
    model.train(
        data=DATA_CFG,
        epochs=EPOCHS_FINE,
        imgsz=IMG_SZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=f"{EXP_NAME}_fine",
        amp=AMP,
        augment="auto",
        hyp=hyp_cfg
    )

    exp_root = PROJECT
    runs = sorted(
        (d for d in os.listdir(exp_root) if d.startswith(EXP_NAME)),
        key=lambda d: os.path.getmtime(os.path.join(exp_root, d))
    )
    if not runs:
        raise FileNotFoundError(f"Не найден ни один эксперимент {EXP_NAME}* в {exp_root}")
    last_run = runs[-1]
    best_weights = os.path.join(exp_root, last_run, "weights", "best.pt")
    print("[RESULT] Лучшие веса:", best_weights)

    print("[VAL] Запуск валидации...")
    metrics = YOLO(best_weights).val(
        data=DATA_CFG,
        batch=BATCH,
        device=DEVICE
    )
    print("Результаты валидации:\n", metrics)

    print("[EXPORT] Экспорт ONNX и TensorRT engine...")
    model = YOLO(best_weights)
    onnx_path   = model.export(format="onnx")
    engine_path = model.export(format="engine", device=DEVICE)
    print(f"Экспорт завершен: ONNX={onnx_path}, TensorRT Engine={engine_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
