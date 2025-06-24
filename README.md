
# Детекция блюд в видео на основе YOLOv11

Проект реализует полный конвейер от подготовки данных до обучения, оптимизации и инференса модели YOLOv11 для автоматического распознавания блюд на видеороликах.


## 🚀 Быстрый старт

1. **Клонировать репозиторий**  
   ```bash
   git clone <https://github.com/RoMoRoToR/Testtask.git>


2. **Создать и активировать виртуальное окружение**

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Linux/macOS
   .venv\Scripts\activate.ps1    # Windows PowerShell
   ```

3. **Установить зависимости**

   ```bash
   pip install -r requirements.txt
   # ultralytics, albumentations, opencv-python, torch/cu118 и пр.
   ```

4. **Подготовить данные**

   * Извлечь кадры из видео:

     ```bash
     bash scripts/extract_frames.sh   # или запустить Python-скрипт
     ```
   * Аннотировать кадры (LabelImg / CVAT / автопометка):

     ```bash
     yolo detect annotate model=yolov8n.pt source=frames/ save-dir=labels_autolabel/ conf=0.4
     ```
   * Разбить на train/val/test:

     ```bash
     python scripts/split_dataset.py
     ```
   * Аугментировать train:

     ```bash
     python scripts/augment.py
     ```
   * Объединить оригинал и аугментации:

     ```bash
     python scripts/merge_datasets.py
     ```

---

## 🛠 Конфигурация

* **`data.yaml`**

  ```yaml
  train: merged/images/train
  val:   dataset/images/val
  test:  dataset/images/test

  nc: 7
  names: ['tea','lavash','meat','salad_1','salad_2','soup','kharcho']
  ```
* **`hyp_custom.yaml`** (пример):

  ```yaml
  lr0: 0.001
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 5
  lrf: 0.01
  ```

---

## 🎓 Обучение

### Базовый скрипт

```bash
python train_gpu.py
```

* 100 эпох, `batch=16`, `imgsz=640`, `device=0`.

### Оптимизированный конвейер

```bash
python train_gpu_optimized.py
```

Этапы:

1. Пересчёт анкоров
2. Гиперпараметрическая эволюция
3. Замороженное обучение (`freeze=10` → 30 эпох)
4. Полная дообучка с AutoAugment и AMP (70 эпох)
5. Валидация лучших весов
6. Экспорт в ONNX и TensorRT (.engine)

---

## 📊 Валидация и анализ

* Разбор `runs/train/<exp>/results.txt` через `parse_results.py`.
* Графики кривых `loss`, `mAP`, F1–confidence и распределений центров/размеров боксов сохраняются в `runs/train/<exp>/results.png`.

---

## 🎥 Инференс на видео

```bash
python infer_video.py
```

Параметры в скрипте:

* `source="video/4.MOV"`
* `model="runs/train/<exp>/weights/best.pt"`
* Сохранение полного видео `output_with_labels.mov`

---

## 🔧 Оптимизация для продакшена

* **Export ONNX**:

  ```bash
  yolo export model=runs/train/<exp>/weights/best.pt format=onnx
  ```
* **Export TensorRT**:

  ```bash
  yolo export model=runs/train/<exp>/weights/best.pt format=engine device=0
  ```
* **Pruning** и **Quantization** — см. документацию `ultralytics prune / export half`.

---

---

## ⚖ Лицензия

Проект распространяется под лицензией MIT. См. файл `LICENSE`.

---

```
```
