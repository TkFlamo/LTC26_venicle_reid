# Команда MLstrom

Используется одна модель Vit_b. Скачать веса reid.pt https://disk.yandex.ru/d/n5t5Z6g53vs3Xw и расположить в папке ./deploy/models_current/members/base_full. 

Результаты инференса на тестовой выборке лежат по пути outputs/test_hackaton/


## 1. Установка окружения

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install \
  torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements-runtime-a5000.txt
python -m pip install --no-deps -e .
```

## 2. Официальный тест скорости 

```bash
PYTHONPATH=src python scripts/63_benchmark_official_extract_pt_v094.py \
  --input-dir /path/to/test_hackaton \
  --deployment-dir deploy/models_current \
  --out artifacts/speed_base_full \
  --device 0 \
  --precision fp16 \
  --workers 8 \
  --prefetch-factor 2
```

## 3. Запуск inference(тестовой выборки) 

```bash
python run_hackathon.py \
  --input-dir /path/to/test_hackaton \
  --out outputs/test_hackaton \
  --device 0 \
  --precision fp16 \
  --batch 32 \
  --workers 8
```
На выходе:

```text
outputs/test_hackaton/
├── submission.csv
├── candidates.csv
└── embeddings.npy
```
Параметры:

| Параметр      | Назначение          | Рекомендуемое значение |
| ------------- | ------------------- | ---------------------: |
| `--device`    | CUDA device         |                    `0` |
| `--precision` | precision inference |                 `fp16` |
| `--batch`     | размер batch        |                   `32` |
| `--workers`   | DataLoader workers  |                    `8` |
| `--member`    | активная модель     |            `base_full` |

`run_hackathon.py` автоматически:

1. проверяет структуру входных CSV и наличие изображений;
2. выбирает `base_full` из packaged deployment;
3. извлекает признаки query;
4. извлекает признаки gallery;
5. выполняет retrieval;
6. применяет reranker;
7. применяет refusal calibration;
8. сохраняет `submission.csv`, `candidates.csv`, `embeddings.npy`;
9. проверяет корректность итоговых submission-файлов.

