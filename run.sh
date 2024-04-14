dvc stage add --force -n prepare \
    -d prepare.py -d data/vineyard_weather_1948-2017.csv \
    -o data/prepared.csv \
    python prepare.py

dvc stage add --force -n featurize \
    -d featurize.py \
    -d data/prepared.csv \
    -o data/train_data.csv \
    -o data/test_data.csv \
    python featurize.py

dvc stage add --force -n train \
    -d train.py \
    -d data/train_data.csv \
    -o model/model.pkl \
    python train.py

dvc stage add --force -n evaluate \
    -d evaluate.py -d model/model.pkl -d data/test_data.csv \
    -o results/evaluation_report.txt \
    python evaluate.py
