dvc stage add --force -n prepare \
    -d prepare.py -d data/vineyard_weather_1948-2017.csv \
    -o data/prepared.csv \
    python prepare.py

dvc stage add --force -n featurize \
    -d featurize.py \
    -d data/prepared.csv \
    -o data/features.csv \
    python featurize.py


dvc stage add --force -n train -d train.py -d data/features.csv -o model/model.pkl python train.py
