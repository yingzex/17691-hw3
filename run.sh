# # add stages 
# dvc stage add --force -n prepare \
#     -d prepare.py -d data/vineyard_weather_1948-2017.csv \
#     -o data/prepared.csv \
#     python prepare.py

# dvc stage add --force -n featurize \
#     -d featurize.py \
#     -d data/prepared.csv \
#     -o data/train_data.csv \
#     -o data/test_data.csv \
#     python featurize.py

# dvc stage add --force -n train \
#     -d train.py \
#     -d params.yaml \
#     -d data/train_data.csv \
#     -o model/model.pkl \
#     python train.py

# dvc stage add --force -n evaluate \
#     -d evaluate.py -d model/model.pkl -d data/test_data.csv \
#     -d params.yaml \
#     -o results/evaluation_report.txt \
#     python evaluate.py

# reproduce experiments from the DVC pipeline. This will execute the steps defined in your dvc.yaml file.
dvc exp run

dvc metrics show