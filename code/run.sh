dvc stage add --name featurize \
              --deps src/featurize.py \
              --deps data/prepared.csv \
              --outs data/features.csv \
              python src/featurize.py data/prepared.csv data/features.csv

dvc stage add --name train \
              --deps src/train.py \
              --deps data/features.csv \
              --outs data/predict.dat \
              python src/train.py data/features.csv


