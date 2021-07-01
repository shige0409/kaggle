import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config


class Trainer():
    def __init__(self) -> None:
        if config.IS_KFOLD_ENSEMBLE:
            self.model = [LogisticRegression(random_state=config.RANDOM_STATE - 333*i) for i in range (config.N_FOLD)]
        else:
            self.model = LogisticRegression(random_state=config.RANDOM_STATE)

    def fit(self, x, y, model_idx = None) -> None:
        if model_idx is None:
            self.model.fit(x, y)
        else:
            self.model[model_idx].fit(x, y)

    def evaluate_auc(self, x, y, model_idx = None) -> float:
        pred = np.zeros(x.shape[0])
        if model_idx is None:
            pred = self.model.predict_proba(x)[:,1]
        else:
            pred = self.model[model_idx].predict_proba(x)[:,1]
        return roc_auc_score(y, pred)

    def save_model(self, name) -> None:
        save_path = os.path.join(config.DIR_PTAH, "model", name)
        pickle.dump(self.model, open(save_path, 'wb'))


if __name__ == "__main__":
    print("starting")
    parser = argparse.ArgumentParser(description='scikit-learnに入力できる形まで整形し、そのデータをpickleなどで保存する')
    parser.add_argument('--parquet_name', help="ファイルのパス")
    args = parser.parse_args()

    df = pd.read_parquet(os.path.join(config.DIR_PTAH, "dataframe", args.parquet_name))
    X, y =  df.drop(columns=config.IGNORE_COLS+config.EXCLUDE_COLS_AFTER), df.TARGET
    print("read parquet done")

    trainer = Trainer()

    if config.IS_KFOLD_ENSEMBLE:
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        scores = []

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
            trainer.fit(X.iloc[train_idx], y.iloc[train_idx], model_idx=n_fold)
            score = trainer.evaluate_auc(X.iloc[valid_idx], y.iloc[valid_idx], model_idx=n_fold)
            print("{0}fold AUC:".format(n_fold), score)
            scores.append(score)

        print('AUC: {0:.4f}±{1:.4f}'.format(np.mean(scores), np.std(scores)))
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=config.RANDOM_STATE, test_size=0.2, stratify=y)
        trainer.fit(X_train, y_train)
        print("train done")
        print("train_auc", trainer.evaluate_auc(X_train, y_train))
        print("val_auc", trainer.evaluate_auc(X_val, y_val))

    trainer.save_model("model.pickle")
    print("save model done")

