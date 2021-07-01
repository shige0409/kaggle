import argparse
import os
import sys
import subprocess
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import config
from utils import preprocess_test, preprocess_test_good
import utils
sys.modules['utils'] = utils


class Tester():
    def __init__(self, prep_name, model_name) -> None:
        self.preprocessor = pickle.load(open(os.path.join(config.DIR_PTAH, "preprocessor", prep_name), "rb"))
        self.model = pickle.load(open(os.path.join(config.DIR_PTAH, "model", model_name), "rb"))

    def predict_proba(self, df:pd.DataFrame) -> np.array:
        if config.IS_KFOLD_ENSEMBLE:
            preds = np.zeros((df.shape[0], 2))
            for i in range(config.N_FOLD):
                preds += self.model[i].predict_proba(df)
            return preds / config.N_FOLD

        else:
            return self.model.predict_proba(df)

    def submit(self, df:pd.DataFrame):
        submit_path = os.path.join(config.DIR_PTAH, "submit", "submit-v{}.csv".format(config.VERSION))
        df[config.IGNORE_COLS].to_csv(submit_path,index=None)
        script = "kaggle competitions submit -c home-credit-default-risk -f {0} -m {1}".format(
          submit_path,
          config.SUBMIT_MESSAGE  
        )
        subprocess.call(script, shell=True)

if __name__ == '__main__':
    print("starting")
    parser = argparse.ArgumentParser(description='scikit-learnに入力できる形まで整形し、そのデータをpickleなどで保存する')
    parser.add_argument('--csv_name', help="ファイルのパス")
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(config.DIR_PTAH, "csv", args.csv_name))
    df["TARGET"] = 0
    # drop
    df.drop(columns=config.EXCLUDE_COLS, inplace=True)
    print("read_csv done")

    tester = Tester(prep_name="prep.pickle", model_name='model.pickle')
    print("pickle load done")

    # 実験で合格したコード => 後でモジュールにまとめる
    df = preprocess_test_good(df)

    # 前処理実験用のコード => 後でモジュールにまとめる
    df = preprocess_test(df)

    df_pre = tester.preprocessor.transform(df)
    pred = tester.predict_proba(df_pre.drop(columns=config.IGNORE_COLS+config.EXCLUDE_COLS_AFTER))[:,1]
    df_pre["TARGET"] = pred
    print("predict done")

    tester.submit(df_pre)
    print("sumit done! message:", config.SUBMIT_MESSAGE)

