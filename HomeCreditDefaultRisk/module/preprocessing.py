import argparse
import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import config
from utils import create_preprocess_pipeline, preprocess_test, preprocess_test_good

class Preprocessor():
    def __init__(self) -> None:
        self.numerical_cols = []
        self.categorical_cols = []
        self.preprocessor = None

    def fit(self, df:pd.DataFrame) -> None:
        # pipeline setup
        self.numerical_cols = [k for k,v in df.dtypes.items() if v != np.object and k not in config.IGNORE_COLS]
        self.categorical_cols = [k for k,v in df.dtypes.items() if v == np.object and k not in config.IGNORE_COLS]
        self.preprocessor = create_preprocess_pipeline(self.numerical_cols, self.categorical_cols)
        self.preprocessor.fit(df)

    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(df)

    def save_df(self, df:pd.DataFrame, name=""):
        save_path = os.path.join(config.DIR_PTAH, "dataframe", name)
        df.to_parquet(save_path)

    def save_preprocessor(self, name):
        # 前処理パイプラインをpickleで保存
        save_path = os.path.join(config.DIR_PTAH, "preprocessor", name)
        pickle.dump(self.preprocessor, open(save_path, 'wb'))


if __name__ == "__main__":
    print("starting")
    parser = argparse.ArgumentParser(description='scikit-learnに入力できる形まで整形し、そのデータをpickleなどで保存する')
    parser.add_argument('--csv_name', help="ファイルのパス")
    args = parser.parse_args()
    df = pd.read_csv(os.path.join(config.DIR_PTAH, "csv", args.csv_name))
    # 処理に含めないカラムをdrop
    df.drop(columns=config.EXCLUDE_COLS, inplace=True)
    print("read_csv done")


    # 実験で合格したコード => 後でモジュールにまとめる
    df = preprocess_test_good(df)
    # 前処理実験用のコード => 後でモジュールにまとめる
    df = preprocess_test(df)

    prep = Preprocessor()
    prep.fit(df)
    print("preprocessor fit done")

    df_pre = prep.transform(df)
    prep.save_df(df_pre, name='application_train.parquet')
    print("df parquet save done")
    prep.save_preprocessor('prep.pickle')
    print("Preprocessor pickle dump done")

    






