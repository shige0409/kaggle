# pipeline
import config

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder


# 欠損値処理クラス
class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, how="median", add_nan=False, return_df = True):
        self.cols = cols
        self.how = how
        self.return_df = return_df
        self.add_nan = add_nan
        
    def fit(self, X):
        # カラム指定があれば
        if self.cols:
            self.impute_dict = X[self.cols].agg(self.how).to_dict()
        # 無ければ全部のカラム
        else:
            self.impute_dict = X.agg(self.how).to_dict()
        if self.add_nan:
            # num_nan_cols = num_df.columns[num_df.isnull().any()]
            self.nan_cols = X[self.cols].columns[X[self.cols].isnull().any()]
        return self
    
    def transform(self, X):
        df = X.copy()
        # 数値の欠損情報を追加
        if self.add_nan:
            nan_cols_impute = [col+"_is_nan" for col in self.nan_cols]
            df[nan_cols_impute] = df[self.nan_cols].isnull().astype("int")
        # 欠損穴埋め
        df.fillna(self.impute_dict, inplace=True)

        if self.return_df:
            return df
        else:
            return df.values
        

# 標準化をデータフレームとして扱うクラス
class StandardScalerDf(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, return_df = True):
        self.cols = cols
        self.return_df = return_df
        self.ss = StandardScaler()
        
    def fit(self, X):
        if self.cols:
            self.ss.fit(X[self.cols])
        # カラム指定を強要
        else:
            raise ValueError("cols is None")
        return self
        
    def transform(self, X):
        DF = X.copy()
        DF[self.cols] = self.ss.transform(X[self.cols])
        if self.return_df:
            return DF
        else:
            return DF.values

# # 実験で合格したコード => 後でモジュールにまとめる
def preprocess_test_good(X):
    df = X.copy()
    # DAYS系
    days_cols = [col for col in df.columns if "DAYS" in col]
    years_cols = [col.replace("DAYS", "YEARS") for col in days_cols]
    df[days_cols] = np.abs(df[days_cols] / 365)
    df.rename(columns={k:v for k,v in zip(days_cols, years_cols)}, inplace=True)
    def separage_age(age):
        if age < 24:
            return "0層"
        elif age < 35:
            return "1層"
        elif age < 50:
            return "2層"
        else:
            return "3層"
    df["M1F1"] = df.CODE_GENDER + df.YEARS_BIRTH.apply(separage_age)
    df["EMPLOYES_DIV_BIRTH"] = df["YEARS_EMPLOYED"] / df["YEARS_BIRTH"]
    df.drop(columns=["CODE_GENDER", "YEARS_BIRTH"], inplace=True)
    # AMT系
    df["GOODS_PRICE_DIV_INCOME"] = df["AMT_GOODS_PRICE"] / df["AMT_INCOME_TOTAL"]
    df["GOODS_PRICE_DIV_CREDIT"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["CREDIT_DIV_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    return df

# 前処理実験用のコード => 後でモジュールにまとめる
def preprocess_test(X):
    df = X.copy()
    return df

def create_preprocess_pipeline(numerical_cols=[], categorical_cols=[]):
    return Pipeline(steps=[
        # 数値データ
        # 数値欠損はとりあえず平均値補完
        ("numerical_pp", Imputer(cols=numerical_cols, how="median", add_nan=True)),
        ("ss", StandardScalerDf(cols=numerical_cols)),
        # カテゴリデータ
        ("categorical_ohe", OneHotEncoder(cols=categorical_cols, use_cat_names=True, handle_missing="ignore", handle_unknown="ignore"))
        # ("categorical_pp", BinaryEncoder(cols=categorical_cols, handle_missing="impute", handle_unknown="ignore"))
    ])