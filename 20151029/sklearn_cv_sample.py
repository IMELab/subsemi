# coding: utf-8

# scikit-learn
## dataset読み込みツール
from sklearn.datasets import fetch_mldata
## random forestを定義したクラス
from sklearn.ensemble import RandomForestClassifier
## 交差検証（クロス・バリデーション）ツール
from sklearn import cross_validation

# データを準備する
mnist = fetch_mldata('MNIST original')

# ランダムフォレストの学習器を作る
rfc = RandomForestClassifier(n_estimators=10)

# クロスバリデーションツールでスコアを計算
score = cross_validation.cross_val_score(rfc, mnist["data"], mnist["target"], cv=5)
print score
