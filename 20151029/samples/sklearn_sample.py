# coding: utf-8

# scikit-learn
## dataset読み込みツール
from sklearn.datasets import fetch_mldata
## random forestを定義したクラス
from sklearn.ensemble import RandomForestClassifier
## 交差検証（クロス・バリデーション）ツール
from sklearn import cross_validation
##評価ツール
from sklearn.metrics import accuracy_score

# データを準備する
mnist = fetch_mldata('MNIST original')
train_data, test_data, train_target, test_target = cross_validation.train_test_split(mnist["data"], mnist["target"], train_size=0.4)

# ランダムフォレストの学習器を作る
rfc = RandomForestClassifier(n_estimators=10)

# 訓練用データを用いて、学習器を訓練する
trained_rfc = rfc.fit(train_data, train_target) 

# 訓練済みの学習器を用いて、評価用の説明変数からラベルを予測
predicted_target = trained_rfc.predict(test_data)

# 実際のラベルと予測したラベルを比較して正答率を計算する
score = accuracy_score(test_target, predicted_target)
print score
