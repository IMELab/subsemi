# coding: utf-8

# numpy
import numpy as np

# scikit-learn
## dataset読み込みツール
from sklearn.datasets import fetch_mldata
## 交差検証（クロス・バリデーション）ツール
from sklearn import cross_validation

# chainer
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F

# データの準備
mnist = fetch_mldata('MNIST original')
data = mnist["data"].astype(np.float32) / 255
target = mnist["target"].astype(np.int32) 
train_data, test_data, train_target, test_target = cross_validation.train_test_split(data, target, train_size=0.4)

# ネットワーク構造の定義
model = FunctionSet(
    l1 = F.Linear(784, 100),
    l2 = F.Linear(100, 100),
    l3 = F.Linear(100,  10),
)

# 最適化関数の初期化
optimizer = optimizers.SGD()
optimizer.setup(model)

# 順伝搬処理の定義
def forward(data, target):
    x = Variable(data)
    t = Variable(target)
    h1 = F.relu(model.l1(x))
    h2 = F.relu(model.l2(h1))
    y = model.l3(h2)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


# ミニバッチ処理による訓練
print("============ 訓練 ============");
batchsize = 100
trainsize = len(train_data)
for epoch in range(20):
    print('epoch %d' % epoch)
    sum_loss, sum_accuracy = 0, 0
    indexes = np.random.permutation(trainsize)    
    for i in range(0, trainsize, batchsize):
        #  データをミニバッチに分割
        data_batch = train_data[indexes[i : i + batchsize]]
        target_batch = train_target[indexes[i : i + batchsize]]
        # 最適化関数のパラメータを初期化（勾配を0にする）
        optimizer.zero_grads()
        # 現在の損失と精度を求める
        loss, accuracy = forward(data_batch, target_batch)
        sum_loss      += loss.data * batchsize
        sum_accuracy  += accuracy.data * batchsize
        # 逆伝搬
        loss.backward()
        # 最適化
        optimizer.update()
    mean_loss     = sum_loss / trainsize
    mean_accuracy = sum_accuracy / trainsize
    print "平均精度：%f" % mean_accuracy
    print "平均損失：%f" % mean_loss

# ミニバッチ処理による評価
print("============ 評価 ============");
sum_loss, sum_accuracy = 0, 0
testsize = len(test_data)
for i in range(0, testsize, batchsize):
    data_batch = test_data[i : i + batchsize]
    target_batch = test_target[i : i + batchsize]
    loss, accuracy = forward(data_batch, target_batch)
    sum_loss += loss.data * batchsize
    sum_accuracy += accuracy.data * batchsize

mean_loss = sum_loss / testsize
mean_accuracy = sum_accuracy / testsize
print "平均精度：%f" % mean_accuracy
print "平均損失：%f" % mean_loss
