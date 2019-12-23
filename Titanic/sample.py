# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#学習データ、テストデータの読み込み
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

#学習データを特徴量と目的変数に分ける
#train_x := 特徴量,train_y := 目的変数
train_x = train.drop(['Survived'], axis=1) #Survivedの列を削除(if x == 1 axis:=列, elif x == 0 axis:=行)
train_y = train['Survived']

#テストデータは特徴量のみなので、そのままでいい
test_x = test.copy()

"""特徴量の作成"""

from sklearn.preprocessing import LabelEncoder

#予測に寄与する変数ではないため変数PassengerIdの列を削除する (入れたままだとモデルが意味のある変数だと勘違いする恐れがある)
train_x = train_x.drop(['PassengerId'], axis = 1)
test_x = test_x.drop(['PassengerId'], axis = 1)

#処理が複雑なので一旦変数Name, Ticket, Cabinを除外する
train_x = train_x.drop(['Name','Ticket','Cabin'], axis = 1)

test_x = test_x.drop(['Name','Ticket','Cabin'], axis = 1)

"""
GBDTでは文字列をそのまま入れてもエラーとなってしまうため、何らかの方法で数値に変換する必要がある
ここでは、labelencodingという手法を使う
"""

#それぞれのカテゴリ変数にlabel encodingを適用する
for c in ['Sex','Embarked']:
    #学習データに基づいてどう変換するかを定める
    le = LabelEncoder()
    #欠損値があれば'NA'文字で埋め,変換したいデータとして選択(le.fit()の引数が変換したいデータ)
    le.fit(train_x[c].fillna('NA'))
    
    #学習データ、テストデータを変換する
    #train_xとtest_xは同様に変換させる
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

"""モデルの作成"""
#GBDTのライブラリの1つであるxgboostを用いてモデルを作成

from xgboost import XGBClassifier

#モデルの作成及び学習データを与えての学習
model = XGBClassifier(n_estimators = 20, random_state = 71)
model.fit(train_x, train_y)

#テストデータの予測値を確率で出力する
pred = model.predict_proba(test_x)[:,1]

#テストデータの予測値を二値に変換する
#pred内の0.5より大きいデータは1,それ以外は0にする
pred_label = np.where(pred > 0.5, 1, 0)

#提出用ファイルの作成
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':pred_label})
submission.to_csv('submission_first.csv' ,index = False)

"""モデルの評価(クロスバリデーション)"""

"""
[クロスバリデーション(交差検証)]
データを複数のブロックに分けて、うち一つを評価用のデータ(バリデーションデータ)とし、
残りを学習用のデータとすることを評価用のデータを入れ替えて繰り返す方法
"""

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
"""
コンペの評価指標はaccuracyであるが、accuracyは小さな変化をとらえづらい
⇒loglossという指標を用いる

logloss := 予測確率が外れているほど高いペナルティが与えられ、低いほど良い指標
"""
#各foldのスコアを保存するリスト
scores_accuracy = []
scores_logloss = []

#クロスバリデーションを行う
#学習データを4分割し、うち1つをバリデーションデータとすることをバリデーションデータを変えて繰り返す
kf = KFold(n_splits = 4, shuffle = True, random_state = 71)
for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
    
    #モデルの学習を行う
    model = XGBClassifier(n_estimator=20, random_state = 71)
    model.fit(tr_x,tr_y)
    
    #バリデーションデータの予測値を確率で出力する
    va_pred = model.predict_proba(va_x)[:,1]
    
    #バリデーションデータでのスコアを計算する
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)
    
    #そのfoldのスコアを保存する
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)
    
#各foldのスコアを計算する
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')

"""モデルのチューニング"""

"""
[ハイパーパラメータ]
学習の前に設定し、学習の方法や速度、どれだけ複雑なモデルにするかを定めるパラメータ
最適でない場合モデルの力を十分に発揮できないため、ハイパーパラメータのチューニングが必要
ここでは、グリッドサーチという手法を用いる

[グリッドサーチ]
チューニング対象のハイパーパラメータのすべての組合せについて探索を行い、最もスコアが良いものを採用する方法
"""

import itertools

"""
[XGBoostのパラメータのチューニング]
min_child_weight(default 1) := 子ノードにおいて観察されるデータの重み付けの合計値の最小の値で、過学習を避けるために用いられる.
                               高い値にすることで特定のサンプルに観察されるような傾向を学習することを避けられる.ただし、高くし過ぎるとフィッティングが悪くなる. 

max_depth(default 6) := 木の深さの最大値.過学習を制御するために用いられる.高いと過学習しやすくなる.
"""

#チューニング候補とするパラメータを準備する
param_space = {
    'max_depth':[3,5,7],
    'min_child_weight': [1.0,2.0,4.0]
}

#探索するパラメータの組合せ
param_combinations = itertools.product(param_space['max_depth'],param_space['min_child_weight'])

#各パラメータの組合せ、それに対するスコアを保存するリスト
params = []
scores = []

#各パラメータの組合せごとに、クロスバリデーションで評価を行う
for max_depth,min_child_weight in param_combinations:
    
    score_folds = []
    #クロスバリデーションを行う
    #学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
    kf = KFold(n_splits = 4, shuffle = True, random_state = 123456)
    for tr_idx, va_idx in kf.split(train_x):
        #学習データを学習データとバリデーションデータに分ける
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        
        #モデルの学習を行う
        model = XGBClassifier(n_estimator = 20, random_state = 71, max_depth=max_depth, min_child_weight= min_child_weight)
        
        model.fit(tr_x,tr_y)
        
        #バリデーションデータでのスコアを計算し、保存する
        va_pred = model.predict_proba(va_x)[:,1]
        logloss = log_loss(va_y,va_pred)
        score_folds.append(logloss)
        
    #各foldのスコアを平均する
    score_mean = np.mean(score_folds)
    
    #パラメータの組合せ、それに対するスコアを保存する
    params.append((max_depth,min_child_weight))
    scores.append(score_mean)
    
#最もスコアが良いものをベストなパラメータとする
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')

"""アンサンブル"""

"""
[アンサンブル]
単一のモデルでのスコアには限界があっても、複数のモデルを組み合わせて予測することでスコアが向上する場合があり、
そのように予測することをアンサンブルという
それぞれのモデルの精度が高いだけでなく、それらのモデルが多様な時にスコアが向上しやすい
"""

from sklearn.linear_model import LogisticRegression

#xgboostモデル
model_xgb = XGBClassifier(n_estimator = 20, random_state = 71)
model_xgb.fit(train_x,train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:,1]

train_x2 = train_x.copy()
test_x2 = test_x.copy()

#ロジスティック回帰モデル
#xgboostモデルとは異なる特徴量を入れる必要があるので、別途train_x2,test_x2を作成
model_lr = LogisticRegression(solver = 'lbfgs', max_iter=300)
model_lr.fit(train_x2, train_y)
pred_lr = model_lr.predict_proba(test_x2)[:,1]

#予測値の加重平均をとる
pred = pred_xgb * 0.8 + pred_lr * 0.2
pred_label = np.where(pred > 0.5, 1, 0)
