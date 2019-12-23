import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_x = train.drop(['Survived'], axis = 1)
train_y = train['Survived']

test_x = test.copy()

"""特徴量の作成"""

from sklearn.preprocessing import LabelEncoder

#予測に寄与する変数ではないため変数PassengerIdの列を削除する
train_x = train_x.drop(['PassengerId'], axis = 1)
test_x = test_x.drop(['PassengerId'], axis = 1)

#処理が複雑なので一旦変数Name, Ticket, Cabinを除外する
train_x = train_x.drop(['Name','Ticket','Cabin'], axis = 1)
test_x = test_x.drop(['Name','Ticket','Cabin'], axis = 1)

#それぞれのカテゴリ変数にlabel encodingを適用する
for c in ['Sex','Embarked']:
    #学習データに基づいてどう変換するかを定める
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    #学習データ、テストデータを変換する
    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

"""モデルの作成"""
#GBDTのライブラリの1つであるxgboostを用いてモデルを作成

from xgboost import XGBClassifier

#モデルの作成及び学習データを与えての学習
model = XGBClassifier(n_estimators = 20, random_state = 71)
model.fit(train_x, train_y)

#テストデータの予測値を確率で出力する
pred = model.predict_proba(test_x)[:,-1]

#テストデータの予測値を二値に変換する
pred_label = np.where(pred > 0.5, 1, 0)

#提出用ファイルの作成
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':pred_label})
submission.to_csv('submission_first.csv' ,index = False)
