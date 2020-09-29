from os.path import dirname, join, normpath

import pandas as pd
from sklearn.metrics import accuracy_score

from dialogue_agent import DialogueAgent  #作った識別器をimport

if __name__ == '__main__':
    BASE_DIR = normpath(dirname(__file__))

    # Training
    training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))

    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label']) #識別器に使用するデータを挿入して学習

    # Evaluation
    test_data = pd.read_csv(join(BASE_DIR, './test_data.csv'))  # テストデータの読みこみ

    predictions = dialogue_agent.predict(test_data['text'])  # テストデータのテキストから予測させる

    print(accuracy_score(test_data['label'], predictions))  # 予測結果と正解クラスIDを比較して評価する。sklearnのaccuracy_scoreで正解率を計算
