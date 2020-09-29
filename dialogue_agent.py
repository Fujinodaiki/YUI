from os.path import dirname, join, normpath
import unicodedata
import neologdn
import MeCab
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC



class DialogueAgent:
    #分かち書き
    def __init__(self):
        self.tagger = MeCab.Tagger()

    def _tokenize(self, text):
        text = unicodedata.normalize('NFKC',text) #unicode正規化
        text = neologdn.normalize(text) #文字列の正規化
        node = self.tagger.parseToNode(text)
        result = []
        while node:
            features = node.feature.split(',')

            if features[0] != 'BOS/EOS':
                if features[0] not in ['助詞','助動詞']:
                    token = features[6]\
                        if features[6] != '*'\
                        else node.surface
                    result.append(token)
            node = node.next
        return result
    #ここまで

    def train(self, texts, labels):
        #語彙の獲得と特徴ベクトル化を一度に行っている。
        vectorizer = TfidfVectorizer(tokenizer=self._tokenize)
        bow = vectorizer.fit_transform(texts)  # <1>
        #ここまで

        classifier = SVC()
        classifier.fit(bow, labels)

       
        self.vectorizer = vectorizer #辞書生成済みのvectorizer
        self.classifier = classifier #学習済みのclassifier(svc)

    def predict(self, texts):
        bow = self.vectorizer.transform(texts) #辞書を元に出現回数を数える
        return self.classifier.predict(bow)    #出現回数を元にsvcで予測している。


if __name__ == '__main__':
    BASE_DIR = normpath(dirname(__file__))#このスクリプトがあるファイルの絶対パスを取得

    training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))  #pandasを使用して学習データのcsvファイルを読み込み

    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label']) #読み込んだ学習データのcsvファイルを辞書化している。

    with open(join(BASE_DIR, './replies.csv')) as f:  #返答用のcsvファイルの読み込み
        replies = f.read().split('\n') #読み込んだ返答用csvファイルを読み込み改行でlist化している。

    input_text = '名前を教えてよ'
    predictions = dialogue_agent.predict([input_text])  # 予測した番号を返答している
    predicted_class_id = predictions[0]  #predictionsに入っている０番の数字を指定している。（情報は一つしか入っていないため０番固定）

    print(replies[predicted_class_id]) #返答リストから番号を参照して表示する。

    while True:
        input_text = input()
        predictions = dialogue_agent.predict([input_text])
        predicted_class_id = predictions[0]
        print(replies[predicted_class_id])
    
