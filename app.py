from flask import Flask, url_for, render_template, request
from dialogue_agent import DialogueAgent
from os.path import dirname, join, normpath
import MeCab
import pandas as pd
import oseti

app = Flask(__name__)

@app.route('/')
def index():
    name = "フォームで名前を入力してください。"
    number = 0
    return render_template("index.html",name = name,number = number)

@app.route('/',methods=["POST"])
def post():
    BASE_DIR = normpath(dirname(__file__))#このスクリプトがあるファイルの絶対パスを取得

    training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))

    dialogue_agent = DialogueAgent()
    dialogue_agent.train(training_data['text'], training_data['label']) #読み込んだ学習データのcsvファイルを辞書化している。

    with open(join(BASE_DIR, './replies.csv')) as f:  #返答用のcsvファイルの読み込み
        replies = f.read().split('\n') #読み込んだ返答用csvファイルを読み込み改行でlist化している。

    input_text = request.form["name"]
    predictions = dialogue_agent.predict([input_text])  # 予測した番号を返答している
    predicted_class_id = predictions[0]  #predictionsに入っている０番の数字を指定している。（情報は一つしか入っていないため０番固定）
    name = replies[predicted_class_id]
    number = predicted_class_id

    return render_template("index.html",name = name,number = number)
    #ここまで
   
if __name__ == "__main__":
    # webサーバー立ち上げ
    app.run()