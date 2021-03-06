## tamago-29225
# アプリ名  
返答するたまご  
  
# 概要  
フォームで入力した文に対して、一問一答で返答を行います。  

# 本番環境  
デプロイはしておりません。  

# 制作背景  
インターネットの普及により、様々な人と繋がりを持てるようにはなりました。  
ですが、今感じている不安や心配事といった悩みをすぐに相談できる人が、身近にいる人というのはそう多くはないと思います。  
私は「問題が解消しなくても人に話すだけで楽になることもある」ということを前職で体験しました。  
この経験から悩みや不安を抱えている時こそ、コミュニケーションをとることが必要だと考えます。  
ですが、会話をするには相手が必要です。  
そこで、もし人と同様に会話ができるアプリケーションがあれば、その環境にいない人でも同じ体験ができるのではないかと思いこのアプリケーションを作成しようと思いました。  


# DEMO  
![tamago](https://user-images.githubusercontent.com/69794984/95052187-7735cc00-0729-11eb-9408-c7b2e85c1e9d.gif)  
  
# 工夫したポイント  
意図せぬ返答がされないように、その都度データの追加を行いました。  
お腹が痛いとお腹が減ったといった似たような言葉で、同じ返答がされるのに対して、n_gramを使用することで解消しました。  
          
# 使用言語  
* python-3.8.5  
* フレームワーク  
* Flask==1.1.2  
* 使用モジュール一覧  
* Flask==1.1.2  
* mecab-python3==1.0.1  
* neologdn==0.4  
* numpy==1.19.2  
* pandas==1.1.2  
* scikit-learn==0.23.2  
* gunicorn==20.0.4  
  
# 課題  
現状の実装では、一問一答しか行えず、使用者の不安解消に至るまでの会話はできません。  
また、学習データが足りていないため、予期せぬ入力に対して間違った返答を行うことがあリます。  
  
# 課題に対して  
これらの問題は、私にpythonと機械学習の知識と技術が足りていないため、現段階では改善できません。  
今後も知識を深めてアップデートしていくことで、  
いずれこのアプリを使用した人が、このアプリのおかげで明日も頑張れるとそう思えるようなものにしたいと考えています。  
学習予定については、まずは機械学習について理解を深めたいと思っています。  
今回使用したSVMについても、これを使用することで機械学習を使用できるといった知識しかありませんので、  
どの機械学習を使用することでどんなことができるのか理解したいと考えています。  
  
# 今回のアプリ作成を通じて  
今回のアプリ作成の１番の目的は、人と会話ができるアプリケーションの基礎となる技術に触れてみることでした。  
新しくpythonという言語や自然言語処理、機械学習というものに触れてみて、様々な可能性を垣間見ることができました。   
今後もこの分野の学習は継続して行い、また興味を持ったところにはとりあえず一歩踏みだすということを意識してプログラムを学んでいきたいと思います。

# 画像使用
たまご：サイト名：illust AC  
url:https://www.ac-illust.com/main/detail.php?id=1037969&word=%E3%82%BF%E3%83%9E%E3%82%B4&searchId=78992954#comment-section  
素材名：タマゴ  
作者：作者: K-factory様  
  
背景：  サイト名：Sozai good  
url:https://sozai-good.com/illust/free-background/landscape-nature/7210  
素材名：草原 木 山 無料背景イラスト

