# tamago-29225
アプリ名

返答するたまご

概要

フォームで入力した文に対して、一問一答で返答を行います。

本番環境

デプロイは未実施です。

制作背景

インターネットの普及により、様々な人と繋がりを持てるようにはなりましたが、
今感じている不安や心配事といった悩みをすぐに相談できる人が、身近にいる人というのはそう多くはないのではと思います。
私が社会人になってからの経験で、問題が解消しなくても話すことで楽になることもあると感じたことがあったことから、
現在すぐに話ができる人がいない人が少しでも気持ちが軽くなれるようなものを作りたいと思ったのが、今回のアプリケーションを作成したきっかけになります。

DEMO
![tamago](https://user-images.githubusercontent.com/69794984/95052187-7735cc00-0729-11eb-9408-c7b2e85c1e9d.gif)

工夫したポイント

意図せぬ返答がされないように、都度データの追加を行いました。
お腹が痛いとお腹が減ったといった似たような言葉で、同じ返答がされるのに対して、n_gramを使用することで解消しました。
        
使用言語

python-3.8.5

フレームワーク

Flask==1.1.2

使用モジュール一覧

Flask==1.1.2

mecab-python3==1.0.1

neologdn==0.4

numpy==1.19.2

pandas==1.1.2

scikit-learn==0.23.2

gunicorn==20.0.4

課題

現状の実装では、一問一答しか行えず、使用者の不安解消に至るまでの会話はできません。
また、学習データが足りていないため、予期せぬ入力に対して間違った返答を行うことがあリます。

課題に対して

これらの問題は、私にpythonと機械学習の知識と技術が足りていないため、現段階では改善できないが、
今後も知識を深め、その度に更新していくことで、いずれこのアプリを使用した人が、このアプリのおかげで明日も頑張れるとそう思えるようなものにしたいと考えています。

今回のアプリ作成を通じて

今回のアプリ作成の１番の目的は、人と会話ができるアプリケーションの基礎となる技術に触れてみることでした。
新しくpythonという言語や自然言語処理、機械学習というものに触れてみて、様々な可能性を垣間見ることができました。
そういったところで、新しい分野に一歩踏み出してみてよかったと感じます。
今後もこの分野の学習は継続して行い、また興味を持ったところにはとりあえず一歩踏みだすということを意識してプログラムを学んでいきたいと思います。


