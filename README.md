# Polls-on-Twitter-by-BERT
Twitterであるテーマについての意見を集計し、ネガポジ判定モデルを用いて世論調査をしてみた！

ネガポジ判定モデルはBERTをファインチューニングしたモデルを使用。





## Setup
Twitter APIを使用するために申請が必要。

キーを取得したら、`src/setting.py`の **** 部分に書き込んでください。
```
CK = '****' # Consumer Key
CS = '****' # Consumer Secret
AT = '****' # Access Token
AS = '****' # Accesss Token Secert
```

## Download Data
ファインチューニングはTwitter日本語評判分析データセットで行う。

https://www.db.info.gifu-u.ac.jp/sentiment_analysis/

上記のurlで得られるデータを以下のように配置済み。
<pre>
.
├── src
├── DATA
    ├── tweets_open.csv
</pre>

以下を実行して、ツイートを取得し、pickleファイルを`DATA/`直下に作成する。
```
python src/tweetExtructor.py
```

## Fine Tuning
以下を実行すると、ファインチューニングを実行する。

`--CUDA_VISIBLE_DEVICES`と`--device`は個人の環境に合わせて変更してください。
```
cd src
python train.py
```

## Execution
以下を実行すると、検索ワードのツイートを集計し、`Result`直下に円グラフ画像を出力する。
```
python main.py --serch_word hoge    
```
