# Polls-on-Twitter-by-BERT

## Download Data
Twitter日本語評判分析データセット
https://www.db.info.gifu-u.ac.jp/sentiment_analysis/

以下のようにデータを配置
<pre>
.
├── src
├── DATA
    ├── tweets_open.csv
</pre>

以下を実行して、ツイートを取得し、pickleファイルを`DATA/`直下に作成する。
```
cd src
python tweetExtructor.py
```

## Fine Tuning
`src`で以下を実行すると、学習する。
```
python train.py
```
