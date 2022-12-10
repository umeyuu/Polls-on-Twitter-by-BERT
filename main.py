from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler
from src.serch_tweet import get_tweets
from src.dataset import My_DATASET
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import os

def id2label(id):
    if id == 0:
        return 'pos'
    elif id == 1:
        return 'neg'
    elif id == 2:
        return 'neu'

# 円グラフを作成する
def show_pi(pred, serch_word, save_result):
    label = []
    count = []
    for k, v in pred.items():
        label.append(k)
        count.append(v)
    plt.pie(count, labels=label, autopct='%.1f%%')
    plt.title(f'Amount of Data is {sum(count)}')
    plt.savefig(f'{save_result}result_of_{serch_word}.png')


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    # モデルを読み込む
    model = BertForSequenceClassification.from_pretrained(
            args.MODEL_NAME, # 日本語Pre trainedモデルの指定
            num_labels = 3, # ラベル数
            output_attentions = False, # アテンションベクトルを出力するか
            output_hidden_states = False, # 隠れ層を出力するか
        )
    model.load_state_dict(torch.load(args.model_path))

    # APIで検索ワードを含むツイートを取得する
    if not os.path.isfile(args.savedir+args.serch_word+'.csv'):
        tweets = get_tweets(args.serch_word, args.min_faves, args.num_tweet, args.savedir)
    else:
        df = pd.read_csv(args.savedir+args.serch_word+'.csv')
        df.columns = ['tweet']
        tweets = df.tweet.values.tolist()

    # データローダー
    dataset = My_DATASET(args.MODEL_NAME, tweets)
    tweet_loader = DataLoader(
                dataset, 
                sampler = SequentialSampler(dataset), # 順番にデータを取得してバッチ化
                batch_size = 1
            )
    model = model.to(args.device)

    pred = {'pos':0, 'neg':0, 'neu':0}

    # Titterで取得したデータをモデルで推論する
    for input_ids, input_mask in tweet_loader:
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)

        output = model(input_ids,
                    token_type_ids=None, 
                    attention_mask=input_mask)

        pred_id = output.logits.argmax(dim=1).item()
        pred_label = id2label(pred_id)
        pred[pred_label] += 1

    # 推論結果を円グラフで表示する
    show_pi(pred, args.serch_word, args.save_result)

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_NAME', default='cl-tohoku/bert-base-japanese-whole-word-masking', help='pretrained model')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='MIG-4bc08680-2516-5102-acd7-f4ce7f6e56e9')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model_path', default='save_model/best_model.pth')
    parser.add_argument('--serch_word', default='サッカー　W杯', help='検索ワード')
    parser.add_argument('--min_faves', type=int, default=100, help='何いいね以上のツイートを取得するか')
    parser.add_argument('--num_tweet', type=int, default=150, help='何ツイート取得するか')
    parser.add_argument('--savedir', default='DATA/serched_tweet/')
    parser.add_argument('--save_result', default='Result/')
    args = parser.parse_args()

    # ツイートの保存先がないなら作成する。
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)
    # 結果の保存先がないなら作成する。
    if not os.path.isdir(args.save_result):
        os.makedirs(args.save_result)
    
    main(args)

  






