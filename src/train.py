from dataset import GET_RAW_DATA, My_DATASET
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
import argparse
import os

import wandb

class FT_BERT():
    def __init__(self, model, device, train_loder, val_loder, optimizer):
        self.model = model.to(device)
        self.device = device
        self.train_loder = train_loder
        self.val_loder = val_loder
        self.optimizer = optimizer

    def train(self):
        self.model.train()
        train_loss = 0
        for i, (input_ids, input_mask, labels) in enumerate(self.train_loder):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(input_ids, 
                        token_type_ids=None, 
                        attention_mask=input_mask, 
                        labels=labels)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            train_loss += loss.item()

            # print(f'step {i}')
        return train_loss / (i+1)

    def validation(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (input_ids, input_mask, labels) in enumerate(self.val_loder):
                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                labels = labels.to(self.device)
                output = self.model(input_ids,
                                token_type_ids=None, 
                                attention_mask=input_mask,
                                labels=labels)
                loss = output.loss
                val_loss += loss.item()
        return val_loss / (i+1)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    wandb.init(project=args.project)

    # データセットを取得
    grd = GET_RAW_DATA(args.data_path)
    sentence = grd.sentence
    label = grd.label
    dataset = My_DATASET(sentence, label, args.MODEL_NAME)
    
    # データセットを分割
    train_size = int(args.rate_train_val * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 訓練データローダー
    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), # ランダムにデータを取得してバッチ化
                batch_size = args.batch_size
            )
    # 検証データローダー
    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), # 順番にデータを取得してバッチ化
                batch_size = args.batch_size
            )

    # BertForSequenceClassification 
    model = BertForSequenceClassification.from_pretrained(
        args.MODEL_NAME, # 日本語Pre trainedモデルの指定
        num_labels = 3, # ラベル数（今回はBinayなので2、数値を増やせばマルチラベルも対応可）
        output_attentions = False, # アテンションベクトルを出力するか
        output_hidden_states = False, # 隠れ層を出力するか
    )
    # 最適化手法の設定
    optimizer = AdamW(model.parameters(), lr=2e-5)

    ft = FT_BERT(model, args.device, train_dataloader, validation_dataloader, optimizer)

    print('-------------------------------------------------------------')
    print('train sart')

    for epoch in range(args.epoch):
        train_loss = ft.train()
        val_loss = ft.validation()
        wandb.log({'epoch':epoch, 'train_loss':train_loss, 'val_loss':val_loss})
        print(f'epoch {epoch}')
        
        torch.save(ft.model.state_dict(), args.save_path + f'epoch={epoch}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='Polls on Twitter by BERT')
    parser.add_argument('--MODEL_NAME', default='cl-tohoku/bert-base-japanese-whole-word-masking', help='pretrained model')
    parser.add_argument('--data_path', default='../DATA/twitterJSA_data.pickle')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='MIG-2830f874-62b4-5f63-b39a-c4ae68478be0')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--rate_train_val', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--save_path', default='../save_model/')
    args = parser.parse_args()

    # モデルの保存先がないなら作成する。
    if ~os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    main(args)






