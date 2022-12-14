import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from transformers import BertJapaneseTokenizer

# pickleファイルから、学習に用いるデータをサンプリングするクラス
class GET_RAW_DATA():
    def __init__(self, path):
        super().__init__()
        df = self.load_pickle(path)
        df = self.make_label_row(df, 'pos')
        df = self.make_label_row(df, 'neg')
        df = self.make_label_row(df, 'neu')

        df_dict = self.slice_df(df)
        self.length = len(df_dict['pos'])

        self.sentence = [] # ツイート文章
        self.label = [] # pos, neg, neuのラベル
        for mode, df in df_dict.items():
            sent, la = self.extract_text_label(df, mode)
            self.sentence.extend(sent)
            self.label.extend(la)
    
    # pickleファイルを読み込む
    def load_pickle(self, path):
        data = pd.read_pickle(path)
        df = pd.DataFrame(data)
        df = df[df.text != '']
        return df

    def get_mode_id(self, mode):
        if mode == 'pos':
            id = 0
        elif mode == 'neg':
            id = 1
        elif mode == 'neu':
            id = 2
        else:
            print('select mode')
        return id
    
    # 新しい列を作成
    def make_label_row(self, df, mode):
        id = self.get_mode_id(mode)
        label_func = lambda x: 1 if x[id+1]==1 else 0
        df[mode] = df.label.apply(label_func)
        return df
    
    # 完全にpositive, negative, neutralのデータを取得。
    # たまにpositiveかつnegativeのデータがある
    def slice_df(self, df):
        df_pos = df[(df.pos==1) & (df.neg==0) & (df.neu==0)]
        df_neg = df[(df.pos==0) & (df.neg==1) & (df.neu==0)]
        df_neu = df[(df.pos==0) & (df.neg==0) & (df.neu==1)]
        df_dict = {'pos':df_pos, 'neg':df_neg, 'neu':df_neu}
        return df_dict

    # pos, neg, neuを均衡にする
    def extract_text_label(self, df, mode):
        sentence = df.text.values.tolist()
        id = self.get_mode_id(mode)
        if mode in ['neg', 'neu']:
            sentence = random.sample(sentence, self.length)
        label = [id] * self.length

        return sentence, label


# 学習データセット
class My_DATASET(Dataset):
    def __init__(self, model_name, sentence, label=None):
        super().__init__()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        # 文章の最大の単語数
        self.maxlen = self.check_maxlen(sentence)
        # 単語を辞書idに変換
        self.input_ids, self.attention_masks, self.labels = self.get_token_id(sentence, label)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        if self.labels == None:
            return self.input_ids[index], self.attention_masks[index]
        else:
            return self.input_ids[index], self.attention_masks[index], self.labels[index]

    def check_maxlen(self, sentence):
        max_len = []
        for sent in sentence:
            # Tokenizeで分割
            token_words = self.tokenizer.tokenize(sent)
            max_len.append(len(token_words))

        return max(max_len)+2

    def get_token_id(self, sentence, label):
        input_ids = []
        attention_masks = []

        for sent in sentence:
            encoded_dict = self.tokenizer.encode_plus(
                sent,                      
                add_special_tokens = True, # Special Tokenの追加
                max_length = self.maxlen, # 文章の長さを固定
                pad_to_max_length = True, # PADDINGで埋める
                return_attention_mask = True,   # Attention maksの作成
                return_tensors = 'pt',     #  Pytorch tensorsで返す
            )

            # 単語IDを取得    
            input_ids.append(encoded_dict['input_ids'])

            # Attention　maskの取得
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        if label != None:
            labels = torch.tensor(label)
            return input_ids, attention_masks, labels
        else:
            return input_ids, attention_masks, label


        




