# -*- encoding: utf-8 -*-
'''
@Time    :   2021/11/12 13:34:23
@Author  :   流氓兔233333 
@Version :   1.0
@Contact :   分类后处理办法CAN (TNEWS 今日头条中文新闻（短文本）分类)
'''

'''
100 民生 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
105 时政 新时代 nineteenth
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu 
109 科技 科技 news_tech
110 军事 军事 news_military
111 宗教 无，凤凰佛教等来源
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
'''

from typing import Counter
from numpy.core.fromnumeric import shape
from numpy.lib.twodim_base import diag
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

data_path = './data_raw/'
save_path = './temp_results/'

id2label = {int(k):i for i,k in enumerate([100,101,102,103,104,106,107,108,109,\
                110,112,113,114,115,116])}
tar2id = {}
tar2nums = {}
data = []
with open('./data_raw/TNEWS.txt', 'r', encoding='utf-8') as f:
    for text in tqdm(f.readlines()):
        text = text.split('_!_')[1:]
        tar2id[int(text[0])] = tar2id.get(int(text[0]), 0)
        tar2id[int(text[0])] = text[1]

        # 每个类计数
        tar2nums[int(text[0])] = tar2nums.get(int(text[0]), 0)
        tar2nums[int(text[0])] += 1 

        # label, context
        data.append([id2label[int(text[0])], text[2]+text[3]])



len(tar2id), len(id2label)
np.array(data).shape
# ['100', '114'] 数量特别少
np.array([x for _,x in tar2nums.items()]) / sum([x for _,x in tar2nums.items()])



# Robert 分词器
from transformers import BertTokenizer, BertModel
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import torch
import random
import os, pickle
import transformers

# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cached_path=cache_path)

# set seed
def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(23333)


# train test split
def train_test_split(data, trn_size=0.8):
    data_trn, data_val = [], []
    count = 0
    for _,v in tar2nums.items():
        y = data[count:count+v][:1000]
        random.shuffle(y)
        data_trn += y[:int(trn_size*len(y))]
        data_val += y[int(trn_size*len(y)):]
        count += v
    return data_trn, data_val


data_trn, data_val = train_test_split(data, trn_size=0.8)
np.array(data_trn).shape, np.array(data_val).shape


def distribution_plot_TextLength(mark_lens):
    # 评论长度分布图：用于确定 过长文本 and 分词的 max_len
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    sns.distplot(mark_lens, fit=stats.norm, color='g')  # 正太概率密度 / 核密度估计图
    plt.tick_params(labelsize=15)
    plt.show()

text_len = [len(x) for _,x in data]
distribution_plot_TextLength(text_len)


class CustomDataset(Data.Dataset):
    def __init__(self, data, maxlen, with_labels=True):
        self.data = data  # pandas dataframe

        #Initialize the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('./RoBerta/')  
        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        # 根据自己输入data的格式修改
        context = self.data[index][1]
        label = self.data[index][0]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(context, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,       # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            # print(token_ids.shape)
            label = torch.tensor(label)
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids


batch_size = 32
dataset_trn = CustomDataset(data_trn, 64)
loader_trn = Data.DataLoader(dataset_trn, batch_size, False)

dataset_val = CustomDataset(data_val, 64)
loader_val = Data.DataLoader(dataset_val, batch_size, False)





# model Roberta
class MyModel(nn.Module):
    def __init__(self, freeze_bert=False, hidden_size=768):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained('./RoBerta/',
                    output_hidden_states=True, return_dict=True)
        self.fc = nn.Linear(hidden_size, len(tar2id))
        
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        
    def forward(self, input_id, attn_masks, token_type):
        # bert_out (last_hidden_states, pool, all_hidden_states)
        # last_hidden_states [bs, seq_len, hid_size=768]
        # pool [bs, hid_size=768]
        # all_hidden_states (embedding, 各层的hidden_states, ....)
        outputs = self.bert(input_ids=input_id, token_type_ids=token_type, attention_mask=attn_masks) 
        last_hidden_states = outputs.last_hidden_state # [bs, seq_len, hid_dim]

        out = self.fc(last_hidden_states[:,0, :]) # 取出 [CLS] Token  [bs, hidden-size]

        return out


model = MyModel()
optimizer = optim.Adam(lr=1e-5, params=model.parameters(), weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    },'./model_RoBerta.pth')
    print('The best model has been saved')



def train_eval(model, criterion, optimizer, loader_trn, loader_val, epochs=2, continue_=True):
    if continue_:
        try:
            checkpoint = torch.load('./model_RoBerta.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.to(device)
            print('-----Continue Training-----')
        except:
            print('No Pretrained model!')
            print('-----Training-----')
    else:
        print('-----Training-----')
    
    model = model.to(device)
    loss_his = []
    for epoch in range(epochs):
        model.train()
        print('eopch: %d/%d'% (epoch, epochs))
        for _, batch in enumerate(tqdm(loader_trn)):
            optimizer.zero_grad()
            
            batch = [x.to(device) for x in batch]
            output = model(batch[0], batch[1], batch[2])
            loss = criterion(output, batch[-1])
            loss_his.append(loss.item())

            loss.backward()
            optimizer.step()
            
        if epoch % 1 == 0:
            print(loss.item())
            eval(model, optimizer, loader_val)
    
    return loss_his

best_score = 0.0

def eval(model, optimizer, loader_val):
    model.eval()
    output_all = []
    real_all = []
    
    with torch.no_grad():
        for _, batch in enumerate(tqdm(loader_val)):
            batch = [x.to(device) for x in batch]
            logits = model(batch[0], batch[1], batch[2])
            logits = logits.detach().cpu().numpy()
            pred = np.argmax(logits, axis=1).flatten()
            
            label_ids = batch[-1].detach().cpu().numpy()
            real_all = real_all + list(label_ids)
            output_all = output_all + list(pred)
    
    
    acc_val = accuracy_score(real_all, output_all)
    
    print("Validation Accuracy: {}".format(acc_val))
    global best_score
    if best_score < acc_val:
        best_score = acc_val
        save(model, optimizer)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel()
optimizer = optim.Adam(lr=1e-5, params=model.parameters(), weight_decay=1e-4)

def test(model, loader_val):
    checkpoint = torch.load(save_path+'./model_RoBerta.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    output_all = []
    real_all = []

    
    # batch = next(iter(loader_val))
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader_val)):
            batch = [x.to(device) for x in batch]
            logits = model(batch[0], batch[1], batch[2])
            logits = logits.softmax(1)
            logits = logits.detach().cpu().numpy()
            logits.shape
            
            if i==0:
                prob_all = logits
            else:
                prob_all = np.concatenate((prob_all, logits), axis=0)
            
            pred = np.argmax(logits, axis=1).flatten()
            
            label_ids = batch[-1].detach().cpu().numpy()
            real_all = real_all + list(label_ids)
            output_all = output_all + list(pred)
            
    acc_val = accuracy_score(real_all, output_all)
    return acc_val, real_all, output_all, prob_all

# train 
# loss_his = train_eval(model, criterion, optimizer, loader_trn, loader_val, epochs=1, continue_=True)


# ===========================================================================
# CAN实验 加载与训练好的 RoBerta model
# train test split
data_trn, data_val = train_test_split(data, trn_size=0.5)
np.array(data_trn).shape, np.array(data_val).shape

# data loader
batch_size = 32
dataset_val = CustomDataset(data_val, 64)
loader_val = Data.DataLoader(dataset_val, batch_size, False)

# test
acc_val, real_all, output_all, prob_all = test(model, loader_val)



# 修正前准确率
print('original acc: %f'%acc_val)


# 评价每个预测结果不确定性
k = 3
y_pred_topk = np.sort(prob_all, axis=1)[:, -k:]
y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True) # 归一化
y_pred_uncerrtainty = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)  # 计算预测的不确定性

# 选择阈值划分低/高 不确定性
threshold = 0.8
y_pred_confidenct = prob_all[y_pred_uncerrtainty < threshold]
y_pred_unconfidenct = prob_all[y_pred_uncerrtainty >= threshold]
print(y_pred_confidenct.shape), print(y_pred_unconfidenct.shape)


y_true_confidenct = np.array(real_all)[y_pred_uncerrtainty < threshold]
y_true_unconfidenct = np.array(real_all)[y_pred_uncerrtainty >= threshold]
print(y_true_confidenct.shape), print(y_true_unconfidenct.shape)


# 显示两部分各自的准确率
acc_confident = (y_pred_confidenct.argmax(1) == y_true_confidenct).mean()
acc_unconfident = (y_pred_unconfidenct.argmax(1) == y_true_unconfidenct).mean()
print('confident acc: %f'%acc_confident)
print('unconfident acc: %f'%acc_unconfident)


# 从训练集统计先验分布
prior = np.array([x for _,x in tar2nums.items()]) / sum([x for _,x in tar2nums.items()])

# 逐个修改地址新都样本，被重新评价acc
right, iters = 0, 1
# i, y = next(enumerate(y_pred_unconfidenct))

for i, y in enumerate(tqdm(y_pred_unconfidenct)):
    Y = np.concatenate([y_pred_confidenct, y[None]], axis=0)
    for j in range(iters):
        # 列归一化
        Y /= Y.sum(axis=0, keepdims=True)
        # Y = (np.diag(1/(Y.dot(np.diag(prior)).sum(axis=1))).dot(Y)).dot(np.diag(prior))
        Y *= prior[None]
        Y /= Y.sum(axis=1, keepdims=True)
    y = Y[-1]
    if y.argmax() == y_true_unconfidenct[i]:
        right += 1


# 输出修正后的准确率
acc_final = (acc_confident*len(y_pred_confidenct)+right) / len(prob_all)
print('new unconfident acc: %f' % (right/(i+1)))
print('final acc: %f' % (acc_final))



from collections import Counter
import bisect
bisect.bisect_left([2,3,4,6], )





