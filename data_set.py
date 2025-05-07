import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
# 加载英日翻译数据集（启用远程代码信任）
dataset = load_dataset(
    "iwslt2017",
    "iwslt2017-en-ja",
    trust_remote_code=True
)

train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

src_lan = 'en'    # 源语言：英语
tgt_lan = 'ja'    # 目标语言：日语

# 使用spacy英语分词器，日本语用basic空格分
token_transform = {
    src_lan: get_tokenizer('spacy', language='en_core_web_sm'),
    tgt_lan: lambda x: list(x)  # 简单处理日语（每个字分开）
}

def yield_tokens(data_iter, language):
    for item in data_iter:
        yield token_transform[language](item["translation"][language])

#将训练集中的每个句子分成每一个词，再将每一个词都对应一个数字
# 构建源语言（英语）词表
#注意看，是从训练数据集来的
SRC_VOCAB = build_vocab_from_iterator(yield_tokens(train_data, src_lan), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
SRC_VOCAB.set_default_index(SRC_VOCAB["<unk>"])

# 构建目标语言（日语）词表
TGT_VOCAB = build_vocab_from_iterator(yield_tokens(train_data, tgt_lan), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
TGT_VOCAB.set_default_index(TGT_VOCAB["<unk>"])

def tensor_transform(token_ids, vocab):
    return torch.cat((
        torch.tensor([vocab['<bos>']],dtype=torch.long),
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor([vocab['<eos>']],dtype=torch.long)
    ))

#将句子转化为一串数字
text_transform = {
    src_lan : lambda x : [SRC_VOCAB[token] for token in token_transform[src_lan](x)],
    tgt_lan : lambda x : [TGT_VOCAB[token] for token in token_transform[tgt_lan](x)]
}

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_text = item["translation"][src_lan]
        tgt_text = item["translation"][tgt_lan]
        # 一串数字再转为tensor
        src_tokens = tensor_transform(text_transform[src_lan](src_text), SRC_VOCAB)
        tgt_tokens = tensor_transform(text_transform[tgt_lan](tgt_text), TGT_VOCAB)

        src_batch.append(src_tokens)
        tgt_batch.append(tgt_tokens)

    src_batch = pad_sequence(src_batch, padding_value=SRC_VOCAB["<pad>"])
    tgt_batch = pad_sequence(tgt_batch, padding_value=TGT_VOCAB["<pad>"])
    return src_batch, tgt_batch

BATCH_SIZE = 32

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)



import torch.nn as nn

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers,
                 emb_size, nhead, src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        #将数字序列tensor加上位置编码
        #encoder的体现
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)
    #forward返回的是最终结果

    def encode(self, src, src_key_padding_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, mask=None, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory,
                                        tgt_mask)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

def generate_square_subsequent_mask(sz, device):
    """
    返回 [sz, sz] 上三角矩阵，用于 Decoder 防止看到未来的信息。
    对角线及以下为 0，对角线以上为 -inf。
    """
    mask = torch.triu(torch.ones((sz, sz), device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def create_mask(src, tgt_input, src_pad_idx, tgt_pad_idx):
    """
    根据 src 和 tgt_input，创建 Transformer 所需的所有 mask。
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt_input.shape[0]
    device = src.device

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)

    src_padding_mask = (src == src_pad_idx).transpose(0, 1)  # [batch_size, src_seq_len]
    tgt_padding_mask = (tgt_input == tgt_pad_idx).transpose(0, 1)  # [batch_size, tgt_seq_len]

    return None, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask

import os
SRC_VOCAB_SIZE = len(SRC_VOCAB)
TGT_VOCAB_SIZE = len(TGT_VOCAB)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
                           SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
model = model.to(device)

PAD_IDX = TGT_VOCAB['<pad>']
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

checkpoint_path = "checkpoint.pth"
start_epoch = 1

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ 加载 checkpoint 成功，恢复训练从第 {start_epoch} 轮开始")
else:
    print("🆕 未发现 checkpoint，从头开始训练")


def train_epoch(model, optimizer):
    model.train()#开启训练模式
    total_loss = 0
    for src, tgt in train_iter:
        #src和tgt为tensor
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]#去掉eos，要传入decoder的，(虽说是整个句子传入，但是会有mask制约)
        targets = tgt[1:, :].reshape(-1)#去掉bos，并变成一维向量

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask = create_mask(
            src, tgt_input, SRC_VOCAB['<pad>'], TGT_VOCAB['<pad>']
        )
        #生成各种mask
        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        # 得到每个位置每个句子对应词表的概率，模型的最终运行结果(未经过softmax)

        optimizer.zero_grad()
        loss = criterion(logits.reshape(-1, logits.shape[-1]), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_iter)

def evaluate(model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in val_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]
            targets = tgt[1:, :].reshape(-1)#实际上有多个句子(具体来说有batch size个)

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask = create_mask(
                src, tgt_input, SRC_VOCAB['<pad>'], TGT_VOCAB['<pad>']
            )

            logits = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, memory_key_padding_mask)

            loss = criterion(logits.reshape(-1, logits.shape[-1]), targets)
            #将三维的logits降为2维的
            total_loss += loss.item()
    return total_loss / len(val_iter)


NUM_EPOCHS = 10

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    train_loss = train_epoch(model, optimizer)
    val_loss = evaluate(model)

    print(f"📘 Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 保存 checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"已保存第 {epoch} 轮的 checkpoint")

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 确保下载了 nltk 的分词工具
nltk.download('punkt')

def greedy_decode(model, src, src_key_padding_mask, max_len, start_symbol):
    src = src.to(device)
    memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        next_word = torch.argmax(prob, dim=1).item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == TGT_VOCAB['<eos>']:
            break

    return ys


def tokens_to_sentence(tokens, vocab):
    itos = list(vocab.get_itos())  # 获取索引到词的列表
    words = []
    for token in tokens:
        word = itos[token]
        if word in ['<bos>', '<pad>']:
            continue
        if word == '<eos>':
            break
        words.append(word)
    return words




def compute_bleu_score(model, data_iter, num_samples=100):
    model.eval()
    total_bleu = 0.0
    smooth_fn = SmoothingFunction().method1

    count = 0
    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        for i in range(src.shape[1]):  # batch 维度
            src_sent = src[:, i].unsqueeze(1)
            tgt_sent = tgt[1:, i]  # 去掉 <bos>

            src_key_padding_mask = (src_sent.squeeze(1) == SRC_VOCAB['<pad>']).unsqueeze(0)
            pred_tokens = greedy_decode(
                model, src_sent, src_key_padding_mask=src_key_padding_mask,
                max_len=50, start_symbol=TGT_VOCAB['<bos>']
            )
            pred_sentence = tokens_to_sentence(pred_tokens.flatten().cpu().numpy(), TGT_VOCAB)
            tgt_sentence = tokens_to_sentence(tgt_sent.cpu().numpy(), TGT_VOCAB)

            bleu = sentence_bleu([tgt_sentence], pred_sentence, smoothing_function=smooth_fn)
            total_bleu += bleu
            count += 1

            if count >= num_samples:
                break
        if count >= num_samples:
            break

    avg_bleu = total_bleu / count
    print(f"Average BLEU score on {count} samples: {avg_bleu:.4f}")
    return avg_bleu

compute_bleu_score(model, val_iter, num_samples=100)
