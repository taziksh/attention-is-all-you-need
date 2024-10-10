from datasets import load_dataset
from transformers import AutoTokenizer

import torch
import math
import os

import wandb

num_epochs = 10

wandb.login()
wandb.init(
    project="attention-is-all-you-need",
    config={
        "epochs": num_epochs
    }
)

#NOTE: they used train split, wiht ~3.5M example
# ds = load_dataset("wmt/wmt14", "de-en", split='test')[:2]#'train')
# training_data = ds['translation']

training_data = load_dataset("wompzik/lambada", split="train[:100]")

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

num_layers = 6
seq_len = 512
d_model = 512

vocab_dim = 50257+1

#TODO: change anywhere with (1, ) to support (batch_size, )

class FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()            
        
        self.d_ff = d_model*4
        self.W_1 = torch.nn.Linear(d_model, self.d_ff)
        self.relu = torch.nn.ReLU()
        self.W_2 = torch.nn.Linear(self.d_ff, d_model)
        
    def forward(self, x):
        # assert x.shape == (seq_len, d_model)    
        out = self.W_1(x)
        out = self.relu(out)
        out = self.W_2(out)

        return out

class MHA(torch.nn.Module):
    def __init__(self, h=8, has_mask=False):  
        super().__init__()          
        
        self.has_mask = has_mask
        self.d_k = d_model // h

        self.d_v = self.d_k        
        
        self.scale = 1/math.sqrt(self.d_k)

        assert d_model == 512
        assert self.d_k == 64
        assert self.d_v == self.d_k

        self.W_Q = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(size=(d_model, self.d_k))))
        self.W_K = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(size=(d_model, self.d_k))))
        self.W_V = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(size=(d_model, self.d_v))))
        self.W_O = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros(size=(h*self.d_v, d_model))))
                
    def attn(self, x_q, x_k, attention_mask):
        Q = torch.einsum('bcd,dk->bck', x_q, self.W_Q)
        K = torch.einsum('bcd,dk->bck', x_k, self.W_K)
        V = torch.einsum('bcd,dv->bcv', x_k, self.W_V)

        causal_mask = torch.ones(1, seq_len, seq_len).to(device) 
        if self.has_mask:
            causal_mask = torch.tril(causal_mask)

        attention_mask = torch.unsqueeze(attention_mask, dim=1)
        causal_mask = causal_mask * attention_mask
        
        # softmax along dim=-1 of Q@K.T => seq_len
        scores = self.scale*Q@K.transpose(-2,-1) # bcd,bdc -> bcc
        scores = scores.masked_fill_(causal_mask == 0, -float('inf'))
        sm = torch.softmax(scores, dim=-1)
        head = sm @ V
        return head

    def forward(self, x_q, x_k, attention_mask):
        assert x_q.shape == (1, seq_len, d_model)
        assert x_k.shape == (1, seq_len, d_model)
        
        heads = torch.cat([self.attn(x_q, x_k, attention_mask) for _ in range(8)], dim=-1)
        res = heads @ self.W_O
        return res

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.M = MHA(has_mask=False)
        self.F = FFN()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.LN = torch.nn.LayerNorm(d_model)
        
    def forward(self, x, attention_mask):
        mha = self.M(x, x, attention_mask)
        mha = self.dropout(mha)

        sl1 = self.LN(x+mha)

        ffn = self.F(sl1)
        ffn = self.dropout(ffn)

        sl2 = self.LN(sl1 + ffn)

        return sl2

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.M1 = MHA(has_mask=True)
        self.M2 = MHA(has_mask=False)
        self.F = FFN()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.LN = torch.nn.LayerNorm(d_model)        
    
    def forward(self, x, enc_out, attention_mask):
        mmha = self.M1(x, x, attention_mask)
        mmha = self.dropout(mmha)

        sl1 = self.LN(x + mmha)

        mha = self.M2(x_q=enc_out, x_k=sl1, attention_mask=attention_mask)
        mha = self.dropout(mha)
        
        sl2 = self.LN(sl1 + mha)

        ffn = self.F(sl2)
        ffn = self.dropout(ffn)

        sl3 = self.LN(sl2 + ffn)

        return sl3

class EncoderDecoder(torch.nn.Module):      
    def __init__(self):
        super().__init__()        
        self.encs = torch.nn.ModuleList([Encoder() for _ in range(num_layers)])
        self.decs = torch.nn.ModuleList([Decoder() for _ in range(num_layers)])
        self.pos_embed = PosEmbed()
        self.dropout = torch.nn.Dropout(p=0.1)        

    def forward(self, x, attention_mask):
        enc_out = x + self.pos_embed.fwd()
        for enc in self.encs:
            enc_out = enc(enc_out, attention_mask)

        dec_out = enc_out + self.pos_embed.fwd()
        dec_out = self.dropout(dec_out)        
        
        for dec in self.decs:
            dec_out = dec(enc_out, dec_out, attention_mask)
            
        return dec_out   

class Embed(torch.nn.Module):
    def __init__(self, vocab_dim, d_model):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=vocab_dim, embedding_dim=d_model, padding_idx=50257)

    def forward(self, x):
        res = self.emb(x)
        return res

class PosEmbed():
    def __init__(self):
        self.emb = torch.zeros(seq_len, d_model).to(device)

        for pos in range(seq_len):
            for i in range(d_model//2):
                arg = pos/(10000**(2*i/d_model))
                if i % 2 == 0:
                    self.emb[pos, 2*i] = math.sin(arg)
                else:
                    self.emb[pos, 2*i+1] = math.cos(arg)

    def fwd(self):   
        return self.emb

class Transformer(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.input_embed = Embed(vocab_dim=vocab_dim, d_model=d_model)
       self.enc_dec = EncoderDecoder()       
       self.linear = torch.nn.Linear(d_model, vocab_dim) # (seq_len, d_model) -> (seq_len, vocab_dim)
       self.pos_embed = PosEmbed()
       self.dropout = torch.nn.Dropout(p=0.1) 
        
    def forward(self, input_ids, attention_mask):
       #Nit: torch.nn.Sequential

       #TODO: I'm not using attention_mask here 
       assert input_ids.shape == (1, seq_len)

       out = self.input_embed(input_ids)

       assert out.shape == (1, seq_len, d_model) # embedding  

       out = self.dropout(out)
    
       out = self.enc_dec(out, attention_mask)  

       assert out.shape == (1, seq_len, d_model) 

       out = self.linear(out)

       assert out.shape == (1, seq_len, vocab_dim)

       # Softmax over dim=1, seq_len, for every position
       softmax = torch.softmax(input=out, dim=1) 
       return softmax


model = Transformer().to(device)

#Nit: tokenizer fork warning
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

criterion = torch.nn.CrossEntropyLoss()
losses = []

#TODO: scheduler
# warmup_steps = 4000
# step_num = 1
# lr = d_model**(-0.5)*min(step_num**(-0.5), step_num * warmup_steps**(-1.5))
optim = torch.optim.Adam(params=model.parameters(), betas=(0.9,0.98), eps=10E-9)#, lr=lr)

for epoch in range(num_epochs):
    for sample in training_data:
        optim.zero_grad()

        input_tokens = tokenizer(sample['until_last'], truncation=True, padding='max_length', max_length=seq_len, return_tensors="pt").to(device)
        target_tokens = tokenizer(sample['last_word'], truncation=True, padding='max_length', max_length=seq_len, return_tensors="pt").to(device)

        output_tokens = model(**input_tokens)

        loss = criterion(output_tokens.transpose(-2,-1), target_tokens['input_ids'])
        losses.append(loss)
        wandb.log({"loss": loss})

        loss.backward()        
        optim.step()
        
    if epoch % 2 == 0:
        print(f"loss: {loss} @ epoch: {epoch}")
