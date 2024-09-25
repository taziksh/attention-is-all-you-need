from datasets import load_dataset
from transformers import AutoTokenizer

import torch
import math
import os

import wandb

wandb.login()

num_epochs = 10

wandb.init(
    project="attention-is-all-you-need",
    config={
        "epochs": num_epochs
    }
)

#NOTE: they used train split, wiht ~3.5M examples
ds = load_dataset("wmt/wmt14", "de-en", split='train')
training_data = ds['translation']

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

num_layers = 6
seq_len = 512
d_model = 512

vocab_dim = 50257+1

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
                
    #TODO: why separate x_k and x_v
    def attn(self, x_k, x_v):
        Q = x_k @ self.W_Q
        K = x_k @ self.W_K
        V = x_v @ self.W_V

        mask = torch.ones(seq_len, seq_len).to(device) 
        if self.has_mask:
            mask = torch.tril(mask)
        
        # softmax along dim=0 of Q@K.T => seq_len
        sm = torch.softmax(input=self.scale*mask*(Q@K.T), dim=0)
        head = sm @ V
        return head

    def forward(self, x_k, x_v):
        assert x_k.shape == (seq_len, d_model)
        assert x_v.shape == (seq_len, d_model)
        
        heads = torch.cat([self.attn(x_k, x_v) for _ in range(8)], dim=1)
        res = heads @ self.W_O
        return res


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

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.M = MHA(has_mask=False)
        self.F = FFN()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.LN = torch.nn.LayerNorm(d_model)
        
    def forward(self, x):
        mha = self.M(x_k=x, x_v=x)
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
    
    def forward(self, x, enc_out):
        mmha = self.M1(x_k=x, x_v=x)
        mmha = self.dropout(mmha)

        sl1 = self.LN(x + mmha)

        mha = self.M2(x_k=enc_out, x_v=sl1)
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

    def forward(self, x):
        enc_out = x
        for enc in self.encs:
            enc_out = enc(enc_out)

        dec_out = enc_out + self.pos_embed.fwd()
        dec_out = self.dropout(dec_out)        
        
        for dec in self.decs:
            dec_out = dec(enc_out, dec_out)
            
        return dec_out   

class Transformer(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.input_embed = Embed(vocab_dim=vocab_dim, d_model=d_model)
       self.enc_dec = EncoderDecoder()       
       self.linear = torch.nn.Linear(d_model, vocab_dim) # (seq_len, d_model) -> (seq_len, vocab_dim)
       self.pos_embed = PosEmbed()
       self.dropout = torch.nn.Dropout(p=0.1) 
        
    def forward(self, x):
       #Nit: torch.nn.Sequential
        
       #TODO: fix this assert
    #    print(x.shape) 
       # assert x.shape == (seq_len)
        
       out = self.input_embed(x) + self.pos_embed.fwd()
       assert out.shape == (seq_len, d_model) # embedding  

        
       out = self.dropout(out)
        
       out = self.enc_dec(out)     
       out = self.linear(out)

       assert out.shape == (seq_len, vocab_dim) 

       # Softmax over dim=0, seq_len
       softmax = torch.softmax(input=out, dim=0) 
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
    print(f"epoch: {epoch}")
    for sample in training_data:
        optim.zero_grad()

        tokenized_de = tokenizer.encode(sample['de'], padding='max_length', max_length=seq_len)
        tokenized_en = tokenizer.encode(sample['en'], padding='max_length', max_length=seq_len)
        
        targets = torch.tensor(tokenized_en).to(device)
        inputs = torch.tensor(tokenized_de).to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        losses.append(loss)
        wandb.log({"loss": loss})

        loss.backward()        
        optim.step()
        
    if epoch % 2 == 0:
        print(f"loss: {loss}")
