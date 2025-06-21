import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size=64
block_size=264
max_iters=5000
eval_interval=500
learning_rate=3e-4
device="cuda" if torch.cuda.is_available() else "cpu"
eval_iters=200
n_embed=384
#-------------------
torch.manual_seed(1337)

# I downloaded input.txt from this guys github karpathy
with open("input.txt","r", encoding="utf-8") as f:      #encoding='utf-8'since python stores text as bytes, it needs to be able to interpret them and not make mistakes
    text=f.read()                                       #this encoding is universal and supports diferent languages

#unique characters in the text
chars=sorted(list(set(text)))    #a list of the set (take all the unique values from the text, and set already removes dupplicates) of text to prepare the data
vocab_size=len(chars)     #this is our vocabulary  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz65
#print("".join(chars))
#print(vocab_size)

#building the encoder and decoder (mapping from characters to int)
stoi={ch:i for i,ch in enumerate(chars) }   #string-to-integers (we build a dictionary that matches characters that we got before into numbers )
itos={i:ch for i,ch in enumerate(chars) }   #index-to-string (does the same but with integers to match the characters so we can reverse it)
encode=lambda s: [stoi[c] for c in s]       #fast function lambda to get the numbers out of a text
decode=lambda l:"".join([itos[i] for i in l])   #same but with the integers it makes text

#print(encode("Hello world, and Nil"))
#print(decode(encode("Hello world, and Nil")))

#we could do it with tiktoken
#import tiktoken
#enc=tiktoken.get_encoding("gpt2")
#assert enc.decode(enc.encode("hello world")) == "hello world"

#enconde the entire dataset and store it in torch.tensor
data=torch.tensor(encode(text),dtype=torch.long)   #we are storing all the text in dtype integer so it can run in a gpu and be faster
#print(data.shape,data.dtype) #torch.Size([1115394]) torch.int64
#print(data[:1000]) we print the first 1000 characters

#hold out method! like in machine learing exam :'(
n=int(0.9*len(data))  #90%training 10% validation
train_data=data[:n]
val_data=data[n:]
'''
#we use blocks of data instead of the whole text, because if not its too expensive
block_size=8   #the window (how much we want the model to see)
x=train_data[:block_size]       #so x is what the gpt sees
y=train_data[1:block_size+1]    #y is what the gpt has to predict
for t in range(block_size):     #it starts with 1 token, and then it keeps adding the token
    context=x[:t+1]
    target=y[t]
    print(f"when input is {context} the target: {target}")
'''


def get_batch(split):
    #generate a small batch of data of input x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval() #evaluation phase (at the moment doesnt do anything)
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train() #training phase (but its good practice, since advanced NN will have diferent phases)
    return out

#xb,yb=get_batch("train")
'''
print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)
'''

#bygram module
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token reads from logits for the next token from a lookup table
        self.token_embedding_table=nn.Embedding(vocab_size,n_embed) #table where each row respresents a vector of each token, that carry meaning
        self.position_embedding_table=nn.Embedding(block_size,n_embed)
        self.lm_head=nn.Linear(n_embed, vocab_size) #language model head, takes the embedding for each token and outputs a vector of logits (raw scores) for each token
    #so lm_head gives you the vectors for the possible next letter/word
    def forward(self,idx,targets=None):
        B,T =idx.shape
        #idx and targets are both (B,T) tensor of integers
        tok_emb=self.token_embedding_table(idx) #(B,T,n_embed) each token becomes a vector of size n_embed
        pos_emb= self.position_embedding_table(torch.arange(T,device=device))
        x=tok_emb+pos_emb
        logits=self.lm_head(x) #turns (B,T,n-embed) into (B,T,vocab_size)
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C) #this becomes 2 dimension so it can work on the cross_entropy
            targets=targets.view(B*T) #1D , if u put -1 it will do the same
            loss=F.cross_entropy(logits,targets) #torch.nn.f library (wants the second input to be the channel)

        return logits, loss
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of inidices in the current context
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss =self(idx)
            #focus only on the last time step
            logits = logits[:,-1,:] #beconmes (B,C)
            #apply softmax to get probabilities
            probs =F.softmax(logits, dim=-1)#(B,C)
            #sample from the distribution
            idx_next =torch.multinomial(probs,num_samples=1) #(B,1)
            #append sampled index to the running sequence
            idx =torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx
model=BigramLanguageModel()
m=model.to(device)
#logits,loss=m(xb,yb)
#print(logits.shape)
#print(loss) #we expect a loss of -ln(1/65) =4.17 so we have a bit of loss

#print(decode(m.generate(idx=torch.zeros((1,1),dtype=torch.long),max_new_tokens=100)[0].tolist()))

#create pytorch optimizer ADAM
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #sample a batch of data
    xb,yb=get_batch("train")
    #evaluate the loss
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True) #zeroing all the gradients
    loss.backward() #getting the gradients from all the parameters
    optimizer.step() #optimizing the parameters
    #print(loss.item())
context=torch.zeros((1,1),dtype=torch.long, device=device)
print(decode(m.generate(context,max_new_tokens=100)[0].tolist()))

B,T,C=4,8,2
x=torch.randn(B,T,C) #used later, its B,T,C
x.shape

'''first version, bad (nested loops)'''
#we want x[b,t] = mean_{i<=t} x[b,i]
xbow=torch.zeros((B,T,C)) #we are averaging a bag of words
for b in range(B): #not very efficient now
    for t in range(T): #iterating over time
        xprev=x[b,:t+1] #(t,C)
        xbow[b,t]=torch.mean(xprev,0) #average of time on the 0 dimension

'''second version, updating weights'''
#this is self-attention, previous step to having updated weights, because wei @ x lets each token look back
wei=torch.tril(torch.ones(T,T)) #short for weights, creates a TxT triangular matrix
wei=wei /wei.sum(1,keepdim=True) #normalizes each row so they sum one, turnning it into a casual averagin kernel
xbow2 = wei @ x # basicaly does matrix multiplication (does broadcasting) and does (the B is added) B,T,T @ B, T, C
#so xbow2 is B,T,C
#@ does the same as np.matmul(A,B) or A.dot(B)
torch.allclose(xbow,xbow2)

'''third version, use softmax'''
tril= torch.tril(torch.ones(T,T)) #trianglular matrix lower
wei=torch.zeros((T,T)) #weight starts with a matrix of 0
wei=wei.masked_fill(tril==0, float('-inf')) #changes wei to a triangular matrix of -infinite
wei=F.softmax(wei,dim=-1) #in softmax u exponentiate all and then divide by the sum of all
xbow3=wei @ x
#we use it in self attention, they begin at 0 (the weights) its telling us how much of each token of the past do we want to agregate.
#the tokens start looking at eachother

'''version 4: self atention'''
