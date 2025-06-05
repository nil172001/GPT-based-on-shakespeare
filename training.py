import torch



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
#we use blocks of data instead of the whole text, because if not its too expensive
block_size=8   #the window (how much we want the model to see)
x=train_data[:block_size]       #so x is what the gpt sees
y=train_data[1:block_size+1]    #y is what the gpt has to predict
for t in range(block_size):     #it starts with 1 token, and then it keeps adding the token
    context=x[:t+1]
    target=y[t]
    print(f"when input is {context} the target: {target}")


torch.manual_seed(1337)
batch_size=4 #how many independent sequences we process in parallel
block_size=8

def get_batch(split):
    #generate a small batch of data of input x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y