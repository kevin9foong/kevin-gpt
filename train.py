import torch 
import torch.nn.functional as F 

from model import MinGPT

# === Scaffold the model === 
text = "hello world"

chars = sorted(list(set(text)))

itos = { i:ch for i, ch in enumerate(chars) }
stoi = { ch:i for i, ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

vocab_size = len(chars)

model = MinGPT(
    vocab_size=vocab_size, 
    embedding_dim=4, 
    context_len=32, 
    num_heads=2, 
    num_layers=4
)

token_ids = torch.tensor(encode(text), dtype=torch.long)

# === Training the model === 

# 1. Create next-token prediction pairs.
#
# x contains every token except the final token.
# y contains every token except the first token.
#
# Therefore y[i] is the correct next token for x[i]'s position.
x = token_ids[:-1] # remove the last token, since there is no next token to predict. 
y = token_ids[1:] # what each token should predict next. 

# 2. Run a forward pass to get the logits. 
logits = model(x)

# 3. Define the loss function
# Correct prediction (ie high probability for right target token) = low loss 
# Incorrect prediction (ie low probability for right target token) = high loss 
# Take the average to get the final loss. 
# Why cross entropy? L=−log(Pcorrect​)
# so if Pcorrect is high, then the loss is low, and vice versa. 
# We want the following: "Given the context, assign high probability to the next correct token."
# Cross entropy measures "how much probability did we assign the the correct token answer?"
loss = F.cross_entropy(logits, y)

