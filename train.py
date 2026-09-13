import torch
import torch.nn as nn
import torch.nn.functional as F
import math

text = 'hello world'

# === Tokenizaton, creating a vocabulary space === 

# use simple character tokenization to get the vocabulary space
chars = sorted(list(set(text)))

print('chars: ', chars)

# convert each token to an integer, which is used as id. this is the vocabulary of the model. 
# the id itself does not store the semantic meaning of the token, we need the embeddings for that. 

stoi = { 
    ch:i for i, ch in enumerate(chars)
}

print('stoi: ', stoi)

# to map back to get the output 
itos = { 
    i:ch for i, ch in enumerate(chars)
}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# test tokenization
print('encode: ', encode(text))
print('decode: ', decode(encode(text)))

# === Token embedding, giving each token a vector representation === 

# Token embedding table, gives each token a vector representation to capture the semantic meaning of the token itself. 
vocab_size = len(chars)

embedding_dim = 4 # we use 4 to keep it simple, in practice, the dimensions are much larger. 

# we create a vocab_size x embedding_dim matrix, where each row is a vector representation of a token. 
# these initially start as random values, which will be trained via backpropagation. 
token_embeddings = nn.Embedding(
    num_embeddings=vocab_size, 
    embedding_dim=embedding_dim
    )

# test embedding table
print('token_embeddings: ', token_embeddings)

# Get embeddings for all tokens in the input sequence
token_ids = torch.tensor(encode(text), dtype=torch.long)

# we can get the embedding for a token by passing the token id to the embedding table. 
input_token_embeddings = token_embeddings(token_ids)

print('input_token_embeddings: ', input_token_embeddings) # prints out the embeddings for each token 
# eg, h => row 1, e => row 2, l => row 3, l => row 4 etc. 
# shape: [sequence_num_tokens, embedding_dim]

# Text
#  ↓
# Tokenizer
#  ↓
# Token IDs
#  ↓
# Embedding matrix
#  ↓
# X = input_token_embeddings = [sequence_length, embedding_dim]

# Next: 
# X
#  ↓
# Position Embeddings       ← next
#  ↓
# Transformer Blocks
#  ↓
# LayerNorm
#  ↓
# LM Head
#  ↓
# Next token

# === Position embedding, the position of the token in the sequence matters, and should be encoded for the transformer to take into account === 
# the end goal: X = Etoken + Eposition, so we take into account token position in self-attention. 

context_len = 32 # this is our full context window, which is the maximum number of tokens that the model can see at once. 

position_embeddings_table = nn.Embedding(
    num_embeddings=context_len,
    embedding_dim=embedding_dim # same dimension as the token embeddings, so we can add them together. usually same as token embeddings for standard GPT. 
) # Another random initialized matrix, which will be trained via backpropagation. 

# Modern architectures might use RoPE (Rotary Position Embedding) for position embeddings, which is a more efficient way to encode position information. 
# This supports longer context windows and is more efficient to compute. 

num_tokens = len(token_ids)
# basically, we assign each token a position id. 
position_ids = torch.arange(num_tokens, dtype=torch.long) # generates [0, 1, 2, ..., num_tokens-1]

position_embeddings = position_embeddings_table(position_ids)

input_embeddings = input_token_embeddings + position_embeddings # =X where each row captures the token meaning + position meaning for each token individually.  

print('input_embeddings: ', input_embeddings)

# using this input, we can now pass it to the transformer blocks with causal self-attention.

# === Transformer block: Causal self attention === 
# how do we get the model to predict the next token, considering all the tokens in the context window?

# 1 layer of transformer: 
# X
# │
# ├──> Causal Self-Attention
# │         │
# │         ▼
# │    tokens communicate
# │
# ├──> Feed-Forward Network
# │         │
# │         ▼
# │    each token processes what it learned
# │
# ▼
# new X

# what is causal self attention for? 
# this is a river bank 
# i went to withdraw money from the bank
# we need to know the context of the word "bank" from the earlier tokens. it needs to ask "which previous tokens are relevant to "bank"? = attention
# eg, which earlier tokens should i pay the most attention to. 
# in the above sentence "i went to withdraw money from the bank": "money" = high attention whereas "to" = low attention to understand the context of "bank"

# self = based on the same input sentence (Q, K and V come from the same source)
# cross = attend to info coming from somewhere else (Q comes from same source, but K and V might come from a different source)
# causal = only attend to itself & previous tokens, not future tokens. 

# For each token, create 3 vectors: Query (Q), Key (K), Value (V)
# Q = What am I looking for? 
# K = What information do i advertise? 
# V = What information do i actually contain? 

# Conceptually: 
# Q = XW_Q
# K = XW_K
# V = XW_V

# X only captures the token meaning + position meaning for each token individually. 
# We need an additional feature to tune what should each token ask for, what it should advertise and what info it should actually contain. 

# We look at K to find the most relevant information to Q, and then use V to get the actual information weighted by the most relevant. 
# 1. Compute dot product of Q and K for each token, to get the attention scores. (How much attention/how relevant is this token to me?)
# Scores = QK^T
# Q: 
# Q =
# [A B] # query vector for token 0
# [C D]
# [E F]

# K^T: (The ^T means transpose, so K^T is the transpose of K. This is necessary for the dot product to be computed.) 
# K.T =
# [G H I]
# [J K L]
# each column is the key vector for each token in the context window. 

# QK^T:
# Q @ K.T =
# [AG+BJ   AH+BK   AI+BL]
# [CG+DJ   CH+DK   CI+DL]
# [EG+FJ   EH+FK   EI+FL]

# 2. Causal mask: We cannot allow the model to attend to future tokens, or it "cheats", so we apply a mask. 
# eg: can only see the previous tokens, future tokens are masked out. 
# i <?>
# i went <?>
# i went to <?> 

# How do we implement this masking? 
# By replacing tokens it should not see with -∞ and then applying softmax to get the masked (causal) attention scores. 

# Scores before masking: QK^T 
# Scores after masking:  
# Before masking, our scores are:
#          key 0   key 1   key 2
# for token 0:   [ A,     B,     C ]   ← query = token 0 and each column value is how much it attends to each other token. 
# for token 1:   [ D,     E,     F ]   ← query = token 1
# for token 2:   [ G,     H,     I ]   ← query = token 2

# We apply this mask:
# [A, -∞, -∞] # since token 0 should not attend to future tokens (ie, 1 and 2), their query x key values are replaced with -∞.
# [D, E, -∞]
# [G, H, I]

# 3. Softmax: Just a useful mathematical function that converts numbers into probabilities, 
# where larger scores get even bigger weight and smaller scores get even smaller weight (it is not linear normalization). 
# Given a masked scoring: 
# [1, 2, -∞]
# Softmax converts to: [0.27, 0.73, 0] 

# A = softmax(masked(QK^T))

# Idea of softmax over linear normalization: 
# - "If one match is substantially better, focus considerably more on it."
# - Convenient to give -∞ a 0 probability, as it is not a valid score. 
# but it is also possible to research with other attention normalization functions eg, Sigmoid, sparse, linear etc

# 4. After computing how much attention A, we compute the actual retrieved value from V. 
# Causal self attention output = AV

# === For minGPT, I will only use a single attention head for simplicity, but it only captures 1 way to compare tokens. === 
# implementation of a single attention head 
embedding_dim = input_embeddings.shape[-1]
head_size = embedding_dim

# learned matrix with 4 input features (embedding dim) and 4 output features (head size)
# Represents a matrix multiplication y = Wx + b, where W is a learnable matrix. 
# b is the bias, which is omitted here for simplicity (many transformer models also omit this for K, Q and V). 
query = nn.Linear(embedding_dim, head_size, bias=False)
key = nn.Linear(embedding_dim, head_size, bias=False)
value = nn.Linear(embedding_dim, head_size, bias=False)

#                     W_Q
#               ┌─────────────> Q
#               │
# X ────────────┼──── W_K ───> K
#               │
#               └──── W_V ───> V

Q = query(input_embeddings)
K = key(input_embeddings)
V = value(input_embeddings)

scores = Q @ K.T 

# scale the scores, to prevent the scores from becoming too large or too small and skewing softmax, 
# especially as embedding_dim increases.  
# since softmax does "if one is considerably better, focus considerably more on it.", we want this but not to the extreme. 
scores = scores / math.sqrt(head_size)

# causal masking 
T = input_embeddings.shape[0] # number of tokens in the sequence
mask = torch.tril(torch.ones(T, T)) # lower triangle = 1, upper triangle = 0
# Generated by torch.tril(torch.ones(T, T))
# 1 0 0 0
# 1 1 0 0
# 1 1 1 0
# 1 1 1 1

scores = scores.masked_fill(mask == 0, float("-inf"))

# softmax 
attention_weights = F.softmax(scores, dim=-1)

head_output = attention_weights @ V
# end of single attention head 

# Having multiple heads: 
# Each head represents a different way to compare tokens. Hence, having more heads allows the model to 
# capture multiple ways to compare tokens. 
#                       ┌─ W_Q1, W_K1, W_V1 -> Head 1
#                       │
# X --------------------├─ W_Q2, W_K2, W_V2 -> Head 2
#                       │
#                       ├─ W_Q3, W_K3, W_V3 -> Head 3
#                       │
#                       └─ W_Q4, W_K4, W_V4 -> Head 4
# the concatenated head should be the same as the input embedding dimension.

# Head 1 output ─┐
# Head 2 output ─┤
# Head 3 output ─┼─> concatenate -> output projection
# Head 4 output ─┘

# eg, head 1 = Which words are grammatically related to me?
# eg, head 2 = Which words help determine what I mean? 
# ... 
# they may each represent different questions regarding previous tokens and relationship to current token. 

print('head_output: ', head_output)

# Run several heads in parallel -> concatenate -> linear projection -> multi head output 

class Head(nn.Module): 
    def __init__(self, embedding_dim, head_size):
        super().__init__()

        # define its own learnable matrices W_Q, W_K, W_V, so that it can represent a different way to compare tokens. 
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

    def forward(self, x):
        sequence_length, embedding_dim = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        head_size = K.shape[-1]

        scores = Q @ K.T
        # soften the scores so that softmax does not explode or vanish for large embedding_dim. 
        scores = scores / math.sqrt(head_size)

        mask = torch.tril(torch.ones(sequence_length, sequence_length))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        head_output = attention_weights @ V

        return head_output 

class MultiHeadAttention(nn.Module): 
    def __init__(self, embedding_dim, num_heads): 
        super().__init__()

        assert embedding_dim % num_heads == 0
        head_size = embedding_dim // num_heads

        # Register a list of neural-network modules in the module, that must be trained. 
        self.heads = nn.ModuleList([
            Head(embedding_dim, head_size) for _ in range(num_heads) # Each head has its independent weights,
            # why multiple heads? Each head captures a different way of comparing tokens. 
        ])
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x): 
        # suppose the X embedding dim = 8 and there are 4 heads, each head will output 2 dimensions. 
        head_outputs = [head(x) for head in self.heads] # Each head receives the same input X, 
        # representing the individual semantic + position meaning of each sequence token. 

        # then we concat all outputs of the 4 heads * 2 to get 8 dimensions per token. 
        out = torch.cat(head_outputs, dim=-1) # Each token gets a num_heads * head_size dimension vector. 
        # After placing the features for each token side by side, we need to mix them to form a cohesive signal. 
        # This is done by a linear projection layer, which learns to combine the features 
        # in a way that is useful for the downstream task. 
        out = self.projection(out)

        return out
    
num_heads = 4
# multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)

# === Layer normalization === 
# ln1 = nn.LayerNorm(embedding_dim)
# multi_head_attention_output = multi_head_attention(ln1(input_embeddings))

# print('multi_head_attention_output: ', multi_head_attention_output)
# print('input shape: ', input_embeddings.shape)
# print('output shape: ', multi_head_attention_output.shape)

# === Residual connection === 
# Why do we not just replace X with the multi_head_attention_output?
# We use a residual connection:
#
#     y = x + F(x)
#
# where:
#   x    = the current token representations entering this sub-layer
#   F(x) = the transformation performed by the sub-layer
#          e.g. multi-head attention (usually applied to LayerNorm(x))

# 1. To avoid the vanishing gradient problem, adding the original input_embeddings allows the signal from the loss to
# reach the earliest semantic and positional embeddings during backpropagation. 
#
# Since:
#
#     y = x + F(x)
#
# the derivative contains an identity term:
#
#     dy/dx = I + dF/dx
#
# The identity term I provides a direct path for gradients during
# backpropagation, instead of forcing the gradient to pass entirely
# through F(x).
#
# In a simplified scalar example, if F'(x) = 0.1:
#
#     without residual: dy/dx = 0.1
#     with residual:    dy/dx = 1 + 0.1 = 1.1
#
# This illustrates why residual connections reduce the tendency for
# gradients to vanish across many layers.

# 2. The attention is only meant to add additional contextual information (considering earlier tokens), not completely replace the original representation. 
# X = input_embeddings + multi_head_attention_output

class TransformerBlock(nn.Module): 
    def __init__(self, embedding_dim, num_heads): 
        super().__init__()
        
        # Why LayerNorm? 
        # Goal: Normalizes the mean and variance of the input to improve training stability. 
        # For example, given 2 input tokens. [1, 2] and [200, 400].
        # The larger values dominate the linear layers and attention, masking the smaller ones. 
        #
        # LayerNorm learns 1 scale and offset for each embedding dimension (shared across all tokens), this is learned to provide the most useful output. 
        # While it sacrifices the information about the absolute mean and scale of the tokens, 
        # this is mitigated by the trained weights, which are tuned to such that it optimizes 
        # to learn which scales / offsets lead to useful output for each embedding dimension. 

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        # === Feed forward network === 
        # After the tokens have talked to other tokens, we need each token to process the gathered information independently. 
        # hence, we use a Multi-layer perceptron (MLP) neural network
        # this FFN does not mix connections again, each token processes its own gathered information. 
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim), # 4x ratio = larger hidden space for more expressiveness. 
            # Why do we need non linear activation function? (activation: +ve neuron is active, -ve neuron is inactive)
            # Without it, we stack (append) linear layers, which is equivalent to a single linear layer. There are limits to expressiveness with just linear layers. 
            # With non-linear activation function, we can add more expressiveness, allowing the model to better represent the data. 
            nn.GELU(), # Non-linear activation function 
            # GELU > ReLU since: 
            # 1. ReLU removes all the negative values (Which might remove useful information ie,  a slightly negative value might still contain useful information)
            # 2. GELU: positive: increasingly retained, small negative: partially retained, large negative: close to 0.  
            nn.Linear(4 * embedding_dim, embedding_dim)
        )
    
    def forward(self, x):
        # Post norm: 
        #         x
        # ├──────────────┐
        # │              │
        # └-> F(x) ------+
        #                ↓
        #               Add
        #                ↓
        #            LayerNorm
        # Pre norm: 
        # x
        # ├───────────────────────────┐
        # │                           │
        # └-> LayerNorm -> F(x) ------+
        #                             ↓
        #                          new x
        # LayerNorm is not applied to the residual (we apply pre-norm instead of post-norm), since: 
        # 1. It keeps the x residual, preventing the vanishing gradient problem.
        # 2. x represents the model's shared running state, we are applying updates considering other tokens from each multi-head attention transformer block. 
        x = x + self.multi_head_attention(self.ln1(x))

        # Each token converts its gathered / communicated information into a new useful representation 
        x = x + self.ffn(self.ln2(x))
    
        return x
    
transformer_block = TransformerBlock(embedding_dim, num_heads)

transformer_block_output = transformer_block(input_embeddings)

print('transformer_block_output: ', transformer_block_output)

# === Stacking transformer blocks to form a transformer encoder === 
# Why? 
# Each block lets tokens communicate + process their representation using the communicated information. 
# We might need multiple layers of refinement to fully capture complex language. (Each subsequent block builds on the previous block's output)
#  Block 1:
# basic contextual relationships

# Block 2:
# relationships based on what Block 1 discovered

# Block 3:
# even richer relationships

# Block 4:
# further refinement

class MinGPT(nn.Module): 
    def __init__(self, vocab_size, embedding_dim, context_len, num_heads, num_layers): 
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_len, embedding_dim)

        # Transformer stack 
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(embedding_dim, num_heads) for _ in range(num_layers)])

        # Layer norm 
        self.final_ln = nn.LayerNorm(embedding_dim)

        # Convert hidden representation into the actual next token probabilities. 
        # the embedding dim is an internal dimension, we need to project it out to actual vocab size. 
        self.lm_head = nn.Linear(embedding_dim, vocab_size) 

    def forward(self, token_ids): 
        T = token_ids.shape[0] # sequence length

        position_ids = torch.arange(T, device=token_ids.device) # run on GPU 

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids)

        X = token_embeddings + position_embeddings
        # compute attention via the transfomrmer block, which also captures the residual 
        X = self.transformer_blocks(X)

        # Final layer norm 
        # Why? After multiple transformer blocks, the values might become too large or small, 
        # we apply layer norm to bring the values back to a reasonable range. 
        X = self.final_ln(X)

        # Convert hidden representation into the actual next token scores. 
        # Note that these are not probabilities yet, just raw model scores.  
        # will be passed to eg, softmax, sigmoid to compute the actual probabilities. 
        logits = self.lm_head(X)

        return logits 

model = MinGPT(vocab_size=vocab_size, embedding_dim=4, context_len=32, num_heads=2, num_layers=4)

logits = model(token_ids)

print('logits: ', logits)
print('logits shape: ', logits.shape) # shape: [sequence_length, vocab_size]
