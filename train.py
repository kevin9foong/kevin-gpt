import os

import torch
import torch.nn.functional as F

from model import MinGPT

CHECKPOINT_PATH = "checkpoints/checkpoint.pt"

# Checkpointing: snapshot our training state so we can continue further training from where we left off.
# │
# ├── model weights                    → what model has learned.
# ├── optimizer state                  → optimizer's running history eg, AdamW can better choose how to update future weights based on past steps/gradients.
# ├── tokenizer mappings (stoi, itos)  → so that the token IDs mean the same thing across runs.
# ├── model configuration              → model architecture, so we can reconstruct the same model.
# └── training step                    → where training stopped
def save_checkpoint(model, optimizer, step, stoi, itos, config, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "stoi": stoi,
        "itos": itos,
        "config": config,
    }
    torch.save(checkpoint, path)


# === Scaffold tokenizer + config ===
text = "The Shima Peninsula is home to japan's most sacred Shinto shrines, the Ise Shrines."

chars = sorted(list(set(text)))  # extracts unique characters from the text

itos = {i: ch for i, ch in enumerate(chars)}
stoi = {ch: i for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


config = {
    "vocab_size": len(chars),
    "embedding_dim": 256,
    "context_len": 128,
    "num_heads": 4,
    "num_layers": 4,
}

# === Load full checkpoint if it exists, otherwise start fresh ===
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
start_step = 0

if os.path.exists(CHECKPOINT_PATH):
    # Close Python tomorrow, then restore everything from this file.
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    # Prefer "config"; fall back if an older checkpoint used "model_config".
    config = checkpoint.get("config", checkpoint.get("model_config"))

    model = MinGPT(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        context_len=config["context_len"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # first arg: tells the optimizer which parameters to update. it will not update others.
    # second arg: learning rate (how much to update the weights by for each step)
    #
    # Basic gradient descent optimizer:
    # Wnew = Wold - learning rate * gradient of the loss with respect to the weights.
    # AdamW (adaptive moment estimation with weight decay) optimizer:
    # "Gradient right now is +2,
    # but I've also observed the recent history
    # of gradients and their magnitudes,
    # so I'll choose a more appropriate update using some math."
    # there are different optimizers which are used to decide how to update weights according to the gradients.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_step = checkpoint["step"] + 1
    print(f"Resumed from checkpoint at step {checkpoint['step']} → continuing at {start_step}")
else:
    model = MinGPT(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        context_len=config["context_len"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print("No checkpoint found — starting from step 0")

token_ids = torch.tensor(encode(text), dtype=torch.long)

# === Training the model ===

# 1. Create next-token prediction pairs.
#
# x contains every token except the final token.
# y contains every token except the first token.
#
# Therefore y[i] is the correct next token for x[i]'s position.
x = token_ids[:-1]  # remove the last token, since there is no next token to predict.
y = token_ids[1:]  # what each token should predict next.

# example:
# original text:
# h e l l o   w o r l d
#
# x:
# h e l l o   w o r l
# with causal attention masks, where we override the attention with -inf,
# future tokens are effectively ignored (low attention scores).
# position 0: h
# position 1: h e
# position 2: h e l
# position 3: h e l l
#
# y:
# e l l o   w o r l d
#
# model receives x and predicts y^ against ground truth y

# 2. Run a forward pass to get the logits.
# logits = model(x)

# 3. Define the loss function
# Correct prediction (ie high probability for right target token) = low loss
# Incorrect prediction (ie low probability for right target token) = high loss
# Take the average to get the final loss.
# Why cross entropy? L=−log(Pcorrect​)
# so if Pcorrect is high, then the loss is low, and vice versa.
# We want the following: "Given the context, assign high probability to the next correct token."
# Cross entropy measures "how much probability did we assign the the correct token answer?"
# loss = F.cross_entropy(logits, y)

# 4. Backpropagate the loss to update the model parameters.
# we are fitting the model to the data by updating the model parameters.


def train(model, optimizer, x, y, start_step, num_steps, stoi, itos, config, checkpoint_path):
    model.train()  # put model into training mode, does not actually run a step.

    # Resume-friendly loop: yesterday 0..999, today 1000..1999, etc.
    for step in range(start_step, start_step + num_steps):
        # without torch.no_grad, it remembers how everything was computed across the layers to allow for differentiation.
        logits = model(x)  # forward pass to get logits

        loss = F.cross_entropy(logits, y)  # higher loss, more updates to params.

        optimizer.zero_grad()  # clear the computed gradient from the previous step, instead of accumulating them (which is useful for other use cases).

        loss.backward()  # computes the gradient of the loss with respect to the model parameters.
        # Understanding why backward: since it updates from the last layer to the first layer.
        #
        # Imagine a tiny network:
        # w -> a -> b -> L, w is the only parameter we are updating, a and b are intermediate values (activations) produced by later layers during forward pass.
        #
        # a = f(w)
        # b = g(a)
        # L = h(b)
        #
        # to update w, we need to compute dL/dw.
        # dL/dw = dL/db * db/da * da/dw (using the chain rule)
        #
        # so we compute from the back:
        # dL/db, backmost layer
        # dL/da = dL/db * db/da
        # dL/dw = dL/da * da/dw, frontmost layer (we get our required gradient)
        #
        # How we use the dL/dw gradient to update w:
        # Loss
        # ^
        # | *
        # |  *
        # |   *
        # |    *
        # |      *
        # |        *
        # |           *  minimum
        # +------------------------> w
        #     1             3
        # we compute dL/dw, if the gradient is negative, we should increase w to reduce the loss. hence, we update w accordingly.
        #
        # after we get the gradient, we need to update the weights
        # we dont want to over-update the weights (too large learning rate/step size), since it will cause oscillations.
        # too small: convergence takes too many steps.
        # example of a simple optimizer step, not adamW:
        # Wnew = Wold - learning rate * gradient of the loss with respect to the weights.
        optimizer.step()  # we do this update for num_steps times.

        if step % 100 == 0:
            print(f"step {step}: loss = {loss.item():.4f}")

        # Save periodically (and on the last step) instead of every step.
        is_last_step = step == start_step + num_steps - 1
        if is_last_step:
            save_checkpoint(
                model,
                optimizer,
                step,
                stoi,
                itos,
                config,
                checkpoint_path,
            )

num_steps = 1000
train(
    model,
    optimizer,
    x,
    y,
    start_step,
    num_steps,
    stoi,
    itos,
    config,
    CHECKPOINT_PATH,
)

# === Prediction ===
def predict_next_token(model, context):
    model.eval()

    token_ids = torch.tensor(encode(context), dtype=torch.long)

    # this disables gradient computation and its memory usage for graph,
    # since this is not a training step.
    with torch.no_grad():
        logits = model(token_ids)

    last_logits = logits[-1]

    probabilities = F.softmax(last_logits, dim=-1)

    y_hat_id = torch.argmax(probabilities).item()

    return itos[y_hat_id]


context = "japan"

num_predictions = 10
for prediction in range(num_predictions):
    next_token = predict_next_token(model, context)
    context += next_token
    print(next_token)
