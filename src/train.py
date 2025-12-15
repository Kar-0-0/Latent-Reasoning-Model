from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import random
import numpy as np
from model import LatentReasoningModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class LongIncrementDataset(Dataset):
    def __init__(self, num_sequences=1_000, seq_len=1_000):
        """
        Args:
            num_sequences: total number of sequences in dataset
            seq_len: length of each sequence
        """
        self.num_sequences = num_sequences
        self.seq_len = seq_len
        self.data = self._generate_sequences()

    def _generate_sequences(self):
        # Generate sequences: 1, 2, 3, ..., n-1, n+5
        sequences = []
        for _ in range(self.num_sequences):
            seq = np.arange(1, self.seq_len + 1)
            seq[-1] += 5  # last element skip by 5
            sequences.append(seq)
        return np.array(sequences, dtype=np.float32)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        seq = self.data[idx]
        # You can return input and target separately if you want to predict the next number
        input_seq = seq[:-1]  # all except last
        target = seq[1:]      # shifted by 1

        return torch.tensor(input_seq), torch.tensor(target)


class HiddenRuleSwitchDataset(Dataset):
    def __init__(self, num_samples, input_length, vocab_size):
        self.data = []
        self.input_length = input_length
        self.vocab_size = vocab_size


        for _ in range(num_samples):
            start = random.randint(1, vocab_size - 2)
            switch_t = random.randint(input_length // 3, 2 * input_length // 3)


            seq = [start]
        for t in range(input_length - 1):
            prev = seq[-1]
            if t < switch_t:
            # Rule A: increment
                nxt = (prev + 1) % vocab_size
            else:
            # Rule B: multiply
                nxt = (prev * 2) % vocab_size
            seq.append(nxt)


        self.data.append(torch.tensor(seq, dtype=torch.long))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]

class StructuredTokenDataset(Dataset):
    def __init__(self, num_samples, input_length, vocab_size):
        self.num_samples = num_samples
        self.input_length = input_length
        self.vocab_size = vocab_size
        self.data = []

        for _ in range(num_samples):
            start = random.randint(0, vocab_size-1)
            # create a deterministic sequence: next token = (prev + 1) % vocab_size
            seq = [(start + i) % vocab_size for i in range(input_length)]
            self.data.append(torch.tensor(seq))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = seq[:-1]  # input
        y = seq[1:]   # target: next token
        return x, y
    

input_length = 30
vocab_size = 512 + 6
batch_size = 32
num_samples = 4000
latent_dim = 64
attn_dim = 64
n_layers = 2
n_heads = 4
dropout = 0.1
lr = 1e-3
epochs = 20

dataset = LongIncrementDataset(num_sequences=1_000, seq_len=500)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




model = LatentReasoningModel(
    n_layers=n_layers,
    latent_dim=latent_dim,
    attn_dim=attn_dim,
    input_length=input_length-1,
    n_heads=n_heads,
    vocab_size=vocab_size,
    dropout=dropout
)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

all_latents = []
for epoch in range(epochs):
    losses = []
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        x, y = x.long(), y.long()
        logits, latents = model(x, return_latents=True)
        B, T, vocab_size = logits.shape
        logits = logits.view(B*T, vocab_size)
        y = y.view(B*T)
        loss = F.cross_entropy(logits, y)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if len(all_latents) < 200:
            all_latents.append(latents[0].detach().cpu().numpy())

        if i % 10 == 0:
            print(f"{i} / {len(dataloader)}")
            with torch.no_grad():
                logits = model(x)
                logits = logits[0]
                preds = logits.argmax(dim=-1)
                y_first = y.view(B, T)[0]
                print(f"Shapes: {preds[0].shape, y_first.shape}\nResult: {torch.allclose(preds[0], y_first)}")
    
    print(f"Epoch {epoch} Loss: {sum(losses)/len(losses)}")



all_latents = np.stack(all_latents) # [N, T, D]
np.save("latent_trajectories.npy", all_latents)
print("Saved latent_trajectories.npy")


# -------------------- quick sanity test --------------------
x_test, y_test = next(iter(dataloader))
x_test = x_test.to(device)
x_test = x_test.long()


with torch.no_grad():
    logits, latents = model(x_test, return_latents=True)
    preds = logits[0].argmax(dim=-1)


print(torch.allclose(preds[0], y_first))