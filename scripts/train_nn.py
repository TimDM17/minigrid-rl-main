import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import random

# === Model ===
# === Model ===
class MultimodalSequenceModel(nn.Module):
    def __init__(self, image_shape=(40, 40, 3), symbolic_dim=5, num_actions=7,
                 hidden_size=128, output_dim=5):
        super().__init__()

        # CNN for image encoding (for 40x40 input)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 40 -> 20
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 20 -> 10
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 10 -> 5
            nn.ReLU(),
            nn.Flatten()
        )
        self.image_embedding_dim = 64 * 5 * 5  # 64 channels × 5 × 5 = 1600

        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, 32)

        # Symbolic state encoder
        self.symbolic_fc = nn.Linear(symbolic_dim, 32)

        # RNN over [image + action + symbolic] concat
        self.rnn_input_dim = self.image_embedding_dim + 32 + 32
        self.rnn = nn.GRU(self.rnn_input_dim, hidden_size, batch_first=True)

        # Prediction head with action_T included
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, image_seq, action_seq, symbolic_seq, action_T):
       # image_seq = torch.nan_to_num(image_seq, nan=0.0)
        B, T, H, W, C = image_seq.shape

        if torch.isnan(image_seq).any():
            print("Warning: NaNs detected in image sequence, replacing them.")
        if torch.isnan(action_seq).any():
            print("Warning: NaNs detected in  action sequence, replacing them.")
        if torch.isnan(symbolic_seq).any():
            print("Warning: NaNs detected in  symbolic sequence, replacing them.")
        image_seq = image_seq.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W).float() / 255.0
        image_embed = self.cnn(image_seq).reshape(B, T, -1)

        action_embed = self.action_embedding(action_seq)
        symbolic_embed = F.relu(self.symbolic_fc(symbolic_seq))

        combined = torch.cat([image_embed, action_embed, symbolic_embed], dim=-1)
        out, _ = self.rnn(combined)
        last_out = out[:, -1, :]
        aT_embed = self.action_embedding(action_T)
        combined_out = torch.cat([last_out, aT_embed], dim=-1)
        logits = self.fc(combined_out)
        return torch.sigmoid(logits)



def train(model, dataloader, optimizer, device, epochs=100):
    model.to(device)
    model.train()
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0
        for images, actions, symbolic_inputs, targets, _ in dataloader:
            images = images.to(device)
            #import matplotlib.pyplot as plt
            #plt.imshow(images[0][0].numpy())
            actions = actions.to(device)
            symbolic_inputs = symbolic_inputs.to(device)
            targets = targets.to(device)

            images_seq = images[:, :-1, :]
            actions_seq = actions[:, :-1]
            symbolic_seq = symbolic_inputs[:, :-1, :]
            action_T = actions[:, -1]
            target_T = targets[:, -1, :]

            optimizer.zero_grad()
            preds = model(images_seq, actions_seq, symbolic_seq, action_T)
            loss = loss_fn(preds, target_T)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


def evaluate(model, dataloader, device):
    model.eval()
    loss_fn = nn.BCELoss()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, actions, symbolic_inputs, targets, _ in dataloader:
            images = images.to(device)
            actions = actions.to(device)
            symbolic_inputs = symbolic_inputs.to(device)
            targets = targets.to(device)

            images_seq = images[:, :-1, :]
            actions_seq = actions[:, :-1]
            symbolic_seq = symbolic_inputs[:, :-1, :]
            action_T = actions[:, -1]
            target_T = targets[:, -1, :]

            outputs = model(images_seq, actions_seq, symbolic_seq, action_T)
            loss = loss_fn(outputs, target_T)
            total_loss += loss.item()

            preds1 = (outputs > 0.5).float()
            correct += (preds1 == target_T).sum().item()
            total += target_T.numel()

    print(f"\n📊 Eval Loss: {total_loss / len(dataloader):.4f}")
    print(f"✅ Accuracy: {correct / total:.4f}")


class EnvSymbolicDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_keys = [k for k in sample if k.startswith("image_seq")]

        numbered_keys = []
        for k in image_keys:
            suffix = k.replace("image_seq", "")
            if suffix.isdigit():
                numbered_keys.append((int(suffix), k))

        if numbered_keys:
            max_len_key = max(numbered_keys)[1]
            action_key = max_len_key.replace("image_seq", "action_seq")
            symbolic_key = max_len_key.replace("image_seq", "env_symbolic_state")
        else:
            max_len_key = "image_seq"
            action_key = "action_seq"
            symbolic_key = "env_symbolic_state"

        return (
            torch.tensor(sample[max_len_key], dtype=torch.uint8),
            torch.tensor(sample[action_key], dtype=torch.long),
            torch.tensor(sample[symbolic_key], dtype=torch.float32)
        )


def collate_fn(batch):
    image_seqs, act_seqs, symbolic_seqs = zip(*batch)
    max_len = max(seq.shape[0] for seq in act_seqs)

    def pad_seq(seq, dim_pad):
        if dim_pad == 4:  # image: (T, H, W, C)
            H, W, C = seq.shape[1:]
            return F.pad(seq, (0, 0, 0, 0, 0, 0, 0, max_len - seq.shape[0]))
        elif dim_pad == 2:  # symbolic: (T, D)
            return F.pad(seq, (0, 0, 0, max_len - seq.shape[0]))
        else:  # action: (T,)
            return F.pad(seq, (0, max_len - seq.shape[0]), value=0)

    padded_images = torch.stack([pad_seq(s, 4) for s in image_seqs])
    padded_actions = torch.stack([pad_seq(s, 1) for s in act_seqs])
    padded_symbolic = torch.stack([pad_seq(s, 2) for s in symbolic_seqs])
    mask = torch.tensor([[1]*s.shape[0] + [0]*(max_len - s.shape[0]) for s in symbolic_seqs], dtype=torch.float32)

    return padded_images, padded_actions, padded_symbolic, padded_symbolic, mask


if __name__ == "__main__":
    with open("expanded_symbolic_dataset2.pkl", "rb") as f:
        raw_data = pickle.load(f)
    random.shuffle(raw_data)
    dataset = EnvSymbolicDataset(raw_data)

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSequenceModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\U0001F4DA Training on {len(train_dataset)} samples")
    train(model, train_loader, optimizer, device, epochs=200)
    torch.save(model.state_dict(), "env_symbolic_model.pt")
    print("✅ Model saved as env_symbolic_model.pt")

    print(f"\n🔍 Evaluating on {len(eval_dataset)} samples")
    model.load_state_dict(torch.load("env_symbolic_model.pt"))
    evaluate(model, eval_loader, device)
