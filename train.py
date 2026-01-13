import smash

import os
import argparse

from tools import (
    model_to_df,
    feature_engineering,
    df_to_network_in,
    build_lstm,
    nse_loss,
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# % Check version and GPU
print("PyTorch version: ", torch.__version__)
print("GPU available: ", torch.cuda.is_available())


# = ARGUMENT PARSER ==
# ====================

parser = argparse.ArgumentParser()

parser.add_argument(
    "-pm",
    "--path_filemodel",
    type=str,
    help="Select the smash Model object",
)

parser.add_argument(
    "-pn",
    "--path_netout",
    type=str,
    default=f"{os.getcwd()}/net",
    help="[optional] Select the output directory for the trained neural network",
)

parser.add_argument(
    "-ss",
    "--sequence_size",
    type=int,
    default=10,
    help="[optional] Select the squence size of inputs in LSTM network",
)

parser.add_argument(
    "-e",
    "--epoch",
    type=int,
    default=200,
    help="[optional] Select the number of epochs for training",
)

parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=512,
    help="[optional] Select the batch size for training",
)


parser.add_argument(
    "-o",
    "--optimizer",
    type=str,
    default="adam",
    help="[optional] Select the optimization algorithm",
)

parser.add_argument(
    "-lr",
    "--lr",
    type=float,
    default=1e-3,
    help="[optional] Select the value of learning rate",
)

parser.add_argument(
    "-l",
    "--loss",
    type=str,
    default="mse",
    choices=["mse", "mae", "nse"],
    help="[optional] Select the loss function to train the neural network",
)

args = parser.parse_args()


# = PRE-PROCESSING DATA ==
# ========================

# % Read model to csv and feature engineering
model = smash.io.read_model(args.path_filemodel)
df = model_to_df(model, args.sequence_size, target_mode=True)
df = feature_engineering(df)

# % Handle missing data
missing = df[df.isna().any(axis=1)]["id"].unique()
train_set = df[~df.id.isin(missing)]

# % Normalize and prepare inputs for the network
train, target = df_to_network_in(train_set, args.sequence_size, target_mode=True)


# = TRAINING ==
# =============

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.path_netout, exist_ok=True)


def make_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


print("</> Training LSTM model...")

net = build_lstm(train.shape[-2:], 1)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)

# Loss function
if args.loss == "nse":
    criterion = None  # call tools.nse_loss directly
elif args.loss == "mse":
    criterion = nn.MSELoss()
elif args.loss == "mae":
    criterion = nn.L1Loss()
else:
    raise ValueError(f"Unknown loss function: {args.loss}")

optimizer = make_optimizer(args.optimizer, net.parameters(), args.lr)

# Dataloaders
train_ds = TensorDataset(
    torch.tensor(train, dtype=torch.float32),
    torch.tensor(target, dtype=torch.float32),
)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

save_path = os.path.join(args.path_netout, "model.pth")

for epoch in range(args.epoch):
    net.train()
    running = 0.0
    for i, (xb, yb) in enumerate(train_loader, 1):
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        out = net(xb)
        if (criterion is None) and (args.loss == "nse"):
            loss = nse_loss(yb, out)
        else:
            loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running += loss.item()

        # Classical PyTorch display every 100 mini-batches
        if i % 100 == 0:
            print(
                f"[Epoch {epoch + 1}/{args.epoch}, Batch {i}/{len(train_loader)}] loss: {running / 100:.6f}"
            )
            running = 0.0

    # Print remainder if last batch wasn't printed
    if (len(train_loader) % 100) != 0:
        remaining = len(train_loader) % 100
        print(
            f"[Epoch {epoch + 1}/{args.epoch}, Batch {len(train_loader)}/{len(train_loader)}] loss: {running / remaining:.6f}"
        )

# Save final model once at the end
torch.save(
    net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
    save_path,
)
print(f"Model saved to: {save_path}")
