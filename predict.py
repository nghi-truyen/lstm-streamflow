import smash

import os
import argparse

import numpy as np
import pandas as pd

from tools import model_to_df, feature_engineering, df_to_network_in, build_lstm

import torch
import torch.nn as nn

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
    help="Select the smash Model object to correct",
)

parser.add_argument(
    "-pn",
    "--path_net",
    type=str,
    help="Path to trained network weights (.pth/.pt) or a directory containing one",
)

parser.add_argument(
    "-po",
    "--path_fileout",
    type=str,
    default=f"{os.getcwd()}/discharge-pred.csv",
    help="[optional] Select path for the output file",
)

parser.add_argument(
    "-ss",
    "--sequence_size",
    type=int,
    default=10,
    help="[optional] Select the squence size of inputs in LSTM network",
)

parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=512,
    help="[optional] Select the batch size for predicting",
)

args = parser.parse_args()


# = PRE-PROCESSING DATA ==
# ========================

# % Read model to csv and feature engineering
model = smash.io.read_model(args.path_filemodel)
df = model_to_df(model, args.sequence_size)
df = feature_engineering(df)

# % Handle missing data
missing = df[df.isna().any(axis=1)]["id"].unique()
pred_set = df[~df.id.isin(missing)]

# % Normalize and prepare inputs for the network
pred, _ = df_to_network_in(pred_set, args.sequence_size)

# = PREDICT ==
# ============

try:
    os.makedirs(os.path.dirname(args.path_fileout), exist_ok=True)
except:
    pass

output_size = 1

# Resolve weight path: accept a file or directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.isdir(args.path_net):
    candidates = sorted(
        [
            os.path.join(args.path_net, f)
            for f in os.listdir(args.path_net)
            if f.endswith(".pt") or f.endswith(".pth")
        ]
    )
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No weight files (*.pt|*.pth) found in directory: {args.path_net}"
        )
    weight_path = candidates[0]
else:
    weight_path = args.path_net
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

net = build_lstm(pred.shape[-2:], output_size)
state = torch.load(weight_path, map_location=device)
net.load_state_dict(state)
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net.to(device)
net.eval()

with torch.no_grad():
    pred_tensor = torch.from_numpy(pred.astype(np.float32)).to(device)
    out_chunks = []
    for i in range(0, pred_tensor.size(0), args.batch_size):
        batch = pred_tensor[i : i + args.batch_size]
        out = net(batch)
        out_chunks.append(out)
    y_pred = torch.cat(out_chunks, dim=0).cpu().numpy().reshape(-1, output_size)

discharge = y_pred[:, 0]

# % Write results to csv file with full series and -99 for missing
df_pred_raw = pd.DataFrame(
    {
        "code": pred_set["code"],
        "timestep": pred_set["timestep"],
        "discharge": discharge,
    }
)

# Create full index from original df and left-join predictions
df_full_index = df[["code", "timestep"]].drop_duplicates()
df_pred = df_full_index.merge(df_pred_raw, on=["code", "timestep"], how="left")
df_pred["discharge"] = df_pred["discharge"].fillna(-99)

df_pred = df_pred.pivot(
    index="timestep", columns="code", values="discharge"
).reset_index()
df_pred.to_csv(args.path_fileout, index=False)
