# Setup

Install PyTorch (CPU wheel shown; use the CUDA wheel if available):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install the lastest version of SMASH (at least 1.0):
```bash
pip install hydro-smash
```

# Training

Train and save neural network weights (e.g. `model.pth`):

```bash
python3 train.py -pm model.hdf5 -pn net -ss 12 -e 500
```
with `model.hdf5` being the SMASH model object.

# Prediction

Use the trained neural network weights:

```bash
python3 predict.py -pm model.hdf5 -pn net/model.pth -po discharge.csv -ss 12
```

# Validation Results

```python
import smash
import pandas as pd

def predict_discharge(model: smash.Model, df: pd.DataFrame):
    model_pred = model.copy()
    q = model_pred.response.q

    for i, code in enumerate(model.mesh.code):
        if code in df.columns:
            q[i, df["timestep"].values] = df[code].values

    return model_pred

df = pd.read_csv("discharge.csv")
model = smash.io.read_model("model.hdf5")
model_pred = predict_discharge(model, df)
```
