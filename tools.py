import smash
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler

from datetime import timedelta


def model_to_df(
    model: smash.Model,
    sequence_size: int,
    target_mode: bool = False,
    precip: bool = True,
    pot_evapot: bool = True,
    precip_ind: bool = True,
    gauge: list | None = None,
):
    """
    Read Model object in smash and extract into a raw DataFrame.
    """
    dict_df = {}

    # % ID
    n_sequence = model.setup.ntime_step // sequence_size

    ntime_step = model.setup.ntime_step - (model.setup.ntime_step % sequence_size)

    id_single_gauge = np.repeat(np.arange(n_sequence), sequence_size)
    dict_df["id"] = np.concatenate(
        [id_single_gauge + i * n_sequence for i in range(model.mesh.ng)]
    )

    # % Catchment code
    dict_df["code"] = np.repeat(model.mesh.code, ntime_step)

    # % Timestep
    dict_df["timestep"] = np.tile(np.arange(ntime_step), model.mesh.ng)

    # % Meaningful timestep in year for learning
    tsy = _timestep_convert(model.setup.start_time, model.setup.dt, ntime_step)
    dict_df["timestep_in_year"] = np.tile(tsy, model.mesh.ng)

    # % Discharge (target)
    if target_mode:
        qo = model.response_data.q[..., :ntime_step]
        qo[qo < 0] = np.nan
        dict_df["discharge"] = qo.flatten(order="C")

    # % Mean precipitation
    if precip:
        prcp = model.atmos_data.mean_prcp[..., :ntime_step]
        prcp[prcp < 0] = np.nan
        dict_df["precipitation"] = prcp.flatten(order="C")

    # % PET
    if pot_evapot:
        pet = model.atmos_data.mean_pet[..., :ntime_step]
        pet[pet < 0] = np.nan
        dict_df["pet"] = pet.flatten(order="C")

    # % Precipitation indices
    if precip_ind:
        prcp_ind = smash.precipitation_indices(model)
        d1 = prcp_ind.d1[..., :ntime_step]
        d1[np.isnan(d1)] = -1
        d2 = prcp_ind.d2[..., :ntime_step]
        d2[np.isnan(d2)] = -1
        dict_df["d1"] = d1.flatten(order="C")
        dict_df["d2"] = d2.flatten(order="C")

    df = pd.DataFrame(dict_df)

    if not gauge is None:
        df = df[df["code"].isin(gauge)]

    return df


def _timestep_convert(st, dt, n_ts, by=None):
    if by is None:
        by = 1
    elif isinstance(by, str):
        if by == "hour":
            by = int(60 * 60 / dt)
        elif by == "day":
            by = int(24 * 60 * 60 / dt)
        elif by == "month":
            by = int(365 / 12 * 24 * 60 * 60 / dt)

    timestep = np.arange(1, int(365 * 24 * 60 * 60 / dt) + 1)

    defst = f"{pd.to_datetime(st).year}-08-01 00:00:00"

    if pd.Timestamp(st) < pd.Timestamp(defst):
        timestamps = pd.date_range(start=st, end=defst, freq=timedelta(seconds=dt))
        s_ind = timestep.size - (len(timestamps) - 1)
    else:
        timestamps = pd.date_range(start=defst, end=st, freq=timedelta(seconds=dt))
        s_ind = len(timestamps) - 1

    return np.array([timestep[(s_ind + i) % len(timestep)] // by for i in range(n_ts)])


def feature_engineering(df: pd.DataFrame):
    """
    Perform feature engineering from the raw DataFrame.
    """
    df["year"] = df["timestep"] // np.max(df["timestep_in_year"])
    drop_cols = ["year"]

    try:
        df["precipitation_cumsum"] = df.groupby(["code", "year"])[
            "precipitation"
        ].cumsum()
    except:
        pass

    try:
        df["pet_cumsum"] = df.groupby(["code", "year"])["pet"].cumsum()
        df["sqrt_pet"] = np.sqrt(df["pet"])
        drop_cols.append("pet")
    except:
        pass

    df = df.drop(drop_cols, axis=1)

    return df


def df_to_network_in(
    df: pd.DataFrame,
    sequence_size: int,
    target_mode: bool = False,
):
    """
    Normalize data and prepare input for the neural network.
    """
    # % Drop info columns
    df = df.drop(["id", "code", "timestep"], axis=1)

    if target_mode:
        # check if 'discharge' is already located in the last column
        if df.columns[-1] != "discharge":
            columns = [col for col in df.columns if not "discharge" in col]
            columns = np.append(columns, "discharge")
            df = df[columns]

        # convert to numpy array
        data = df.to_numpy()[..., :-1]
        target = df.to_numpy()[..., -1]
        target = target.reshape(-1, sequence_size, 1)

    else:
        data = df.to_numpy()
        target = None

    # % Normalize
    data = RobustScaler().fit_transform(data)
    data = data.reshape(-1, sequence_size, data.shape[-1])

    return data, target


def nse(y_true, y_pred):
    if y_true.dim() == 3:
        y_true = y_true[..., 0]
    if y_pred.dim() == 3:
        y_pred = y_pred[..., 0]
    y_true_mean = torch.mean(y_true, dim=1, keepdim=True)
    num = torch.sum((y_true - y_pred) ** 2, dim=1)
    den = torch.sum((y_true - y_true_mean) ** 2, dim=1)
    score = 1.0 - (num / (den + 1e-8))
    return torch.mean(score)


def nse_loss(y_true, y_pred):
    return 1.0 - nse(y_true, y_pred)


class LSTMNet(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256 * 2,
            hidden_size=128,
            batch_first=True,
            bidirectional=False,
        )
        self.fc1 = nn.Linear(128, 64)
        self.act1 = nn.SELU()
        self.fc2 = nn.Linear(64, output_size)
        self.act2 = nn.ReLU()

    def forward(self, x):
        y, _ = self.lstm1(x)
        y, _ = self.lstm2(y)
        y = self.fc1(y)
        y = self.act1(y)
        y = self.fc2(y)
        y = self.act2(y)
        return y


def build_lstm(input_shape: tuple, output_size: int):
    """
    The LSTM neural network for learning streamflow prediction error.
    input_shape: (seq_len, feature_count)
    """
    input_size = int(input_shape[1])
    return LSTMNet(input_size=input_size, output_size=output_size)
