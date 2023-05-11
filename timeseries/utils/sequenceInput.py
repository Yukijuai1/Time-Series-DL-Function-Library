import torch
import numpy as np

def zerocenter(Sequence, Mean):
    if Mean == -0xffffffff:
        Sequence = Sequence - np.mean(Sequence, axis=(0, 1))
    else:
        Sequence = Sequence - Mean
    return Sequence


def zscore(Sequence, Mean, StandardDeviation):
    if Mean == -0xffffffff:
        Mean = np.mean(Sequence, axis=(0, 1))
    if StandardDeviation == -0xffffffff:
        StandardDeviation = np.std(Sequence, axis=(0, 1))
    Sequence = (Sequence - Mean) / StandardDeviation
    return Sequence


def rescale_symmetric(Sequence, Min, Max):
    if Min == []:
        Min = np.min(Sequence, axis=(0, 1))
    if Max == []:
        Max = np.max(Sequence, axis=(0, 1))

    assert Sequence.shape[-1] == Max.shape[0] == Min.shape[0]

    scaled_Sequence = (2 * (Sequence - Min)) / (Max - Min) - 1
    mask = Sequence >= Max
    scaled_Sequence[mask] = 1
    mask = Sequence <= Min
    scaled_Sequence[mask] = -1
    return scaled_Sequence


def rescale_zero_one(Sequence, Min, Max):
    if Min == []:
        Min = np.min(Sequence, axis=(0, 1))
    if Max == []:
        Max = np.max(Sequence, axis=(0, 1))


    assert Sequence.shape[-1] == Max.shape[0] == Min.shape[0]

    scaled_Sequence = (Sequence - Min) / (Max - Min)
    mask = Sequence >= Max
    scaled_Sequence[mask] = 1
    mask = Sequence <= Min
    scaled_Sequence[mask] = 0
    return scaled_Sequence

def sequenceInput(
    data,
    label,
    Normalization='none',
    Mean=-0xffffffff,
    StandardDeviation=-0xffffffff,
    Min=[],
    Max=[]
):
    assert type(data) == np.ndarray and type(label) == np.ndarray
    assert data.shape[0] == label.shape[0]
    if Normalization == 'zerocenter':
        data = zerocenter(data, Mean)
    elif Normalization == 'zscore':
        data = zscore(data, Mean, StandardDeviation)
    elif Normalization == 'rescale_symmetric':
        data = rescale_symmetric(data, Min, Max)
    elif Normalization == 'rescale_zero_one':
        data = rescale_zero_one(data, Min, Max)
    else:
        pass
    data = torch.from_numpy(data).to(dtype=torch.float32)
    label = torch.from_numpy(label).to(dtype=torch.float32)
    return data, label

