import torch
from sklearn import preprocessing as pp

def min_max_scaler(data):
    """
    数据归一化
    :param data:
    :return:
    """
    min_value = torch.min(data)
    max_value = torch.max(data)

    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data


if __name__ == "__main__":
    data = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)
    scaler = pp.MinMaxScaler().fit(data)
    data = scaler.transform(data)
    print(type(data))
