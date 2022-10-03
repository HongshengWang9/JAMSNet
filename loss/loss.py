import torch


def cal_negative_pearson(x, y):
    " Negative Pearson loss function, x is the predicted value, y is the true value "
    n = len(x)
    sum_x = torch.sum(x)
    sum_y = torch.sum(y)
    sum_xy = torch.sum(torch.mul(x, y))
    sum_x2 = torch.sum(x.pow(2))
    sum_y2 = torch.sum(y.pow(2))
    molecular = n * sum_xy - torch.mul(sum_x, sum_y)
    denominator = torch.sqrt((sum_x2 * n - sum_x.pow(2))*(n * sum_y2 - sum_y.pow(2)))
    return (1 - molecular/denominator)