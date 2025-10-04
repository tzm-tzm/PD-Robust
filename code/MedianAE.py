import numpy as np

def median_absolute_error(y_true, y_pred):
    # 计算绝对差值
    absolute_errors = np.abs(y_true - y_pred)
    # 按绝对差值排序
    sorted_errors = np.sort(absolute_errors)

    # 计算中位数
    n = len(sorted_errors)
    if n % 2 == 0:
        median_error = (sorted_errors[n//2 - 1] + sorted_errors[n//2]) / 2
    else:
        median_error = sorted_errors[(n-1) // 2]
    return median_error
