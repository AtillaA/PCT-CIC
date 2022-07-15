import numpy as np

def scaler(bound, target_range, value):
    value = np.clip(value, bound[0], bound[1])
    v_std = (value-bound[0]) / (bound[1]-bound[0])
    return v_std * (target_range[1] - target_range[0]) + target_range[0]