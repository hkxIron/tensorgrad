import numpy as np
"""
len1<len2: 
    =0:s1,s2不需要broadcast
    <0:s1为需要broadcast的值
    >0:s2需要broadcast
    如：s1 (100,3,2), len=3
        s2 (2), len=1 
    s2需要repeat
    s2需要repeat的维度为:(0,1) = np.arange(2=3-1)
"""
def get_repeat_axis(left_shape: tuple,
                    right_shape: tuple):
    len1 = len(left_shape)
    len2 = len(right_shape)
    left_not_repeat = len1 - len2
    repeat_axis = tuple(np.arange(abs(len1 - len2)))
    return left_not_repeat, repeat_axis

def accumulative_add_by_shape(is_accumulate_not_repeat: int,
                              repeat_axis:tuple,
                              accumulate_add:np.ndarray,
                              to_add:np.ndarray):
    if is_accumulate_not_repeat >= 0: # accumulate shape较大，本身不需要repeat
        accumulate_add += to_add
    else:  # accumulate shape较小，需要对to_add的维进行sum
        accumulate_add += to_add.sum(axis=repeat_axis, keepdims=False)

# one_dims = find_axis_in_shape(out.data.shape, 1)
# if len(one_dims)>0: out.data = out.data.squeeze(axis=one_dims)
def find_axis_in_shape(shape_tuple:tuple,
                       shape_value:int):
    axis = []
    for i, shape in enumerate(shape_tuple):
        if shape == shape_value:
            axis.append(i)
    return tuple(axis)

