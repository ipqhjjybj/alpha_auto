# coding=utf-8


def is_arr_sorted(arr):
    """
    该取值是否是明显的值
    """
    for i in range(2, len(arr)):
        if (arr[i] - arr[i-1]) * (arr[i-1] - arr[i-2]) < 0:
            return False
    return True
