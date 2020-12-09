# coding=utf-8


def parse_maxint_from_str(s):
    max_v = -1 * float("inf")
    now_v = 0
    has_number_flag = False
    flag = False
    ll = len(s)
    for i in range(ll):
        if s[i].isdigit():
            has_number_flag = True
            now_v = now_v * 10 + int(s[i])
        else:
            if has_number_flag:
                if flag:
                    now_v = now_v * -1
                max_v = max(max_v, now_v)
            if s[i] == '-':
                flag = True
            else:
                flag = False
            has_number_flag = False
            now_v = 0
    if has_number_flag:
        if flag:
            now_v = now_v * -1
        max_v = max(max_v, now_v)
    return max_v




