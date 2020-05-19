import ctypes


def dec2bin(x, signed) -> str:
    """
    get binary representation of int32/uint32
    you should rewrite the code to avoid built-in f-string conversions
    """
    if signed:
        sign = False
        if x < 0:
            sign = True
            _x = -x
        else:
            _x = x
        res_str = ''
        while _x > 0:
            res_str = str(_x % 2) + res_str
            _x = _x // 2
        if sign:
            while len(res_str) < 32:
                res_str = '1' + res_str
        else:
            while len(res_str) < 32:
                res_str = '0' + res_str
        return res_str
    else:
        _x = x
        res_str = ''
        while _x > 0:
            res_str = str(_x % 2) + res_str
            _x = _x // 2
        while len(res_str) < 32:
            res_str = '0' + res_str
        return res_str


def bin2dec(x, signed) -> int:
    """
    get integer from it's binary representation
    you should rewrite the code to avoid built-in int(x, 2) conversion
    """
    _x = 0
    for i in range(len(x)):
        if x[i] == '1':
            _x += pow(2,32-i-1)
    return _x - 2**32 * int(x[0]) * signed


def float2bin(x) -> str:
    """
    get binary representation of float32
    you may leave it as is
    """
    return f'{ctypes.c_uint32.from_buffer(ctypes.c_float(x)).value:>032b}'


def bin2float(x) -> float:
    """
    get float from it's binary representation
    you should rewrite the code to avoid ctypes conversions
    """
    _x = 0
    sign, exp, mantis = x[:1], int(x[1:9], 2) - 127, x[9:32]
    _x += pow(2, exp)
    for i in range(len(mantis)):
        if mantis[i] == '1':
            _x += pow(2, exp-i-1)
    if sign=='1':
        _x = -_x
        print(_x, 'lil', sign)
    else:
        print(_x, 'lil')
    print(ctypes.c_float.from_buffer(ctypes.c_uint32(int(x, 2))).value, 'you')
    return _x
