"""
python blist-test.py

list creation times: [9.358998635, 9.171715939, 9.334420755]
blist creation times: [10.672724386999999, 11.64826429, 11.026489831999996]

list slicing times: [380.934246628, 378.24431810799996, 367.74622772500004]
blist slicing times: [12.468614235000132, 11.690742849999879, 11.607023963999836]

Process finished with exit code 0
"""
import timeit
from random import random, randrange
from blist import blist

N = 1000000


def create_list():
    return [random() for _ in range(N)]


def create_blist():
    return blist([random() for _ in range(N)])


def slice_list():
    return make_slices(create_list())


def slice_blist():
    return make_slices(create_blist())


def make_slices(float_list):
    for _ in range(1000):
        r = (randrange(0, N), randrange(0, N))
        s = min(r)
        e = max(r)
        float_list[s:e]


if __name__ == '__main__':
    print('list creation times: ', end='')
    print(timeit.repeat(create_list, setup='from __main__ import create_list', number=100, repeat=3))
    print('blist creation times: ', end='')
    print(timeit.repeat(create_blist, setup='from __main__ import create_blist', number=100, repeat=3))
    print()
    print('list slicing times: ', end='')
    print(timeit.repeat(slice_list, setup='from __main__ import slice_list', number=100, repeat=3))
    print('blist slicing times: ', end='')
    print(timeit.repeat(slice_blist, setup='from __main__ import slice_blist', number=100, repeat=3))
