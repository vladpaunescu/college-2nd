from random import random

def subtract(perm1, perm2):
    l1, l2 = len(perm1), len(perm2)

    if l1 != l2:
        raise Exception("Different perm {} {}".format(perm1, perm2))

    swap_seq = []
    p1 = perm1[:]
    i = 0
    while p1 != perm2 and i < l1:
        id1 = p1.index(i)
        id2 = perm2.index(i)
        if id1 != id2:
            swap_seq.append((id1, id2))
            p1 = swap(p1, (id1, id2))
        i += 1

    return tuple(swap_seq)


def apply_swap_seq(perm, ss):
    for s in ss:
        perm = swap(perm, s)
    return perm


def swap(perm, so):
    tmp = perm[so[0]]
    perm[so[0]] = perm[so[1]]
    perm[so[1]] = tmp
    return perm

"""
    concatenates 2 swap sequences
    ((2,1),(3,2))+((2,5),(6,4),(5,6))=((2,1),(3,2),(2,5),(6,4),(5,6))
"""
def swap_concat(ss1, ss2):
    print ss1
    result = list(ss1)
    for s in ss2:
        if s not in ss1:
            result.append(s)

    return tuple(result)

def prob_concat(ss1, ss2, prob):
    result = list(ss1)
    for s in ss2:
        p = random()
        if s not in ss1 and p < prob:
            result.append(s)

    return tuple(result)


if __name__ == "__main__":
    dif = subtract([0, 2, 3, 4, 1], [2, 0, 1, 4, 3])
    print dif
    concat = swap_concat(((2,1),(3,2)), ((2,5),(6,4),(3,2),(5,6)))
    print concat
    concat = prob_concat(((2,1),(3,2)), ((2,5),(6,4),(3,2),(5,6)), 0.9)
    print concat

    print apply_swap_seq([0, 2, 3, 4, 1], dif)