
import numpy as np
from math import ceil, floor, sqrt

def partition_column_pairs(N):
    """
    Partition all column pairs (i, j) with i < j into batches.
    Each batch contains disjoint pairs (no column reused).
    Returns:
        List[List[Tuple[int, int]]]
    """
    used_pairs = set()
    batches = []

    all_pairs = {(i, j) for i in range(N) for j in range(i + 1, N)}

    while used_pairs != all_pairs:
        used_cols = set()
        batch = []

        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in used_pairs:
                    continue
                if i in used_cols or j in used_cols:
                    continue
                batch.append((i, j))
                used_pairs.add((i, j))
                used_cols.update([i, j])

        batches.append(batch)
    return batches

def roate(i, j, A, solved, threshold, V):
    alpha = np.dot(A[:, i], A[:, j])
    beta = np.dot(A[:, i], A[:, i])
    gamma = np.dot(A[:, j], A[:, j])

    if (alpha * alpha) / (beta * gamma) < threshold:
        solved += 1
    else:
        q = beta - gamma
        c = sqrt(4 * alpha * alpha + q * q)

        if q >= 0:
            cosi = sqrt((c + q) / (2 * c))
            sine = alpha / (c * cosi)
        else:
            if alpha >= 0:
                sine = sqrt((c - q) / (2 * c))
            else:
                sine = -sqrt((c - q) / (2 * c))
            cosi = alpha / (c * sine)

        tAi = A[:, i].copy()
        A[:, i] = cosi * A[:, i] + sine * A[:, j]
        A[:, j] = -sine * tAi + cosi * A[:, j]

        tVi = V[:, i].copy()
        V[:, i] = cosi * V[:, i] + sine * V[:, j]
        V[:, j] = -sine * tVi + cosi * V[:, j]


# 对小矩阵进行svd，串行的，更新A和V
def svd_sub(A, V):
    A_local = A.copy()
    V_local = V.copy()
    N = A.shape[1]
    for i in range(N):
        for j in range(i, N):
            roate(i, j, A_local, 0, 1e-10, V_local)
    return A_local, V_local
# 找到pair位置
def find_pair_index(pairs, target):
    for pair_index, (a, b) in enumerate(pairs):
        if target == a:
            return pair_index, 0
        elif target == b:
            return pair_index, 1
