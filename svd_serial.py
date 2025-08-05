import numpy as np


def one_sided_jacobi_svd(A, tol=1e-10, max_iter=100):
    """
    正确的单边Jacobi算法计算矩阵的SVD

    参数:
        A: 输入矩阵 (m x n), m >= n
        tol: 收敛阈值
        max_iter: 最大迭代次数

    返回:
        U: (m x n) 左奇异向量
        S: (n,) 奇异值
        V: (n x n) 右奇异向量
    """
    m, n = A.shape
    B = A.copy()
    V = np.eye(n)

    for _ in range(max_iter):
        converged = True

        for j in range(n):
            for k in range(j + 1, n):
                # 计算B的列j和k的内积
                a = B[:, j]
                b = B[:, k]
                dot = np.dot(a, b)

                # 计算列j和k的范数
                a_norm = np.dot(a, a)
                b_norm = np.dot(b, b)

                # 检查是否已正交
                if abs(dot) > tol * np.sqrt(a_norm * b_norm):
                    converged = False

                    # 计算旋转参数 (使列j和k正交)
                    theta = 0.5 * np.arctan2(2 * dot, a_norm - b_norm)
                    c = np.cos(theta)
                    s = np.sin(theta)

                    # 更新B的列
                    B[:, j] = c * a + s * b
                    B[:, k] = -s * a + c * b

                    # 更新V的列
                    V[:, [j, k]] = V[:, [j, k]] @ np.array([[c, -s], [s, c]])

        if converged:
            break

    # 提取奇异值和U
    S = np.linalg.norm(B, axis=0)
    U = B / S

    # 确保奇异值降序排列
    order = np.argsort(S)[::-1]
    S = S[order]
    U = U[:, order]
    V = V[:, order]

    return U, S, V.T


# 测试
if __name__ == "__main__":
    np.random.seed(42)
    A = np.random.rand(5, 3)

    print("输入矩阵A:")
    print(A)

    U, S, Vt = one_sided_jacobi_svd(A)
    print("\n重构误差:", np.linalg.norm(U @ np.diag(S) @ Vt - A))

    # 与NumPy的SVD对比
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
    print("\nNumPy SVD重构误差:", np.linalg.norm(U_np @ np.diag(S_np) @ Vt_np - A))

    s, v, d = np.linalg.svd(A)

    print()

    8