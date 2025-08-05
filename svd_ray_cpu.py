import ray
import numpy as np

from svd.tools import partition_column_pairs, svd_sub, find_pair_index

N = 4
# 分
# batches = partition_column_pairs(N)
M = 8

batches = [
    [(0, 1), (2, 3)],
    [(0, 2), (1, 3)],
    [(1, 2), (0, 3)],
    ]
print(batches)

# # 初始化 Ray
ray.init(ignore_reinit_error=True)

A = np.random.rand(2, M)
#
# u, s, v = np.linalg.svd(A)
#
# print(s[:10])

indices = np.array_split(range(M), N)

print()


@ray.remote
class JacobiWorker:
    def __init__(self, A1, A2, i, j):
        self.A1 = A1
        self.A2 = A2
        self.blockId1 = i
        self.blockId2 = j

        self.U1 = np.zeros_like(A1)
        self.U2 = np.zeros_like(A2)

    def get_data(self):
        return self.A1, self.A2

    def roate(self):
        print()

    # 初始化
    def init(self):

        print("A1", self.A1[0][0])
        self.A1, self.U1 = svd_sub(self.A1, self.U1)
        self.A2, self.U2 = svd_sub(self.A2, self.U2)
        print("A2", self.A1[0][0])
        # print(self.A2)
        print()
        return self.blockId1, self.blockId2, self.A1, self.A2, self.U1, self.U2

    # 获取数据
    def get_data(self, flag):
        # 制定获取那个数据，flag 0表示A1 1表示A2
        if flag == 0: #
            return [self.blockId1, self.A1, self.U1]
        else:
            return [self.blockId2, self.A2, self.U2]
        # return self.data

    # 获得这一个
    def set_data(self, pos, data_ref):
        print(type(data_ref))
        res = ray.get(data_ref) # 获取data
        # blockId1
        print(A.shape)
        # A = data[0]
        # U = data[1]
        if pos == 0:
            self.A1, self.U1 = A, U
        else:
            self.A2, self.U2 = A, U
        return 0


    def get_all_data(self):
        return

# 创建 worker，每对索引(i, j) 对应一个 worker，传入对应子矩阵
workers = []
# for batch in batches:
# 初始化
for (i, j) in batches[0]:
    A1 = A[:, indices[i]]
    A2 = A[:, indices[j]]
    worker = JacobiWorker.remote(A1, A2, i, j)
    workers.append(worker)

# 第一次计算矩阵自己内部的更新


A_blocks = []
V_blocks = []


data_readyfor_change = []

# 给出batches的所有序列可能，循环序列，然后计算前一个序列和下一个序列的差，序列和序列一个pair只传递一个值
for i in range(len(batches) - 1):
    row1 = batches[i]
    row2 = batches[i + 1]
    for idx, (pair1, pair2) in enumerate(zip(row1, row2)):
        if pair1[0] != pair2[0]:  # 如果变化的是左边，就获取，然后暂存，然后赋值
            preworkerid, pos1 = idx, 0
            nextworkerid, pos2 = find_pair_index(row2, pair1[0])
            # print("从第", preworkerid, "获取到了第" + pos1 + "个位置的数据传输到" +  +)
        else: #如果第二个不一样
            preworkerid, pos1 = idx, 1
            nextworkerid, pos2 = find_pair_index(row2, pair1[1])  #
        data_ref = workers[preworkerid].get_data.remote(pos1)
        data_readyfor_change.append([data_ref, nextworkerid, pos2])  # 数据，要送到的位置和左右
    # 开始切换

    # ray.get(data_readyfor_change)
    print("传递结束")

    futures = []
    for data_ref, worker_id, pos in data_readyfor_change:
        futures.append(workers[worker_id].set_data.remote(pos, data_ref))




    res = ray.get(futures)

    futures = [worker.get_all_data.remote for worker in workers]

    res = ray.get(futures)

    print(res)




# 初始化，更新每个子块A_i
futures = [worker.init.remote() for worker in workers]






# 然后开始根据batches去更新

for item in res:
    _, _, A1, A2, V1, V2 = item
    A_blocks.extend([A1, A2])
    V_blocks.extend([V1, V2])

A = np.concatenate(A_blocks, axis=1)
V = np.concatenate(V_blocks, axis=1)

# 提取奇异值和U
S = np.linalg.norm(A, axis=0)
U = V / S

# 确保奇异值降序排列
order = np.argsort(S)[::-1]
S = S[order]
U = U[:, order]
V = V[:, order]





print(S[:10])



# # 示例：获取第一个 worker 的数据
# A1_result, A2_result = ray.get(workers[0].get_data.remote())
# print("A1 shape:", A1_result.shape)
# print("A2 shape:", A2_result.shape)