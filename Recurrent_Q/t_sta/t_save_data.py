import pandas as pd
from pathlib import Path
import numpy as np


root_path = Path.cwd().resolve()
print(root_path)

num_seed = 50

save_path = root_path

data_path = root_path / "help_result_data"
seeds_path = data_path / "seeds.csv"
seeds = pd.read_csv(str(seeds_path))['seeds'].tolist()

i = 99
group_data_all_seeds_array = np.empty((num_seed, 3, 3))
for seed_index, seed in enumerate(seeds[:num_seed]):
    group_coop_data = pd.read_csv(str(
        data_path / "seed{0}_result".format(seed) / "group_coop_times_{0}-{1}step.csv".format(
            i * 10000, (i + 1) * 10000))).values[:, 9901:].tolist()

    # 協力数がそれぞれいくつあったかをカウントする(3groupそれぞれ0, 1, 2人協力が集まったかをカウントする)
    group_coop_counter = [[group_coop_data[j].count(k) for k in range(3)] for j in range(3)]

    # 平均値，標準偏差を計算するためにストック
    group_data_all_seeds_array[seed_index, :, :] = np.array(group_coop_counter).reshape((1, 3, 3))


np.save("4ac_rnn_stadata.npy", group_data_all_seeds_array)
