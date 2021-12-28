import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

num_seed = 50

root_path = Path.cwd().resolve().parent.parent
print(root_path)

result_path = root_path
save_path = result_path / "visualization_all_seeds"
Path.mkdir(save_path)

data_path = result_path / "rnn_Help_result_data"
seeds_path = data_path / "seeds.csv"
seeds = pd.read_csv(str(seeds_path))['seeds'].tolist()

fig, group_axes = plt.subplots(nrows=3, ncols=1, sharex=True)
fig1, agent_axes = plt.subplots(nrows=3, ncols=1, sharex=True)

x = np.arange(1, 101)

for seed in seeds[:num_seed]:
    Path.mkdir(save_path / "_seed{}".format(seed))
    save_path2 = save_path / "_seed{}".format(seed) / "coop_times_seq"
    Path.mkdir(save_path2)
    save_path3 = save_path / "_seed{}".format(seed) / "group_coop_times"
    Path.mkdir(save_path3)
    for i in range(100):
        agent_coop_data = pd.read_csv(str(
            data_path / "seed{0}_result".format(seed) / "agent_coop_times_{0}-{1}step.csv".format(
                i * 10000, (i + 1) * 10000))).values[:, 9901:]

        group_coop_data = pd.read_csv(str(
            data_path / "seed{0}_result".format(seed) / "group_coop_times_{0}-{1}step.csv".format(
                i * 10000, (i + 1) * 10000))).values[:, 9901:]
        

        # グループそれぞれの協力の遷移をプロット
        group_axes[0].plot(x, group_coop_data[0, :])
        group_axes[1].plot(x, group_coop_data[1, :])
        group_axes[2].plot(x, group_coop_data[2, :])

        # エージェントそれぞれの協力の遷移をプロット
        agent_axes[0].plot(x, agent_coop_data[0, :])
        agent_axes[1].plot(x, agent_coop_data[2, :])
        agent_axes[2].plot(x, agent_coop_data[1, :])

        fig.savefig(str(save_path2 / "group_{0}_{1}.png".format((i + 1) * 10000 - 100, (i + 1) * 10000)),
                    bbox_inches="tight", pad_inches=0.05)
        fig1.savefig(str(save_path2 / "agent_{0}_{1}.png".format((i + 1) * 10000 - 100, (i + 1) * 10000)),
                     bbox_inches="tight", pad_inches=0.05)

        for data_row in range(3):
            group_axes[data_row].cla()
            agent_axes[data_row].cla()
