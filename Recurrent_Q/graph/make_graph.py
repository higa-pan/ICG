import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import const as const

root_path = Path.cwd().resolve().parent.parent
print(root_path)
seeds_path = root_path / "rnn_Help_result_data" / "seeds.csv"
seeds = pd.read_csv(str(seeds_path))['seeds'].tolist()
data_path = root_path / "rnn_Help_result_data"
save_path = root_path / "visualization"
Path.mkdir(save_path)

GROUP_NUM = 3

# csvファイルを読み込み、平均値を計算

group_payoff = np.zeros((len(seeds), 3))
agent_payoff = np.zeros(len(seeds))

for n, seed in enumerate(seeds):
    group_payoff_path = data_path / "seed{0}_result".format(seed) / "group{0}_payoff.csv".format(GROUP_NUM)
    agent_payoff_path = data_path / "seed{0}_result".format(seed) / "agent{0}_payoff.csv".format(GROUP_NUM)

    group_payoff[n] = np.array(pd.read_csv(str(group_payoff_path))["Group-payoff"])
    agent_payoff[n] = np.array(pd.read_csv(str(agent_payoff_path))["Agent-payoff"])

group_payoff_means = group_payoff.mean(axis=0)
agent_payoff_means = agent_payoff.mean(axis=0)
data_frame = pd.DataFrame({"Group-payoff": group_payoff_means.tolist()})
data_frame.to_csv(str(save_path / "group_ava.csv"))
data_frame = pd.DataFrame({"Agent-payoff": agent_payoff_means.tolist()})
data_frame.to_csv(str(save_path / "agent_ava.csv"))

# 各統計値を計算し保存

with open(str(save_path) + "/Statistical_data.txt", mode='w') as f:
    f.write("group_payoff\n")
    f.write("平均: {0}\n".format(np.mean(group_payoff_means)))
    f.write("中央値: {0}\n".format(np.median(group_payoff_means)))
    f.write("標準偏差: {0}\n".format(np.std(group_payoff_means)))
    f.write("分散: {0}\n".format(np.var(group_payoff_means)))
    f.write("-----------------------------------------------\n")
    f.write("agent_payoff\n")
    f.write("平均: {0}\n".format(np.mean(agent_payoff_means)))
    f.write("中央値: {0}\n".format(np.median(agent_payoff_means)))
    f.write("標準偏差: {0}\n".format(np.std(agent_payoff_means)))
    f.write("分散: {0}\n".format(np.var(agent_payoff_means)))

# グラフ描画用の設定値

file_group_number = 3
group_max = 76
group_min = -76
agent_max = 86
agent_min = -86

# 各グループとエージェントが取得した利得の棒グラフを作成

x_axis = np.linspace(1, file_group_number, file_group_number)

fig_g = plt.figure()
fig_a = plt.figure()
ax_g = fig_g.add_subplot(111, ylim=(group_min, group_max))
ax_a = fig_a.add_subplot(111, ylim=(agent_min, agent_max))

ax_g.bar(x_axis, group_payoff_means)
ax_g.set_xlabel("group_number")
ax_g.set_ylabel("payoff")

ax_a.bar(x_axis, agent_payoff_means)
ax_a.set_xlabel("agent_number")
ax_a.set_ylabel("payoff")

fig_g.savefig(str(save_path) + '/group_result.png', bbox_inches="tight", pad_inches=0.05)
fig_a.savefig(str(save_path) + '/agent_result.png', bbox_inches="tight", pad_inches=0.05)

# 各グループに集まった協力回数をグラフ化

fig_g_coop = plt.figure()
coop_g_fig = fig_g_coop.add_subplot(111)
coop_g_fig.set_xlabel("cooperation times")
coop_g_fig.set_ylabel("count")
for seed in seeds:
    seed_path = save_path / ("seed" + str(seed) + "_graph")
    Path.mkdir(seed_path)

    pd_group_coop_list = pd.read_csv(str(data_path / "seed{0}_result".format(
        seed) / "group_coop_times_{0}-{1}step.csv".format(const.SIMULATION_TIMES - 10000,
                                                          const.SIMULATION_TIMES))).values.tolist()

    group_coop_counter = [[pd_group_coop_list[i][1:].count(j) for j in range(3)] for i in range(3)]

    for i in range(3):
        coop_g_fig.bar([0, 1, 2], group_coop_counter[i])
        fig_g_coop.savefig(str(seed_path) + '/group{}_coopTimes.png'.format(i), bbox_inches="tight", pad_inches=0.05)
        coop_g_fig.cla()
