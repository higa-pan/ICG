import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import const_value as const

root_path = Path.cwd().resolve().parent
print(root_path)
seeds_path = root_path / "random_result_data" / "seeds.csv"
seeds = pd.read_csv(str(seeds_path))['seeds'].tolist()
data_path = root_path / "random_result_data"
save_path = root_path / "visualization"

GROUP_NUM = 3

Path.mkdir(save_path)
group_payoff = []
agent_payoff = []

path_g = data_path / "seed{0}_result".format(seeds[0]) / "group{0}_payoff".format(
    GROUP_NUM)
path_a = data_path / "seed{0}_result".format(seeds[0]) / "agent{0}_payoff".format(
    GROUP_NUM)

fg = pd.read_csv(str(path_g) + ".csv".format(seeds[0]))
fa = pd.read_csv(str(path_a) + ".csv".format(seeds[0]))
group_payoff.append(np.array(fg["Group-payoff"]))
agent_payoff.append(np.array(fa["Agent-payoff"]))

for seed in seeds:
    if seed == seeds[0]:
        continue

    path_g = data_path / "seed{0}_result".format(seed) / "group{0}_payoff".format(
        GROUP_NUM)
    path_a = data_path / "seed{0}_result".format(seed) / "agent{0}_payoff".format(
        GROUP_NUM)
    d_g = pd.read_csv(str(path_g) + ".csv".format(seed))
    d_a = pd.read_csv(str(path_a) + ".csv".format(seed))
    group_payoff[0] += np.array(d_g["Group-payoff"])
    agent_payoff[0] += np.array(d_a["Agent-payoff"])

group_payoff[0] /= len(seeds)
agent_payoff[0] /= len(seeds)

data_frame = pd.DataFrame({"Group-payoff": group_payoff[0].tolist()})
data_frame.to_csv(str(save_path / "group_ava.csv"))
data_frame = pd.DataFrame({"Agent-payoff": agent_payoff[0].tolist()})
data_frame.to_csv(str(save_path / "agent_ava.csv"))

file_group_number = 3

group_max = 76
group_min = -76
agent_max = 86
agent_min = -86

path_g = save_path / "group_ava.csv"
path_a = save_path / "agent_ava.csv"

fg = pd.read_csv(str(path_g))
fa = pd.read_csv(str(path_a))
group_payoff = np.array(fg["Group-payoff"])
agent_payoff = np.array(fa["Agent-payoff"])

with open(str(save_path) + "/Statistical_data.txt", mode='w') as f:
    f.write("group_payoff\n")
    f.write("平均: {0}\n".format(np.mean(group_payoff)))
    f.write("中央値: {0}\n".format(np.median(group_payoff)))
    f.write("標準偏差: {0}\n".format(np.std(group_payoff)))
    f.write("分散: {0}\n".format(np.var(group_payoff)))
    f.write("-----------------------------------------------\n")
    f.write("agent_payoff\n")
    f.write("平均: {0}\n".format(np.mean(agent_payoff)))
    f.write("中央値: {0}\n".format(np.median(agent_payoff)))
    f.write("標準偏差: {0}\n".format(np.std(agent_payoff)))
    f.write("分散: {0}\n".format(np.var(agent_payoff)))

x_axis = np.linspace(1, file_group_number, file_group_number)

fig_g = plt.figure()
fig_a = plt.figure()
ax_g = fig_g.add_subplot(111, ylim=(group_min, group_max))
ax_a = fig_a.add_subplot(111, ylim=(agent_min, agent_max))

ax_g.bar(x_axis, group_payoff)
ax_g.set_xlabel("group_number")
ax_g.set_ylabel("payoff")

ax_a.bar(x_axis, agent_payoff)
ax_a.set_xlabel("agent_number")
ax_a.set_ylabel("payoff")

fig_g.savefig(str(save_path) + '/group_result.png', bbox_inches="tight", pad_inches=0.05)
fig_a.savefig(str(save_path) + '/agent_result.png', bbox_inches="tight", pad_inches=0.05)

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

    group_coop_counter = [[pd_group_coop_list[i].count(j) for j in range(3)] for i in range(3)]

    for i in range(3):
        coop_g_fig.bar([0, 1, 2], group_coop_counter[i])
        fig_g_coop.savefig(str(seed_path) + '/group{}_coopTimes.png'.format(i), bbox_inches="tight", pad_inches=0.05)
        coop_g_fig.cla()


# グループ,エージェントの協力人数,協力先のプロット
seed_path = save_path / ("seed" + str(seeds[0]) + "_graph")
start = 1
x_step = 100

pd_agent_coop_list = pd.read_csv(str(
    data_path / "seed{0}_result".format(seeds[0]) / "agent_coop_times_{0}-{1}step.csv".format(
        const.SIMULATION_TIMES - 10000, const.SIMULATION_TIMES))).values.tolist()
pd_group_coop_list = pd.read_csv(str(
    data_path / "seed{0}_result".format(seeds[0]) / "group_coop_times_{0}-{1}step.csv".format(
        const.SIMULATION_TIMES - 10000, const.SIMULATION_TIMES))).values.tolist()

plot_1 = plt.figure()
plot_agent_graph = plot_1.add_subplot(111)
plot_2 = plt.figure()
plot_group_graph = plot_2.add_subplot(111)

fig, group_axes = plt.subplots(nrows=3, ncols=1, sharex=False)
fig1, agent_axes = plt.subplots(nrows=3, ncols=1, sharex=False)

x_axis_last = np.arange(const.SIMULATION_TIMES - x_step + 1, const.SIMULATION_TIMES + 1, 1)

for data_row in range(len(pd_agent_coop_list)):
    # エージェント、グループの協力の遷移をプロット
    plot_agent_graph.plot(x_axis_last,
                          pd_agent_coop_list[data_row][10000 - x_step + 1:10000 + 1],
                          label=str(data_row))
    plot_group_graph.plot(x_axis_last,
                          pd_group_coop_list[data_row][10000 - x_step + 1:10000 + 1],
                          label=str(data_row))
    # グループそれぞれの協力の遷移をプロット
    group_axes[data_row].plot(x_axis_last, pd_group_coop_list[data_row][10000 - x_step + 1:10000 + 1],
                              label=str(data_row))
    # エージェントそれぞれの協力の遷移をプロット
    agent_axes[data_row].plot(x_axis_last, pd_agent_coop_list[data_row][10000 - x_step + 1:10000 + 1],
                              label=str(data_row))

plt.legend(loc='upper right')
plot_1.savefig(str(seed_path / "agent_{0}_{1}_mix.png".format(const.SIMULATION_TIMES - x_step, const.SIMULATION_TIMES)),
               bbox_inches="tight", pad_inches=0.05)
plot_2.savefig(str(seed_path / "group_{0}_{1}_mix.png".format(const.SIMULATION_TIMES - x_step, const.SIMULATION_TIMES)),
               bbox_inches="tight", pad_inches=0.05)
fig.savefig(str(seed_path / "group_{0}_{1}.png".format(const.SIMULATION_TIMES - x_step, const.SIMULATION_TIMES)),
            bbox_inches="tight", pad_inches=0.05)
fig1.savefig(str(seed_path / "agent_{0}_{1}.png".format(const.SIMULATION_TIMES - x_step, const.SIMULATION_TIMES)),
             bbox_inches="tight", pad_inches=0.05)

"""
start = 1
end = 100
for i in range(0, 1000000, 10000):
    pd_agent_coop_list = pd.read_csv(str(data_path/"seed{0}_result".format(seeds[0])/"agent_coop_times_{0}-{1}step.csv".format(i, i+10000))).values.tolist()
    pd_group_coop_list = pd.read_csv(str(data_path / "seed{0}_result".format(
        seeds[0]) / "group_coop_times_{0}-{1}step.csv".format(i, i + 10000))).values.tolist()

    plot_1 = plt.figure()
    plot_agent_graph = plot_1.add_subplot(111)
    plot_2 = plt.figure()
    plot_group_graph = plot_2.add_subplot(111)

    x_axis_last = np.arange(i+10000-end, i+10000+1, 1)

    for data_row in range(len(pd_agent_coop_list)):
        #print(len(pd_agent_coop_list[data_row][i + start:i + end + 1]))
        plot_agent_graph.plot(x_axis_last, pd_agent_coop_list[data_row][10000 - end:10000 + 1], label=str(data_row))
        plot_group_graph.plot(x_axis_last, pd_group_coop_list[data_row][10000 - end:10000 + 1], label=str(data_row))
    plt.legend(loc='upper right')
    plot_1.savefig(str(seed_path/"agent_{0}_{1}.png".format(i + 10000 - end, i + 10000)), bbox_inches="tight", pad_inches=0.05)
    plot_2.savefig(str(seed_path / "group_{0}_{1}.png".format(i + 10000 - end, i + 10000)), bbox_inches="tight",
                   pad_inches=0.05)

    plot_agent_graph.cla()
    plot_group_graph.cla()
    """
# print("fin")
