#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

root_path = Path.cwd().resolve().parent
print(root_path)

# ファイル構造
# a-06_r-01 
# |
# | - help
#      |
#      | - rnn_Help_result_data
# |
# | - non_help
#      |
#      | - rnn_non_Help_result_data

num_seed = 50

pra_list = ["a-06_r-01"]
help_nonhelp = ["help", "non_help"]
result_data_dir = ["rnn_Help_result_data", "rnn_non_Help_result_data"]


for pra_name in pra_list:
    for help_or_nonhelp in range(2):
        # 作成したグラフの保存場所
        result_path = root_path / pra_name / help_nonhelp[help_or_nonhelp]
        save_path = result_path / "visualization_all_seeds" / "group_coop_times_std"
        Path.mkdir(save_path)

        # 実験で使った乱数seedたちを取得
        data_path = result_path / result_data_dir[help_or_nonhelp]
        seeds_path = data_path / "seeds.csv"
        seeds = pd.read_csv(str(seeds_path))['seeds'].tolist()

        # 2 * 2 の計4つのグラフを1つの画像として作成する(*けど3つしか使ってない)
        fig_coop, group_coop_axes = plt.subplots(nrows=2, ncols=2, sharex=True)

        # 1万stepごとのグループへの協力人数の平均と標準偏差を棒グラフにする
        for i in range(100):
            group_data_all_seeds_array = np.empty((3, 3, 3))

            # 50seed分のデータを取得
            for n, seed in enumerate(seeds[:num_seed]):
                # データの読み込み
                group_coop_data = pd.read_csv(str(
                    data_path / "seed{0}_result".format(seed) / "group_coop_times_{0}-{1}step.csv".format(
                        i * 10000, (i + 1) * 10000))).values[:, 9901:].tolist()

                # 協力した人数が0人, 1人, 2人の回数を数える
                group_coop_counter = [[group_coop_data[j].count(k) for k in range(3)] for j in range(3)]
                # 平均値，標準偏差を計算するためにストック
                group_data_all_seeds_array[n, :, :] = np.array(group_coop_counter).reshape((1, 3, 3))

            # 平均値, 標準偏差の値を計算する
            mean_array = group_data_all_seeds_array.mean(axis=0)
            std_array = group_data_all_seeds_array.std(axis=0)
            
            # 棒グラフの上限値を設定
            group_coop_axes[0][0].set_ylim(top=60)
            group_coop_axes[0][1].set_ylim(top=60)
            group_coop_axes[1][0].set_ylim(top=60)

            # 平均の棒グラフをプロット
            group_coop_axes[0][0].bar([0, 1, 2], mean_array[0])
            group_coop_axes[0][1].bar([0, 1, 2], mean_array[1])
            group_coop_axes[1][0].bar([0, 1, 2], mean_array[2])

            # error_bar をプロット
            group_coop_axes[0][0].errorbar([0, 1, 2], mean_array[0], std_array[0], linestyle='None', marker='^')
            group_coop_axes[0][1].errorbar([0, 1, 2], mean_array[1], std_array[1], linestyle='None', marker='^')
            group_coop_axes[1][0].errorbar([0, 1, 2], mean_array[2], std_array[2], linestyle='None', marker='^')


            fig_coop.savefig(str(save_path) + '/{}-{}step_coop_times_std.png'.format(i * 10000, (i + 1) * 10000),
                             bbox_inches="tight", pad_inches=0.05)

            # 描画したものを消す
            group_coop_axes[0][0].cla()
            group_coop_axes[0][1].cla()
            group_coop_axes[1][0].cla()


