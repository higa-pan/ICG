from simulation import Simulation
from random_seeds import RandomSeeds
import pathlib
import const_value as const

if __name__ == "__main__":
    print("step:", const.SIMULATION_TIMES)
    print("group", const.NUM_OF_GROUPS)
    root = pathlib.Path(__file__)
    abs_root = root.resolve().parent.parent
    data_path = abs_root / "nonHelp_result_data"
    pathlib.Path.mkdir(data_path)

    rand = RandomSeeds()
    rand_seeds = rand.make_seeds(num=const.SIMULATION_LOOP_TIMES)
    rand.seeds_list_to_csv(str(data_path))

    simulate = Simulation(const.NUM_OF_AGENTS, const.NUM_OF_GROUPS)  # Simulation をインスタンス化し、各種初期化(agent・集団の生成など)している

    for rand_seed in rand_seeds:  # 1回の実験をSIMULATION_LOOP_TIMES(用意したseedの数だけ)回行う
        print("{0}seed目".format(rand_seed))

        rand_seed_path = data_path / "seed{0}_result".format(rand_seed)
        pathlib.Path.mkdir(rand_seed_path)  # seedごとの保存ファイルの作成

        # seed・グループの利得、エージェントの内部状態(Qtableなど)を再設定
        simulate.reset_seed(rand_seed)
        simulate.reset_group_payoff()
        simulate.clear_agent_state()

        # 学習が安定する頃までゲームさせる(最後の1万ステップの情報だけ取ろうという狙い)
        simulate.play_game(const.SIMULATION_TIMES - 10000)

        # データをとる最後の1万ステップ
        simulate.reset_sequence_data()              # 1.エージェントの協力先・グループへの協力数、Qtableのデータをリセット
        simulate.play_game_add_q_table(10000)       # 2.ICゲームを行う(行動決定、Qtableの更新を含む)
        simulate.save_sequence_data(const.SIMULATION_TIMES // 10000 - 1, str(rand_seed_path))  # 3.エージェントの協力先・グループへの協力数、Qtableのデータを保存
        """
        for simulation_time in range(const.SIMULATION_TIMES // 5):  # 100万ステップ行うため1万ステップごとに区切っている
            simulate.reset_sequence_data()  # 1.エージェントの協力先・グループへの協力数、Qtableのデータをリセット
            simulate.play_game(5)  # 2.ICゲームを行う(行動決定、Qtableの更新を含む)
            simulate.save_sequence_data(simulation_time, str(rand_seed_path))  # 3.エージェントの協力先・グループへの協力数、Qtableのデータを保存
        """
        simulate.save_payoff_data(str(rand_seed_path))  # エージェント・グループの利得データを保存
