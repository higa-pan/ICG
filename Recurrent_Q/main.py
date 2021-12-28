from simulation.simulation import Simulation
from random_seeds import RandomSeeds
import pathlib
from const import const_value as const


if __name__ == "__main__":
    print("step:", const.SIMULATION_TIMES)
    print("group", const.NUM_OF_GROUPS)
    root = pathlib.Path(__file__)
    abs_root = root.resolve().parent.parent
    data_path = abs_root / "rnn_Help_result_data"
    pathlib.Path.mkdir(data_path)

    rand = RandomSeeds()
    rand_seeds = rand.make_seeds(num=const.SIMULATION_LOOP_TIMES)
    rand_seeds = rand_seeds[:10]
    rand.seeds_list_to_csv(str(data_path))

    simulate = Simulation(const.NUM_OF_AGENTS, const.NUM_OF_GROUPS)  # Simulation をインスタンス化し、各種初期化(agent・集団の生成など)している

    for rand_seed in rand_seeds:  # 1回の実験をSIMULATION_LOOP_TIMES(用意したseedの数だけ)回行う
    
        print("{0}seed目".format(rand_seed))

        rand_seed_path = data_path / "seed{0}_result".format(rand_seed)
        pathlib.Path.mkdir(rand_seed_path)  # seedごとの保存ファイルの作成

        # seed・グループの利得、エージェントの内部状態(rnnの重みなど)を再設定
        simulate.reset_seed(rand_seed)
        simulate.reset_group_payoff()
        simulate.clear_agent_state()

        for term in range(const.SIMULATION_TIMES // const.LEARNING_SPAN):  # 100万 / 1万 で100回シミュレーションを行う処理でいいかなと思う
            print("{}step目".format(term))
            simulate.save_weight_data(term, str(rand_seed_path))
            simulate.no_save_play_game(const.LEARNING_SPAN - const.LEARN_STEP)
            simulate.play_game(const.LEARN_STEP)

            simulate.update_weight()  # 10000回分の入力と教師データを学習させる
            simulate.save_sequence_data(term, str(rand_seed_path))  # 集団に何人協力したか，エージェントがどこに協力したかを保存する
            simulate.reset_sequence_data()

        simulate.save_payoff_data(str(rand_seed_path))  # エージェント・グループの利得データを保存
