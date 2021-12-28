import const_value as const


class Agent:
    def __init__(self, my_agent_number):
        self.my_number = my_agent_number
        # group_member_mapに所属グループごとのメンバーを記録する
        self.group_member_map = {}

        # [どのグループに協力するのか, 誰に協力要請するのか]が入る
        self.cooperation_group = 0
        # ゲームで自身が得た報酬が入る
        self.get_reward = 0.0
        # ゲームで得た報酬の合計が入る
        self.payoff = 0.0
        # エージェント0は[協力した、してない],
        # エージェント1は[協力した、してない],
        # [自分が取るべき行動] という順序で参照するように作ってる

    def choose_first_cooperate_group(self, rand0_or_1):
        self.cooperation_group = list(self.group_member_map.keys())[rand0_or_1]

    # 所属集団とそのメンバー以外の変数を初期化する
    def clear_vars(self, rand0_or_1):
        self.get_reward = 0.0
        self.payoff = 0.0
        self.choose_first_cooperate_group(rand0_or_1)

    # 所属集団のどちらに協力するのかを決める
    def decide_action(self, rand_0_to_1):
        if rand_0_to_1 > 0.5:
            other_groups = list(self.group_member_map.keys())
            other_groups.remove(self.cooperation_group)
            self.cooperation_group = other_groups[0] 
