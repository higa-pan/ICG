# NUM_OF_GROUPS は全体のグループ数を表す
NUM_OF_GROUPS = 3
# NUM_OF_AGENTS は全体のエージェント数を表す
NUM_OF_AGENTS = NUM_OF_GROUPS
# NUM_OF_JOINED_GROUPS はエージェントの所属するグループ数を表す
NUM_OF_JOINED_GROUPS = 2
# NUM_OF_MEMBER_IN_GROUPS はグループ内に所属する人数を表す
NUM_OF_MEMBER_IN_GROUPS = 2
# NUM_OF_ACTIONS はエージェントが行える行動の数を表す
NUM_OF_ACTIONS = 2
# SIMULATION_TIMES は一回のゲームあたりのステップ数
SIMULATION_TIMES = 1000000
# SIMULATION_LOOP_TIMES は何シード行うかを表す数
SIMULATION_LOOP_TIMES = 50
# LEARNING_RATE_A はQ学習の学習率を表す
LEARNING_RATE_A = 0.9
# DISCOUNT_RATE_G はQ学習の割引率を表す
DISCOUNT_RATE_G = 0.9
# TEMPERATURE はソフトマックス方式の温度パラメータを表す
TEMPERATURE = 1.0