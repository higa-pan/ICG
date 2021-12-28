import numpy as np


# 推定されたQ値を元にソフトマックス方式で行動を決める
def softmax(values, rand0_to_1, Tenperture=1.0, Num_args=4):
    softmax_value = 0.0
    exp_out = np.exp(values / Tenperture)
    sum_exp_out = np.sum(exp_out)
    for action in range(Num_args):
        softmax_value += exp_out[0][action] / sum_exp_out  # valuesの型が1*n行列の形なのでexp_out[0][action]という参照になってる
        if softmax_value >= rand0_to_1:  # ランダム値よりソフトマックス値が大きくなったら, その行動を返す
            return action
