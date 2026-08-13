import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    pos_sin = []

    for pos in range(seq_len):
        curr_pos = []
        for i in range(d_model):
            tmp_i = (i//2)*2
            wq = pos/np.power(base, tmp_i/d_model)
            if i%2 == 0:
                curr_pos.append(np.sin(wq))
            else:
                curr_pos.append(np.cos(wq))
        pos_sin.append(curr_pos)


    return np.array(pos_sin)
        