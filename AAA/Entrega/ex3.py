import numpy as np
import pandas as pd

# P[state_from][observation][state_to]
P = np.array([
    # 'a' transitions           'b' transitions
    [[0.0, 0.2, 0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],  # from state 0
    [[0.0, 0.3, 0.0, 0.3, 0.0], [0.0, 0.0, 0.4, 0.0, 0.0]],  # from state 1
    [[0.0, 0.0, 0.3, 0.0, 0.2], [0.0, 0.0, 0.0, 0.5, 0.0]],  # from state 2
    [[0.0, 0.0, 0.6, 0.0, 0.2], [0.0, 0.0, 0.0, 0.2, 0.0]],  # from state 3
    [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]   # from state 4 (terminal state)
])

def compute_entropy_measures(sequence):
    Q_size = P.shape[0]
    w_len = len(sequence)
    
    H = np.zeros((w_len + 1, Q_size))
    c = np.zeros((w_len + 1, Q_size))
    c[0, 0] = 1.0  # Initial c_0(0)

    for t in range(1, w_len + 1):
        char = sequence[t - 1]
        char_index = 0 if char == 'a' else 1  # 'a' corresponds to 0, 'b' corresponds to 1
        
        for j in range(Q_size):
            numerator = sum(c[t - 1, i] * P[i, char_index, j] for i in range(Q_size))
            denominator = sum(sum(c[t - 1, i] * P[i, char_index, k] for i in range(Q_size)) for k in range(Q_size))
            c[t, j] = numerator / denominator if denominator != 0 else 0
            c[t, j] = round(c[t, j], 3)

            p = np.zeros(Q_size)
            for i in range(Q_size):
                p_numerator = c[t - 1, i] * P[i, char_index, j]
                p_denominator = sum(c[t - 1, k] * P[k, char_index, j] for k in range(Q_size))
                p[i] = p_numerator / p_denominator if p_denominator != 0 else 0
                # p[i] = round(p[i], 3)

            H[t, j] = sum(H[t - 1, i] * p[i] for i in range(Q_size))
            H[t, j] -= sum(p[i] * np.log2(p[i] + 1e-20) for i in range(Q_size) if p[i] > 0)
            H[t, j] = round(H[t, j], 3)

    return H, c

sequence = "ababaa"
H, c = compute_entropy_measures(sequence)

df_H = pd.DataFrame(H, columns=[f'H_t({i})' for i in range(H.shape[1])])
df_c = pd.DataFrame(c, columns=[f'c_t({i})' for i in range(c.shape[1])])

df_H['Sequence'] = [''] + list(sequence)
df_H.set_index('Sequence', inplace=True)
df_c['Sequence'] = [''] + list(sequence)
df_c.set_index('Sequence', inplace=True)

print("H_t(j) Table")
print(df_H)
print("\nc_t(j) Table")
print(df_c)
