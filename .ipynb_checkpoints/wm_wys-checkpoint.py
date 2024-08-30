import pandas as pd
import hashlib
import numpy as np

filename = "./pre-do-urban.csv"
df = pd.read_csv(filename)
origin_mean = df.mean()
origin_var = df.var()
print(df.describe())
wm = "1010101000101000100010010000101010001010001010100001"
lsbf = 0.3
gamma = 1009
np.random.seed(0)
k_set = key_list = np.random.randint(0, 0xffff, size=10)
print(f'secret keys: {k_set}')

# Hash function
def H(k:int, v) -> int:
    h = hashlib.sha256()
    h.update((str(k) + str(v)).encode())
    return int(h.hexdigest(), 16)

# embedding function
def em(val, w, sk, idx, col):
    if type(val) == np.int64:
        bin_val = list(bin(abs(val)))[2:]
        lsb = round(len(bin_val) * lsbf)
        lsb = 1 if lsb == 0 else lsb
        i = H(sk, idx) % lsb  
        bin_val[-i-1] = w
        if val < 0:
            return np.int64(-int(''.join(bin_val), 2))
        else: 
            return np.int64(int(''.join(bin_val), 2))
    elif type(val) == np.float64:
        bin_val = str(val).split('.')
        frac_bin = list(bin(int(bin_val[1])))[2:]
        lsb = round(len(frac_bin) * lsbf)
        lsb = 1 if lsb == 0 else lsb
        i = H(sk, idx) % lsb 
        frac_bin[-i-1] = w
        val_list = bin_val[0] + '.' + str(int(''.join(frac_bin), 2))
        return np.float64(float(val_list))

# detection function
def de(val,sk,idx):
    if type(val) == np.int64:
        bin_val = list(bin(val))[2:]
        lsb = round(len(bin_val) * lsbf)
        lsb = 1 if lsb == 0 else lsb
        i = H(sk, idx) % lsb
        return bin_val[-i-1] 
    
    elif type(val) == np.float64:
        bin_val = str(val).split('.')
        frac_bin = list(bin(int(bin_val[1])))[2:]
        lsb = round(len(frac_bin) * lsbf)
        lsb = 1 if lsb == 0 else lsb
        i = H(sk, idx) % lsb
        return frac_bin[-i-1]

# embedding process
for sk in k_set:
    hash_mod_gamma = df.index.map(lambda x: H(sk,x) % gamma)
    selected_indices = df.index[hash_mod_gamma == 0] 
    column_indices = selected_indices.map(lambda x: H(sk,x) % len(df.columns))
    for idx, col_idx in zip(selected_indices, column_indices):
        col_name = df.columns[col_idx]
        val = df.at[idx, col_name]
        wi = wm[H(sk, idx) % len(wm)]
        df.at[idx, col_name] = em(val, wi, sk, idx, col_name)
        
# select part of key to detect
n = 5
k_provide = np.random.choice(k_set, n, replace=False)
print(f'k_provide: {k_provide}')

# detection process
zero = [0] * len(wm)
one = [0] * len(wm)
for sk in k_provide:
    hash_mod_gamma = df.index.map(lambda x: H(sk,x) % gamma)
    selected_indices = df.index[hash_mod_gamma == 0] 
    column_indices = selected_indices.map(lambda x: H(sk,x) % len(df.columns))
    detected_wm = ['-'] * len(wm)
    for idx, col_idx in zip(selected_indices, column_indices):
        col_name = df.columns[col_idx]
        val = df.at[idx, col_name]
        dw = de(val, sk, idx)  
        # dew.append(dw)
        if dw == '0':
            zero[H(sk, idx) % len(wm)] += 1
        else:
            one[H(sk, idx) % len(wm)] += 1

for i in range(len(wm)):
    if (zero[i]>one[i]):
        detected_wm[i] = '0'
    elif zero[i]<one[i]:
        detected_wm[i] = '1'

print(df.describe())
print(wm)
print(''.join(detected_wm))