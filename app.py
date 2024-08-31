from flask import Flask, request, send_file, jsonify
import pandas as pd
import hashlib
import numpy as np
import os

app = Flask(__name__)

# 生成密钥集和哈希函数
def generate_keys(seed=0, size=10):
    np.random.seed(seed)
    return np.random.randint(0, 0xffff, size=size)

def H(k:int, v) -> int:
    h = hashlib.sha256()
    h.update((str(k) + str(v)).encode())
    return int(h.hexdigest(), 16)

# 嵌入水印的函数
def em(val, w, sk, idx, lsbf):
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

# 提取水印的函数
def de(val, sk, idx, lsbf):
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

@app.route('/embed', methods=['POST'])
def embed_watermark():
    file = request.files['file']
    watermark_file = request.files['watermark']
    
    # 设置参数
    lsbf = 0.3
    gamma = 1009
    k_set = generate_keys()

    # 读取文件
    df = pd.read_csv(file)
    wm = watermark_file.read().decode('utf-8').strip()

    # 嵌入水印过程
    for sk in k_set:
        hash_mod_gamma = df.index.map(lambda x: H(sk, x) % gamma)
        selected_indices = df.index[hash_mod_gamma == 0] 
        column_indices = selected_indices.map(lambda x: H(sk, x) % len(df.columns))
        for idx, col_idx in zip(selected_indices, column_indices):
            col_name = df.columns[col_idx]
            val = df.at[idx, col_name]
            wi = wm[H(sk, idx) % len(wm)]
            df.at[idx, col_name] = em(val, wi, sk, idx, lsbf)
    
    # 将水印长度编码到CSV文件的隐藏列中
    df['__watermark_length__'] = len(wm)

    # 保存文件
    watermarked_file = 'watermarked.csv'
    df.to_csv(watermarked_file, index=False)

    # 保存密钥文件
    keys_file = 'keys.txt'
    with open(keys_file, 'w') as kf:
        kf.write(','.join(map(str, k_set)))
    
    return jsonify({
        "watermarked_file_url": watermarked_file,
        "keys_file_url": keys_file
    })


@app.route('/extract', methods=['POST'])
def extract_watermark():
    file = request.files['file']
    keys_file = request.files['keys']
    
    # 设置参数
    lsbf = 0.3
    gamma = 1009

    # 读取文件
    df = pd.read_csv(file)

    # 从隐藏列中读取水印长度
    wm_length = int(df['__watermark_length__'].iloc[0])

    # 从keys.txt读取密钥
    k_provide = list(map(int, keys_file.read().decode('utf-8').strip().split(',')))

    # 删除水印长度隐藏列（不影响水印提取）
    df.drop(columns=['__watermark_length__'], inplace=True)

    # 初始化检测水印的零和一计数列表
    zero = [0] * wm_length
    one = [0] * wm_length

    # 水印提取过程
    for sk in k_provide:
        hash_mod_gamma = df.index.map(lambda x: H(sk, x) % gamma)
        selected_indices = df.index[hash_mod_gamma == 0] 
        column_indices = selected_indices.map(lambda x: H(sk, x) % len(df.columns))
        for idx, col_idx in zip(selected_indices, column_indices):
            col_name = df.columns[col_idx]
            val = df.at[idx, col_name]
            dw = de(val, sk, idx, lsbf)
            if dw == '0':
                zero[H(sk, idx) % wm_length] += 1
            else:
                one[H(sk, idx) % wm_length] += 1

    detected_wm = ['0' if zero[i] > one[i] else '1' for i in range(wm_length)]
    detected_wm_str = ''.join(detected_wm)

    # 打印水印信息到终端
    print(detected_wm_str)

    # 保存提取的水印信息到文件
    detected_wm_file = 'detected_watermark.txt'
    with open(detected_wm_file, 'w') as dwf:
        dwf.write(detected_wm_str)

    return send_file(detected_wm_file, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))

