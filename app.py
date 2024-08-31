from flask import Flask, request, send_file, jsonify
import pandas as pd
import hashlib
import numpy as np
import os
import logging
import tempfile

app = Flask(__name__)

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 生成密钥集和哈希函数
def generate_keys(seed=0, size=10):
    np.random.seed(seed)
    keys = np.random.randint(0, 0xffff, size=size)
    logging.info(f"Generated keys: {keys}")
    return keys

def H(k:int, v) -> int:
    h = hashlib.sha256()
    h.update((str(k) + str(v)).encode())
    return int(h.hexdigest(), 16)

# 嵌入水印的函数
def em(val, w, sk, idx, lsbf):
    try:
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
    except Exception as e:
        logging.error(f"Error embedding watermark at index {idx}, column {col}: {e}")
        raise

# 提取水印的函数
def de(val, sk, idx, lsbf):
    try:
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
    except Exception as e:
        logging.error(f"Error extracting watermark at index {idx}: {e}")
        raise

@app.route('/embed', methods=['POST'])
def embed_watermark():
    try:
        file = request.files['file']
        watermark_file = request.files['watermark']
        
        logging.info(f"Received file: {file.filename}")
        logging.info(f"Received watermark file: {watermark_file.filename}")

        # 设置参数
        lsbf = 0.3
        gamma = 1009
        k_set = generate_keys()

        # 读取文件
        df = pd.read_csv(file)
        wm = watermark_file.read().decode('utf-8').strip()

        logging.info(f"Dataframe loaded with shape: {df.shape}")
        logging.info(f"Watermark loaded with length: {len(wm)}")

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

        logging.info("Watermark embedding completed successfully.")

        # 将水印长度编码到CSV文件的隐藏列中
        df['__watermark_length__'] = len(wm)

        # 保存嵌入水印后的文件到临时文件
        watermarked_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(watermarked_file.name, index=False)

        # 保存密钥文件
        keys_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        with open(keys_file.name, 'w') as kf:
            kf.write(','.join(map(str, k_set)))

        logging.info(f"Watermarked file saved to: {watermarked_file.name}")
        logging.info(f"Keys file saved to: {keys_file.name}")

        return jsonify({
            "watermarked_file_url": watermarked_file.name,
            "keys_file_url": keys_file.name
        })

    except Exception as e:
        logging.error(f"Error during watermark embedding: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/extract', methods=['POST'])
def extract_watermark():
    try:
        file = request.files['file']
        keys_file = request.files['keys']
        
        logging.info(f"Received file: {file.filename}")
        logging.info(f"Received keys file: {keys_file.filename}")

        # 设置参数
        lsbf = 0.3
        gamma = 1009

        # 读取文件
        df = pd.read_csv(file)

        # 从隐藏列中读取水印长度
        wm_length = int(df['__watermark_length__'].iloc[0])

        logging.info(f"Watermark length retrieved: {wm_length}")

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

        logging.info(f"Detected watermark: {detected_wm_str}")

        # 保存提取的水印信息到文件
        detected_wm_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        with open(detected_wm_file.name, 'w') as dwf:
            dwf.write(detected_wm_str)

        return send_file(detected_wm_file.name, as_attachment=True)
    
    except Exception as e:
        logging.error(f"Error during watermark extraction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
    pass
