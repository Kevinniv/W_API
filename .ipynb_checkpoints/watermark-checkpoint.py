from flask import Flask, request, jsonify, send_file
import pandas as pd
import hashlib
import numpy as np
import os
import tempfile
from PIL import Image

app = Flask(__name__)

# Secret parameter configurations
lsbf = 0.3
gamma = 1009
np.random.seed(0)
k_set = np.random.randint(0, 0xffff, size=10)
print(f'secret keys: {k_set}')

def H(k:int, v) -> int:
    """Hash function."""
    h = hashlib.sha256()
    h.update((str(k) + str(v)).encode())
    return int(h.hexdigest(), 16)

def em(val, w, sk, idx, col):
    """Watermark embedding function."""
    if isinstance(val, np.int64):
        bin_val = list(bin(abs(val)))[2:]
        lsb = max(1, round(len(bin_val) * lsbf))
        i = H(sk, idx) % lsb  
        bin_val[-i-1] = w
        if val < 0:
            new_val = np.int64(-int(''.join(bin_val), 2))
        else: 
            new_val = np.int64(int(''.join(bin_val), 2))
        print(f"Embedding in int64: Original {val}, New {new_val}, Watermark bit {w}")
        return new_val
    elif isinstance(val, np.float64):
        frac_bin = list(bin(int(str(val).split('.')[1])))[2:]
        lsb = max(1, round(len(frac_bin) * lsbf))
        i = H(sk, idx) % lsb 
        frac_bin[-i-1] = w
        val_list = str(val).split('.')[0] + '.' + str(int(''.join(frac_bin), 2))
        new_val = np.float64(val_list)
        print(f"Embedding in float64: Original {val}, New {new_val}, Watermark bit {w}")
        return new_val

def de(val, sk, idx):
    """Watermark detection function."""
    if isinstance(val, np.int64):
        bin_val = list(bin(abs(val)))[2:]
        lsb = max(1, round(len(bin_val) * lsbf))
        i = H(sk, idx) % lsb
        detected_bit = bin_val[-i-1]
        print(f"Detected from int64: Value {val}, Detected bit {detected_bit}")
        return detected_bit
    
    elif isinstance(val, np.float64):
        frac_bin = list(bin(int(str(val).split('.')[1])))[2:]
        lsb = max(1, round(len(frac_bin) * lsbf))
        i = H(sk, idx) % lsb
        detected_bit = frac_bin[-i-1]
        print(f"Detected from float64: Value {val}, Detected bit {detected_bit}")
        return detected_bit

@app.route('/embed', methods=['POST'])
def embed_watermark():
    if 'file' not in request.files or 'watermark' not in request.files:
        return jsonify({"error": "Missing file or watermark data"}), 400
    
    file = request.files['file']
    watermark_file = request.files['watermark']
    
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    watermark_path = os.path.join(tempfile.gettempdir(), watermark_file.filename)
    file.save(temp_path)
    watermark_file.save(watermark_path)

    # Load dataset
    df = pd.read_csv(temp_path)
    print(f'Original DataFrame head:\n{df.head()}')

    # Load watermark based on file type
    if watermark_file.mimetype == 'text/plain':
        with open(watermark_path, 'r') as f:
            watermark_bits = f.read().strip()
    else:
        watermark_image = Image.open(watermark_path).convert('1')
        watermark_array = np.array(watermark_image)
        watermark_bits = ''.join(map(str, watermark_array.flatten()))

    print(f'Watermark bits: {watermark_bits}')

    # Embedding process
    for sk in k_set:
        hash_mod_gamma = df.index.map(lambda x: H(sk, x) % gamma)
        selected_indices = df.index[hash_mod_gamma == 0] 
        column_indices = selected_indices.map(lambda x: H(sk, x) % len(df.columns))
        for idx, col_idx in zip(selected_indices, column_indices):
            col_name = df.columns[col_idx]
            val = df.at[idx, col_name]
            wi = watermark_bits[H(sk, idx) % len(watermark_bits)]
            df.at[idx, col_name] = em(val, wi, sk, idx, col_name)

    print(f'Watermarked DataFrame head:\n{df.head()}')

    output_filename = "watermarked_" + file.filename
    output_path = os.path.join(tempfile.gettempdir(), output_filename)
    df.to_csv(output_path, index=False)

    keys_filename = "watermark_keys.txt"
    keys_path = os.path.join(tempfile.gettempdir(), keys_filename)
    with open(keys_path, 'w') as f:
        f.write(','.join(map(str, k_set)))

    return jsonify({
        "message": "Watermark embedded successfully", 
        "download_link": f"/download/{output_filename}",
        "keys_link": f"/download/{keys_filename}"
    }), 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True, download_name=filename)
    else:
        return jsonify({"error": "File not found"}), 404

@app.route('/detect', methods=['POST'])
def detect_watermark():
    if 'file' not in request.files or 'keys' not in request.form:
        return jsonify({"error": "Missing file or key data"}), 400
    
    file = request.files['file']
    keys_provided = request.form['keys'].split(',')
    keys_provided = [int(k) for k in keys_provided]
    wm_length = int(request.form.get('wm_length', 64))
    
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    df = pd.read_csv(temp_path)

    zero = [0] * wm_length
    one = [0] * wm_length
    detected_wm = ['0'] * wm_length  # 初始化detected_wm列表

    for sk in keys_provided:
        hash_mod_gamma = df.index.map(lambda x: H(sk, x) % gamma)
        selected_indices = df.index[hash_mod_gamma == 0] 
        column_indices = selected_indices.map(lambda x: H(sk, x) % len(df.columns))
        
        for idx, col_idx in zip(selected_indices, column_indices):
            col_name = df.columns[col_idx]
            val = df.at[idx, col_name]
            dw = de(val, sk, idx)
            
            if dw == '0':
                zero[H(sk, idx) % wm_length] += 1
            else:
                one[H(sk, idx) % wm_length] += 1

    for i in range(wm_length):
        if zero[i] > one[i]:
            detected_wm[i] = '0'
        elif zero[i] < one[i]:
            detected_wm[i] = '1'
    
    print(f'Detected watermark: {"".join(detected_wm)}')

    return jsonify({"detected_watermark": ''.join(detected_wm)}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)





# from flask import Flask, request, jsonify, send_file
# import pandas as pd
# import hashlib
# import numpy as np
# import os
# import tempfile
# from PIL import Image

# app = Flask(__name__)

# # Secret parameter configurations
# lsbf = 0.3
# gamma = 1009
# np.random.seed(0)
# k_set = np.random.randint(0, 0xffff, size=10)
# print(f'secret keys: {k_set}')

# def H(k:int, v) -> int:
#     """Hash function."""
#     h = hashlib.sha256()
#     h.update((str(k) + str(v)).encode())
#     return int(h.hexdigest(), 16)

# def em(val, w, sk, idx, col):
#     """Watermark embedding function."""
#     if isinstance(val, np.int64):
#         bin_val = list(bin(abs(val)))[2:]
#         lsb = round(len(bin_val) * lsbf)
#         lsb = 1 if lsb == 0 else lsb
#         i = H(sk, idx) % lsb  
#         bin_val[-i-1] = w
#         if val < 0:
#             return np.int64(-int(''.join(bin_val), 2))
#         else: 
#             return np.int64(int(''.join(bin_val), 2))
#     elif isinstance(val, np.float64):
#         bin_val = str(val).split('.')
#         frac_bin = list(bin(int(bin_val[1])))[2:]
#         lsb = round(len(frac_bin) * lsbf)
#         lsb = 1 if lsb == 0 else lsb
#         i = H(sk, idx) % lsb 
#         frac_bin[-i-1] = w
#         val_list = bin_val[0] + '.' + str(int(''.join(frac_bin), 2))
#         return np.float64(float(val_list))

# def de(val, sk, idx):
#     """Watermark detection function."""
#     if isinstance(val, np.int64):
#         bin_val = list(bin(val))[2:]
#         lsb = round(len(bin_val) * lsbf)
#         lsb = 1 if lsb == 0 else lsb
#         i = H(sk, idx) % lsb
#         return bin_val[-i-1] 
    
#     elif isinstance(val, np.float64):
#         bin_val = str(val).split('.')
#         frac_bin = list(bin(int(bin_val[1])))[2:]
#         lsb = round(len(frac_bin) * lsbf)
#         lsb = 1 if lsb == 0 else lsb
#         i = H(sk, idx) % lsb
#         return frac_bin[-i-1]

# @app.route('/embed', methods=['POST'])
# def embed_watermark():
#     print("Request files:", request.files)
#     print("Request form:", request.form)

#     if 'file' not in request.files or 'watermark' not in request.files:
#         print("Missing data")
#         return jsonify({"error": "Missing file or watermark data"}), 400
    
#     # 获取上传的文件
#     file = request.files['file']
#     watermark_file = request.files['watermark']
    
#     # 保存文件到临时目录
#     temp_path = os.path.join(tempfile.gettempdir(), file.filename)
#     watermark_path = os.path.join(tempfile.gettempdir(), watermark_file.filename)
#     file.save(temp_path)
#     watermark_file.save(watermark_path)

#     # 加载数据集
#     df = pd.read_csv(temp_path)

#     # 根据上传的水印文件类型加载水印
#     if watermark_file.mimetype == 'text/plain':
#         # 如果是文本文件，读取二进制水印
#         with open(watermark_path, 'r') as f:
#             watermark_bits = f.read().strip()
#     else:
#         # 如果是图像文件，读取并转换为二进制图像
#         watermark_image = Image.open(watermark_path).convert('1')  # Convert to binary image (black & white)
#         watermark_array = np.array(watermark_image)
#         watermark_bits = ''.join(map(str, watermark_array.flatten()))

#     # 水印嵌入过程
#     for sk in k_set:
#         hash_mod_gamma = df.index.map(lambda x: H(sk, x) % gamma)
#         selected_indices = df.index[hash_mod_gamma == 0] 
#         column_indices = selected_indices.map(lambda x: H(sk, x) % len(df.columns))
#         for idx, col_idx in zip(selected_indices, column_indices):
#             col_name = df.columns[col_idx]
#             val = df.at[idx, col_name]
#             wi = watermark_bits[H(sk, idx) % len(watermark_bits)]  # 从水印中获取相应的比特
#             df.at[idx, col_name] = em(val, wi, sk, idx, col_name)
    
#     # 保存带有水印的数据集到当前工作目录
#     output_filename = "watermarked_" + file.filename
#     output_path = os.path.join(tempfile.gettempdir(), output_filename)
#     df.to_csv(output_path, index=False)

#     # 将密钥保存到一个文件
#     keys_filename = "watermark_keys.txt"
#     keys_path = os.path.join(tempfile.gettempdir(), keys_filename)
#     with open(keys_path, 'w') as f:
#         f.write(','.join(map(str, k_set)))

#     # 返回保存成功的消息和下载文件的链接
#     return jsonify({
#         "message": "Watermark embedded successfully", 
#         "download_link": f"/download/{output_filename}",
#         "keys_link": f"/download/{keys_filename}"
#     }), 200

# @app.route('/download/<filename>', methods=['GET'])
# def download_file(filename):
#     """提供嵌入水印的数据文件下载。"""
#     file_path = os.path.join(tempfile.gettempdir(), filename)
#     if os.path.exists(file_path):
#         return send_file(file_path, as_attachment=True, download_name=filename)
#     else:
#         return jsonify({"error": "File not found"}), 404

# @app.route('/detect', methods=['POST'])
# def detect_watermark():
#     print("Request files:", request.files)
#     print("Request form:", request.form)

#     if 'file' not in request.files or 'keys' not in request.form:
#         return jsonify({"error": "Missing file or key data"}), 400
    
#     # 获取文件和密钥
#     file = request.files['file']
#     keys_provided = request.form['keys'].split(',')
#     keys_provided = [int(k) for k in keys_provided]  # 将密钥转换为整数列表
#     wm_length = int(request.form.get('wm_length', 64))  # 默认水印长度为64
    
#     temp_path = os.path.join(tempfile.gettempdir(), file.filename)
#     file.save(temp_path)

#     # 加载数据集
#     df = pd.read_csv(temp_path)

#     # 初始化计数器
#     zero = [0] * wm_length
#     one = [0] * wm_length

#     # 水印提取过程
#     for sk in keys_provided:
#         hash_mod_gamma = df.index.map(lambda x: H(sk, x) % gamma)
#         selected_indices = df.index[hash_mod_gamma == 0] 
#         column_indices = selected_indices.map(lambda x: H(sk, x) % len(df.columns))
        
#         for idx, col_idx in zip(selected_indices, column_indices):
#             col_name = df.columns[col_idx]
#             val = df.at[idx, col_name]
#             dw = de(val, sk, idx)
            
#             # 根据提取结果更新计数器
#             if dw == '0':
#                 zero[H(sk, idx) % wm_length] += 1
#             else:
#                 one[H(sk, idx) % wm_length] += 1

#     # 生成检测到的水印
#     detected_wm = ['0' if zero[i] > one[i] else '1' for i in range(wm_length)]
    
#     return jsonify({"detected_watermark": ''.join(detected_wm)}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


# from flask import Flask, request, jsonify, send_file
# import pandas as pd
# import hashlib
# import numpy as np
# import os
# import tempfile
# from PIL import Image

# app = Flask(__name__)

# # Secret parameter configurations
# lsbf = 0.3
# gamma = 1009
# np.random.seed(0)
# k_set = np.random.randint(0, 0xffff, size=10)
# print(f'secret keys: {k_set}')

# def H(k:int, v) -> int:
#     """Hash function."""
#     h = hashlib.sha256()
#     h.update((str(k) + str(v)).encode())
#     return int(h.hexdigest(), 16)

# def em(val, w, sk, idx, col):
#     """Watermark embedding function."""
#     if isinstance(val, np.int64):
#         bin_val = list(bin(abs(val)))[2:]
#         lsb = round(len(bin_val) * lsbf)
#         lsb = 1 if lsb == 0 else lsb
#         i = H(sk, idx) % lsb  
#         bin_val[-i-1] = w
#         if val < 0:
#             return np.int64(-int(''.join(bin_val), 2))
#         else: 
#             return np.int64(int(''.join(bin_val), 2))
#     elif isinstance(val, np.float64):
#         bin_val = str(val).split('.')
#         frac_bin = list(bin(int(bin_val[1])))[2:]
#         lsb = round(len(frac_bin) * lsbf)
#         lsb = 1 if lsb == 0 else lsb
#         i = H(sk, idx) % lsb 
#         frac_bin[-i-1] = w
#         val_list = bin_val[0] + '.' + str(int(''.join(frac_bin), 2))
#         return np.float64(float(val_list))

# def de(val, sk, idx):
#     """Watermark detection function."""
#     if isinstance(val, np.int64):
#         bin_val = list(bin(val))[2:]
#         lsb = round(len(bin_val) * lsbf)
#         lsb = 1 if lsb == 0 else lsb
#         i = H(sk, idx) % lsb
#         return bin_val[-i-1] 
    
#     elif isinstance(val, np.float64):
#         bin_val = str(val).split('.')
#         frac_bin = list(bin(int(bin_val[1])))[2:]
#         lsb = round(len(frac_bin) * lsbf)
#         lsb = 1 if lsb == 0 else lsb
#         i = H(sk, idx) % lsb
#         return frac_bin[-i-1]

# @app.route('/embed', methods=['POST'])
# def embed_watermark():
#     print("Request files:", request.files)
#     print("Request form:", request.form)

#     if 'file' not in request.files or 'watermark' not in request.files:
#         print("Missing data")
#         return jsonify({"error": "Missing file or watermark data"}), 400
    
#     # 获取上传的文件
#     file = request.files['file']
#     watermark_file = request.files['watermark']
    
#     # 保存文件到临时目录
#     temp_path = os.path.join(tempfile.gettempdir(), file.filename)
#     watermark_path = os.path.join(tempfile.gettempdir(), watermark_file.filename)
#     file.save(temp_path)
#     watermark_file.save(watermark_path)

#     # 加载数据集
#     df = pd.read_csv(temp_path)

#     # 根据上传的水印文件类型加载水印
#     if watermark_file.mimetype == 'text/plain':
#         # 如果是文本文件，读取二进制水印
#         with open(watermark_path, 'r') as f:
#             watermark_bits = f.read().strip()
#     else:
#         # 如果是图像文件，读取并转换为二进制图像
#         watermark_image = Image.open(watermark_path).convert('1')  # Convert to binary image (black & white)
#         watermark_array = np.array(watermark_image)
#         watermark_bits = ''.join(map(str, watermark_array.flatten()))

#     # 水印嵌入过程
#     for sk in k_set:
#         hash_mod_gamma = df.index.map(lambda x: H(sk,x) % gamma)
#         selected_indices = df.index[hash_mod_gamma == 0] 
#         column_indices = selected_indices.map(lambda x: H(sk,x) % len(df.columns))
#         for idx, col_idx in zip(selected_indices, column_indices):
#             col_name = df.columns[col_idx]
#             val = df.at[idx, col_name]
#             wi = watermark_bits[H(sk, idx) % len(watermark_bits)]  # 从水印中获取相应的比特
#             df.at[idx, col_name] = em(val, wi, sk, idx, col_name)
    
#     # 保存带有水印的数据集到当前工作目录
#     output_filename = "watermarked_" + file.filename
#     output_path = os.path.join(tempfile.gettempdir(), output_filename)
#     df.to_csv(output_path, index=False)

#     # 返回保存成功的消息和下载文件的链接
#     return jsonify({"message": "Watermark embedded successfully", "download_link": f"/download/{output_filename}"}), 200

# @app.route('/download/<filename>', methods=['GET'])
# def download_file(filename):
#     """提供嵌入水印的数据文件下载。"""
#     file_path = os.path.join(tempfile.gettempdir(), filename)
#     if os.path.exists(file_path):
#         return send_file(file_path, as_attachment=True, download_name=filename)
#     else:
#         return jsonify({"error": "File not found"}), 404

# @app.route('/detect', methods=['POST'])
# def detect_watermark():
#     print("Request files:", request.files)
#     print("Request form:", request.form)

#     if 'file' not in request.files or 'keys' not in request.form:
#         return jsonify({"error": "Missing file or key data"}), 400
    
#     # 获取文件和密钥
#     file = request.files['file']
#     keys_provided = request.form['keys'].split(',')
#     keys_provided = [int(k) for k in keys_provided]  # 将密钥转换为整数列表
#     wm_length = int(request.form.get('wm_length', 64))  # 默认水印长度为64
    
#     temp_path = os.path.join(tempfile.gettempdir(), file.filename)
#     file.save(temp_path)

#     # 加载数据集
#     df = pd.read_csv(temp_path)

#     # 初始化计数器
#     zero = [0] * wm_length
#     one = [0] * wm_length

#     # 水印提取过程
#     for sk in keys_provided:
#         hash_mod_gamma = df.index.map(lambda x: H(sk, x) % gamma)
#         selected_indices = df.index[hash_mod_gamma == 0] 
#         column_indices = selected_indices.map(lambda x: H(sk, x) % len(df.columns))
        
#         for idx, col_idx in zip(selected_indices, column_indices):
#             col_name = df.columns[col_idx]
#             val = df.at[idx, col_name]
#             dw = de(val, sk, idx)
            
#             # 根据提取结果更新计数器
#             if dw == '0':
#                 zero[H(sk, idx) % wm_length] += 1
#             else:
#                 one[H(sk, idx) % wm_length] += 1

#     # 生成检测到的水印
#     detected_wm = ['0' if zero[i] > one[i] else '1' for i in range(wm_length)]
    
#     return jsonify({"detected_watermark": ''.join(detected_wm)}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
