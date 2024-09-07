from flask import Flask, request, send_file, jsonify, url_for
import pandas as pd
import hashlib
import numpy as np
import os
import logging
import tempfile
from PIL import Image

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 允许最大上传100MB的文件

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 生成密钥集和哈希函数
def generate_keys(seed=0, size=10):
    np.random.seed(seed)
    keys = np.random.randint(0, 0xffff, size=size)
    logging.info(f"Generated keys: {keys}")
    return keys

def H(k: int, v) -> int:
    h = hashlib.sha256()
    h.update((str(k) + str(v)).encode())
    return int(h.hexdigest(), 16)

# 嵌入比特水印的函数
def embed_bit_watermark(val, w, sk, idx, lsbf):
    try:
        if type(val) == np.int64:
            bin_val = list(bin(abs(val)))[2:]
            lsb = round(len(bin_val) * lsbf)
            lsb = 1 if lsb == 0 else lsb
            i = H(sk, idx) % lsb
            bin_val[-i - 1] = w
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
            frac_bin[-i - 1] = w
            val_list = bin_val[0] + '.' + str(int(''.join(frac_bin), 2))
            return np.float64(float(val_list))
    except Exception as e:
        logging.error(f"Error embedding watermark at index {idx}: {e}")
        raise

def extract_bit_watermark(val, sk, idx, lsbf):
    try:
        if type(val) == np.int64:
            bin_val = list(bin(val))[2:]
            lsb = round(len(bin_val) * lsbf)
            lsb = 1 if lsb == 0 else lsb
            i = H(sk, idx) % lsb
            return bin_val[-i - 1]

        elif type(val) == np.float64:
            bin_val = str(val).split('.')
            frac_bin = list(bin(int(bin_val[1])))[2:]
            lsb = round(len(frac_bin) * lsbf)
            lsb = 1 if lsb == 0 else lsb
            i = H(sk, idx) % lsb
            return frac_bin[-i - 1]
    except Exception as e:
        logging.error(f"Error extracting watermark at index {idx}: {e}")
        raise

# 优化后的嵌入图像水印的函数
def embed_image_watermark(image, watermark):
    try:
        watermark = watermark.convert("1")  # 转换为二值图像
        wm_array = np.asarray(watermark, dtype=np.uint8)  # 转换为numpy数组，0-1
        img_array = np.asarray(image, dtype=np.uint8)  # 转换为numpy数组

        if img_array.shape != wm_array.shape:
            raise ValueError("Watermark and image must be of the same size")

        # 在图像中嵌入水印
        watermarked_array = np.bitwise_xor(img_array, wm_array)
        return Image.fromarray(watermarked_array)
    except Exception as e:
        logging.error(f"Error embedding image watermark: {e}")
        raise

# 优化后的提取图像水印的函数
def extract_image_watermark(image, watermark):
    try:
        watermark = watermark.convert("1")  # 转换为二值图像
        wm_array = np.asarray(watermark, dtype=np.uint8)  # 转换为numpy数组
        img_array = np.asarray(image, dtype=np.uint8)  # 转换为numpy数组

        # 提取水印
        extracted_array = np.bitwise_xor(img_array, wm_array)
        return Image.fromarray(extracted_array)
    except Exception as e:
        logging.error(f"Error extracting image watermark: {e}")
        raise

# 随机选择部分密钥进行返回
def select_partial_keys(keys, portion=0.5):
    num_keys = int(len(keys) * portion)  # 返回部分密钥的比例，默认返回一半
    selected_keys = np.random.choice(keys, size=num_keys, replace=False)
    logging.info(f"Selected partial keys: {selected_keys}")
    return selected_keys

@app.route('/embed', methods=['POST'])
def embed_watermark():
    try:
        file = request.files['file']
        watermark_file = request.files['watermark']

        logging.info(f"Received file: {file.filename}")
        logging.info(f"Received watermark file: {watermark_file.filename}")

        # 判断文件类型
        if file.filename.endswith('.csv'):
            # 处理CSV文件
            lsbf = 0.3
            gamma = 1009
            k_set = generate_keys()

            df = pd.read_csv(file)

            # 处理水印图像
            watermark = Image.open(watermark_file.stream).convert("1")  # 转换为二值图像
            wm_array = np.asarray(watermark, dtype=np.uint8).flatten()  # 转换为1D numpy数组
            logging.info(f"Watermark loaded with shape: {wm_array.shape}")

            if wm_array.size == 0:
                raise ValueError("Watermark image is empty or invalid")

            watermark_bits = wm_array  # 直接使用1D数组
            watermark_length = len(watermark_bits)

            for sk in k_set:
                hash_mod_gamma = df.index.map(lambda x: H(sk, x) % gamma)
                selected_indices = df.index[hash_mod_gamma == 0]
                column_indices = selected_indices.map(lambda x: H(sk, x) % len(df.columns))
                for idx, col_idx in zip(selected_indices, column_indices):
                    col_name = df.columns[col_idx]
                    val = df.at[idx, col_name]
                    bit_idx = H(sk, idx) % watermark_length
                    bit_value = str(watermark_bits[bit_idx])  # 直接使用比特值
                    df.at[idx, col_name] = embed_bit_watermark(val, bit_value, sk, idx, lsbf)

            df['__watermark_length__'] = watermark_length

            watermarked_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            df.to_csv(watermarked_file.name, index=False)

            keys_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')

            # 仅返回部分密钥，默认返回50%
            partial_keys = select_partial_keys(k_set, portion=0.5)
            with open(keys_file.name, 'w') as kf:
                kf.write(','.join(map(str, partial_keys)))

            logging.info(f"Watermarked file saved to: {watermarked_file.name}")
            logging.info(f"Partial keys saved to: {keys_file.name}")

            return jsonify({
                "watermarked_file_url": url_for('download_file', filename=os.path.basename(watermarked_file.name), _external=True),
                "keys_file_url": url_for('download_file', filename=os.path.basename(keys_file.name), _external=True)
            })

        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 处理图像文件
            image = Image.open(file.stream)
            watermark = Image.open(watermark_file.stream)

            watermarked_image = embed_image_watermark(image, watermark)

            watermarked_image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            watermarked_image.save(watermarked_image_file.name)

            logging.info(f"Watermarked image saved to: {watermarked_image_file.name}")

            return jsonify({
                "watermarked_image_url": url_for('download_file', filename=os.path.basename(watermarked_image_file.name), _external=True)
            })

        else:
            return jsonify({"status": "error", "message": "Unsupported file type"}), 400

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

        # Set parameters
        lsbf = 0.3
        gamma = 1009

        # Read dataset
        df = pd.read_csv(file)

        # Read watermark length from hidden column
        wm_length = int(df['__watermark_length__'].iloc[0])

        logging.info(f"Watermark length retrieved: {wm_length}")

        # Read keys from keys.txt
        k_provide = list(map(int, keys_file.read().decode('utf-8').strip().split(',')))

        # Remove watermark length hidden column
        df.drop(columns=['__watermark_length__'], inplace=True)

        # Initialize counters for detected bits
        zero = [0] * wm_length
        one = [0] * wm_length

        # Watermark extraction process
        for sk in k_provide:
            hash_mod_gamma = df.index.map(lambda x: H(sk, x) % gamma)
            selected_indices = df.index[hash_mod_gamma == 0]
            if len(df.columns) == 0:
                logging.error("No columns in DataFrame for modulo operation")
                continue

            column_indices = selected_indices.map(lambda x: H(sk, x) % len(df.columns) if len(df.columns) > 0 else 0)
            for idx, col_idx in zip(selected_indices, column_indices):
                col_name = df.columns[col_idx]
                val = df.at[idx, col_name]
                dw = extract_bit_watermark(val, sk, idx, lsbf)
                if dw == '0':
                    zero[H(sk, idx) % wm_length] += 1
                else:
                    one[H(sk, idx) % wm_length] += 1

        # Determine the watermark bits based on majority voting
        detected_wm = ['0' if zero[i] > one[i] else '1' for i in range(wm_length)]
        
        logging.info(f"Detected watermark bits: {detected_wm[:100]}... (truncated for logging)")

        # Reconstruct the image dimensions
        original_image_size = int(np.sqrt(wm_length))  # Assuming the watermark image is square
        if original_image_size ** 2 != wm_length:
            raise ValueError("The number of extracted bits does not match a perfect square, cannot reshape to an image.")

        # Convert detected bits back into a binary image
        watermark_array = np.array([int(bit) for bit in detected_wm]).reshape((original_image_size, original_image_size)) * 255

        # Save the binary image
        reconstructed_image = Image.fromarray(watermark_array.astype(np.uint8))
        reconstructed_image_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        reconstructed_image.save(reconstructed_image_file.name)

        logging.info(f"Reconstructed image saved to: {reconstructed_image_file.name}")

        return send_file(reconstructed_image_file.name, as_attachment=True)
    
    except Exception as e:
        logging.error(f"Error during watermark extraction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        file_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({"status": "error", "message": "File not found"}), 404
    except Exception as e:
        logging.error(f"Error during file download: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
