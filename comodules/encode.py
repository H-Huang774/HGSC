import numpy as np
import zlib
import os
import time
from scipy.spatial import KDTree
from comodules.gpcc_reflectance.encode_with_gpcc_reflectance import compress_use_reflectance
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import BinaryPPM


def compress_and_save_residuals(residuals, file_path_prefix):
    for key, value in residuals.items():
        residuals_path_prefix = 'residuals_precompression'
        residuals_path = f"{residuals_path_prefix}_{key}.txt"
        value_reshaped = value.reshape(value.shape[0], -1)
        np.savetxt(residuals_path, value_reshaped, fmt='%f')

        value_flattened = value.flatten().astype(np.float32)  # 展平数据
        print(f"{key} original shape: {value.shape}, flattened shape: {value_flattened.shape}")
        value_bytes = value_flattened.tobytes()
        compressed_value = zlib.compress(value_bytes)
        file_path = f"{file_path_prefix}_{key}.bin"
        with open(file_path, 'wb') as f:
            f.write(compressed_value)
        print(f"{key} compressed and saved to {file_path}")

def load_and_decompress_residuals(file_path_prefix, shapes):
    decompressed_residuals = {}
    for key in ['features_dc', 'features_rest', 'scaling', 'rotation', 'opacity']:
        file_path = f"{file_path_prefix}_{key}.bin"
        with open(file_path, 'rb') as f:
            compressed_value = f.read()
            value_bytes = zlib.decompress(compressed_value)
            decompressed_value = np.frombuffer(value_bytes, dtype=np.float32)
            print(f"{key} decompressed shape: {decompressed_value.shape}, expected shape: {shapes[key]}")
            decompressed_value = decompressed_value.reshape(shapes[key])
            decompressed_residuals[key] = decompressed_value
        print(f"{key} loaded and decompressed from {file_path}")

        # 保存为txt文件
        txt_file_path = f"{file_path_prefix}_{key}.txt"
        decompressed_value = decompressed_value.reshape(decompressed_value.shape[0],-1)
        np.savetxt(txt_file_path, decompressed_value)
        print(f"{key} saved to {txt_file_path}")
        return decompressed_residuals
    
def gpcc_encoder(ply_path, tmc_path):
    # tmc_path = '/home/old/huanghe/GS_repository/HGSC/comodules/tmc/linux/tmc3'

    gs_xyz_encoded_bin = os.path.splitext(ply_path)[0] + '.bin'

    start_time = time.time()
    tmc_encode_cmd = "{} --mode=0 --trisoupNodeSizeLog2=0 --neighbourAvailBoundaryLog2=8 \
            --intra_pred_max_node_size_log2=6 --inferredDirectCodingMode=0 --maxNumQtBtBeforeOt=4\
            --uncompressedDataPath={} --compressedStreamPath={}".format(tmc_path, ply_path, gs_xyz_encoded_bin)
    tmc_encode_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    os.system(tmc_encode_cmd)
    end_time = time.time()
    print("GPCC encoding time: {}".format(end_time - start_time))
    print("Encoded bitstream length: {}".format(os.path.getsize(gs_xyz_encoded_bin)))
    return end_time - start_time, os.path.getsize(gs_xyz_encoded_bin)

def gpcc_decoder(ply_path, tmc_path):
    # tmc_path = '/home/old/huanghe/GS_repository/HGSC/comodules/tmc/linux/tmc3'

    gs_xyz_encoded_bin = os.path.splitext(ply_path)[0] + '.bin'

    gs_xyz_decoded_ply = os.path.splitext(ply_path)[0] + '_rec.ply'
    start_time = time.time()
    tmc_decode_cmd = "{} --mode=1 --compressedStreamPath={} --reconstructedDataPath={} --outputBinaryPly=0".\
                                        format(tmc_path, gs_xyz_encoded_bin, gs_xyz_decoded_ply)
    end_time = time.time()
    tmc_decode_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    os.system(tmc_decode_cmd)
    print("GPCC decoding time: {}".format(end_time - start_time))
    return end_time - start_time

def anchor_attribute_compressor(ply_path):
    coord, feat, ret = compress_use_reflectance(ply_path)
    print(ret)
    return coord, feat, ret

def LOD_attribute_compressor_zlib(anchor_points, anchor_attributes, LOD_points, LOD_attributes, bit_depth = [4, 8]):
    start_time = time.time()
    kd_tree = KDTree(anchor_points)
    _, indices = kd_tree.query(LOD_points)
    
    attribute_residuals = []
    for i, idx in enumerate(indices):
        nearest_anchor_attributes = anchor_attributes[idx]
        residual = LOD_attributes[i] - nearest_anchor_attributes
        attribute_residuals.append(residual)
    
    attribute_residuals = np.array(attribute_residuals)

    # 提取各部分数据
    features_dc = attribute_residuals[:, :3]
    features_rest_Y = attribute_residuals[:, 3:18]
    features_rest_U = attribute_residuals[:, 18:33]
    features_rest_V = attribute_residuals[:, 33:48]
    opacity = attribute_residuals[:, 48:49]
    scaling = attribute_residuals[:, 49:52]
    rotation = attribute_residuals[:, 52:56]
    
    # 量化函数
    def quantize(data, bits):
        data_min, data_max = data.min(), data.max()
        data_normalized = (data - data_min) / (data_max - data_min)  # 归一化到[0, 1]
        data_quantized = np.round(data_normalized * (2**bits - 1)).astype(np.int32)  # 量化到指定位深
        return data_quantized, data_min, data_max

    # 反量化函数
    def dequantize(data_quantized, data_min, data_max, bits):
        data_normalized = data_quantized / (2**bits - 1)  # 从整数范围[0, 2^bits - 1]恢复到[0, 1]
        data = data_normalized * (data_max - data_min) + data_min  # 从[0, 1]恢复到原始范围
        return data
    def bitstream_to_bytes(bitstream):
        # 将二进制比特流转换为字节数组
        byte_array = bytearray()
        for i in range(0, len(bitstream), 8):
            byte_array.append(int(bitstream[i:i+8], 2))
        return bytes(byte_array)
    def to_bitstream(data_quantized, bits):
        bitstream = ''.join([format(int(x), f'0{bits}b') for x in data_quantized.flatten()])
        # 添加填充位，使其长度能被8整除
        padding_length = (8 - len(bitstream) % 8) % 8
        bitstream += '0' * padding_length
        return bitstream, padding_length
    def compress_data(data_quantized, bits):
        # 将量化后的数据转换为比特流，并压缩成字节数组
        bitstream, padding_length = to_bitstream(data_quantized, bits)
        byte_data = bitstream_to_bytes(bitstream)
        compressed_data = zlib.compress(byte_data)
        return compressed_data, padding_length
    def decompress_data(compressed_data, original_shape, bits, padding_length):
        # 解压字节数组，并将其转换回二进制比特流
        byte_data = zlib.decompress(compressed_data)
        bitstream = ''.join(format(byte, '08b') for byte in byte_data)
        # 去除填充位
        bitstream = bitstream[:-padding_length] if padding_length > 0 else bitstream
        data_quantized = np.array([int(bitstream[i:i+bits], 2) for i in range(0, len(bitstream), bits)])
        return data_quantized.reshape(original_shape)

    def save_to_bin(filename, data):
        with open(filename, 'wb') as f:
            f.write(data)

    def load_from_bin(filename):
        with open(filename, 'rb') as f:
            data = f.read()
        return data
    
    # 量化处理
    features_dc_quant, dc_min, dc_max = quantize(features_dc, bit_depth[1])
    features_rest_Y_quant, y_min, y_max = quantize(features_rest_Y, bit_depth[1])
    features_rest_U_quant, u_min, u_max = quantize(features_rest_U, bit_depth[0])
    features_rest_V_quant, v_min, v_max = quantize(features_rest_V, bit_depth[0])
    opacity_quant, opacity_min, opacity_max = quantize(opacity, bit_depth[1])
    scaling_quant, scaling_min, scaling_max = quantize(scaling, bit_depth[1])
    rotation_quant, rotation_min, rotation_max = quantize(rotation, bit_depth[1])

    # 压缩数据
    features_dc_compressed,features_dc_padding = compress_data(features_dc_quant, bit_depth[1])
    features_rest_Y_compressed, features_rest_Y_padding = compress_data(features_rest_Y_quant, bit_depth[1])
    features_rest_U_compressed, features_rest_U_padding = compress_data(features_rest_U_quant, bit_depth[0])
    features_rest_V_compressed, features_rest_V_padding = compress_data(features_rest_V_quant, bit_depth[0])
    opacity_compressed, opacity_padding = compress_data(opacity_quant, bit_depth[1])
    scaling_compressed, scaling_padding = compress_data(scaling_quant, bit_depth[1])
    rotation_compressed, rotation_padding = compress_data(rotation_quant, bit_depth[1])

    LOD_bitstream = len(features_dc_compressed)+len(features_rest_Y_compressed)+len(features_rest_U_compressed)+len(features_rest_V_compressed)+len(opacity_compressed)+len(scaling_compressed)+len(rotation_compressed)
    encoding_time = time.time()
    # 解压数据
    features_dc_decompressed = decompress_data(features_dc_compressed, features_dc.shape, bit_depth[1], features_dc_padding)
    features_rest_Y_decompressed= decompress_data(features_rest_Y_compressed, features_rest_Y.shape, bit_depth[1], features_rest_Y_padding)
    features_rest_U_decompressed = decompress_data(features_rest_U_compressed, features_rest_U.shape, bit_depth[0], features_rest_U_padding)
    features_rest_V_decompressed = decompress_data(features_rest_V_compressed, features_rest_V.shape, bit_depth[0], features_rest_V_padding)
    opacity_decompressed = decompress_data(opacity_compressed, opacity.shape, bit_depth[1], opacity_padding)
    scaling_decompressed = decompress_data(scaling_compressed, scaling.shape, bit_depth[1], scaling_padding)
    rotation_decompressed = decompress_data(rotation_compressed, rotation.shape, bit_depth[1], rotation_padding)

    # 反量化处理
    features_dc_reconstructed = dequantize(features_dc_decompressed, dc_min, dc_max, bit_depth[1])
    features_rest_Y_reconstructed = dequantize(features_rest_Y_decompressed, y_min, y_max, bit_depth[1])
    features_rest_U_reconstructed = dequantize(features_rest_U_decompressed, u_min, u_max, bit_depth[0])
    features_rest_V_reconstructed = dequantize(features_rest_V_decompressed, v_min, v_max, bit_depth[0])
    opacity_reconstructed = dequantize(opacity_decompressed, opacity_min, opacity_max, bit_depth[1])
    scaling_reconstructed = dequantize(scaling_decompressed, scaling_min, scaling_max, bit_depth[1])
    rotation_reconstructed = dequantize(rotation_decompressed, rotation_min, rotation_max, bit_depth[1])
    
    # 重建 LOD_attributes
    attribute_residuals_reconstructed = np.concatenate([
        features_dc_reconstructed,
        features_rest_Y_reconstructed,
        features_rest_U_reconstructed,
        features_rest_V_reconstructed,
        opacity_reconstructed,
        scaling_reconstructed,
        rotation_reconstructed
    ], axis=1)
    
    LOD_attributes_reconstructed = attribute_residuals_reconstructed + anchor_attributes[indices]
    decoding_time = time.time()
    return LOD_attributes_reconstructed, LOD_bitstream, encoding_time - start_time, decoding_time - encoding_time

def LOD_attribute_compressor_AE(anchor_points, anchor_attributes, LOD_points, LOD_attributes, bit_depth = [4, 8]):
    kd_tree = KDTree(anchor_points)
    _, indices = kd_tree.query(LOD_points)
    
    attribute_residuals = []
    for i, idx in enumerate(indices):
        nearest_anchor_attributes = anchor_attributes[idx]
        residual = LOD_attributes[i] - nearest_anchor_attributes
        attribute_residuals.append(residual)
    
    attribute_residuals = np.array(attribute_residuals)

    # 提取各部分数据
    features_dc = attribute_residuals[:, :3]
    features_rest_Y = attribute_residuals[:, 3:18]
    features_rest_U = attribute_residuals[:, 18:33]
    features_rest_V = attribute_residuals[:, 33:48]
    opacity = attribute_residuals[:, 48:49]
    scaling = attribute_residuals[:, 49:52]
    rotation = attribute_residuals[:, 52:56]
    
    model = BinaryPPM(k=3)
    compressor = AECompressor(model)

    # 量化函数
    def quantize(data, bits):
        data_min, data_max = data.min(), data.max()
        data_normalized = (data - data_min) / (data_max - data_min)  # 归一化到[0, 1]
        data_quantized = np.round(data_normalized * (2**bits - 1)).astype(np.int32)  # 量化到指定位深
        return data_quantized, data_min, data_max

    # 反量化函数
    def dequantize(data_quantized, data_min, data_max, bits):
        data_normalized = data_quantized / (2**bits - 1)  # 从整数范围[0, 2^bits - 1]恢复到[0, 1]
        data = data_normalized * (data_max - data_min) + data_min  # 从[0, 1]恢复到原始范围
        return data
    def bitstream_to_bytes(bitstream):
        # 将二进制比特流转换为字节数组
        byte_array = bytearray()
        for i in range(0, len(bitstream), 8):
            byte_array.append(int(bitstream[i:i+8], 2))
        return bytes(byte_array)
    def to_bitstream(data_quantized, bits):
        bitstream = []
        for num in data_quantized.flatten():# 将量化后的数据转换为比特流
            binary_string = format(int(num), f'0{bits}b')
            bitstream.extend(int(bit) for bit in binary_string)
        padding_length = 0
        return bitstream, padding_length
    def compress_data(data_quantized, bits):
        # 将量化后的数据转换为比特流
        bitstream, padding_length = to_bitstream(data_quantized, bits)
        
        # bitstream_symbol = list(bitstream)
        # 使用算数编码器进行压缩
        compressed_data = compressor.compress(bitstream)
        padding_length = len(bitstream)
        return compressed_data, padding_length

    def decompress_data(compressed_data, original_shape, bits, padding_length):
        # 使用算数编码器进行解压
        bitstream = compressor.decompress(compressed_data, padding_length)
        bitstream = ''.join(map(str, bitstream))
        
        # 将比特流转换回量化数据
        data_quantized = np.array([int(bitstream[i:i+bits], 2) for i in range(0, len(bitstream), bits)])
        
        return data_quantized.reshape(original_shape)

    def save_to_bin(filename, data):
        with open(filename, 'wb') as f:
            f.write(data)

    def load_from_bin(filename):
        with open(filename, 'rb') as f:
            data = f.read()
        return data
    
    # 量化处理
    features_dc_quant, dc_min, dc_max = quantize(features_dc, bit_depth[1])
    features_rest_Y_quant, y_min, y_max = quantize(features_rest_Y, bit_depth[1])
    features_rest_U_quant, u_min, u_max = quantize(features_rest_U, bit_depth[0])
    features_rest_V_quant, v_min, v_max = quantize(features_rest_V, bit_depth[0])
    opacity_quant, opacity_min, opacity_max = quantize(opacity, bit_depth[1])
    scaling_quant, scaling_min, scaling_max = quantize(scaling, bit_depth[1])
    rotation_quant, rotation_min, rotation_max = quantize(rotation, bit_depth[1])

    # 压缩数据
    features_dc_compressed,features_dc_padding = compress_data(features_dc_quant, bit_depth[1])
    features_rest_Y_compressed, features_rest_Y_padding = compress_data(features_rest_Y_quant, bit_depth[1])
    features_rest_U_compressed, features_rest_U_padding = compress_data(features_rest_U_quant, bit_depth[0])
    features_rest_V_compressed, features_rest_V_padding = compress_data(features_rest_V_quant, bit_depth[0])
    opacity_compressed, opacity_padding = compress_data(opacity_quant, bit_depth[1])
    scaling_compressed, scaling_padding = compress_data(scaling_quant, bit_depth[1])
    rotation_compressed, rotation_padding = compress_data(rotation_quant, bit_depth[1])

    print(f"LOD残差bitstream长度: {len(features_dc_compressed)+len(features_rest_Y_compressed)+len(features_rest_U_compressed)+len(features_rest_V_compressed)+len(opacity_compressed)+len(scaling_compressed)+len(rotation_compressed)}")
    
    # 解压数据
    features_dc_decompressed = decompress_data(features_dc_compressed, features_dc.shape, bit_depth[1], features_dc_padding)
    features_rest_Y_decompressed= decompress_data(features_rest_Y_compressed, features_rest_Y.shape, bit_depth[1], features_rest_Y_padding)
    features_rest_U_decompressed = decompress_data(features_rest_U_compressed, features_rest_U.shape, bit_depth[0], features_rest_U_padding)
    features_rest_V_decompressed = decompress_data(features_rest_V_compressed, features_rest_V.shape, bit_depth[0], features_rest_V_padding)
    opacity_decompressed = decompress_data(opacity_compressed, opacity.shape, bit_depth[1], opacity_padding)
    scaling_decompressed = decompress_data(scaling_compressed, scaling.shape, bit_depth[1], scaling_padding)
    rotation_decompressed = decompress_data(rotation_compressed, rotation.shape, bit_depth[1], rotation_padding)

    # 反量化处理
    features_dc_reconstructed = dequantize(features_dc_decompressed, dc_min, dc_max, bit_depth[1])
    features_rest_Y_reconstructed = dequantize(features_rest_Y_decompressed, y_min, y_max, bit_depth[1])
    features_rest_U_reconstructed = dequantize(features_rest_U_decompressed, u_min, u_max, bit_depth[0])
    features_rest_V_reconstructed = dequantize(features_rest_V_decompressed, v_min, v_max, bit_depth[0])
    opacity_reconstructed = dequantize(opacity_decompressed, opacity_min, opacity_max, bit_depth[1])
    scaling_reconstructed = dequantize(scaling_decompressed, scaling_min, scaling_max, bit_depth[1])
    rotation_reconstructed = dequantize(rotation_decompressed, rotation_min, rotation_max, bit_depth[1])
    
    # 重建 LOD_attributes
    attribute_residuals_reconstructed = np.concatenate([
        features_dc_reconstructed,
        features_rest_Y_reconstructed,
        features_rest_U_reconstructed,
        features_rest_V_reconstructed,
        opacity_reconstructed,
        scaling_reconstructed,
        rotation_reconstructed
    ], axis=1)
    
    LOD_attributes_reconstructed = attribute_residuals_reconstructed + anchor_attributes[indices]
    
    return LOD_attributes_reconstructed
    

    
if __name__ == "__main__":
    # 示例数据
    anchor_points = np.random.rand(10, 3)
    anchor_attributes = np.random.rand(10, 56)
    LOD_points = np.random.rand(5, 3)
    LOD_attributes = np.random.rand(5, 56)

    # 调用函数
    LOD_attributes_reconstructed = LOD_attribute_compressor_AE(anchor_points, anchor_attributes, LOD_points, LOD_attributes)
    print("Reconstructed LOD attributes:", LOD_attributes_reconstructed)
    