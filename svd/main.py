import numpy as np
import argparse
from PIL import Image
import struct
import time
from numpy.linalg import qr

SVD_HEADER = b'SVDS'
HEADER_SIZE = 16

def compress_numpy(channel: np.ndarray, k: int):
    u, s, vt = np.linalg.svd(channel, full_matrices=False)
    data = np.concatenate((u[:, :k].flatten(), s[:k], vt[:k, :].flatten()))
    return data.astype(np.float32).tobytes()

def find_eigenvector_and_value(channel, v, time_bound, eps):
    transposed_product = np.dot(channel.T, channel)
    while time.time() * 1000 < time_bound:
        v_new = np.dot(transposed_product, v)
        v_new /= np.linalg.norm(v_new)
        if np.allclose(v_new, v, eps):
            break
        v = v_new
    eigenvalue = np.dot(np.dot(transposed_product, v), v.T)
    return v, eigenvalue

def simplified_channel_compression(channel: np.ndarray, rank: int, time_limit: int):
    eps = 1e-10
    np.random.seed(0)
    n, m = channel.shape
    v = np.random.rand(m)
    v /= np.linalg.norm(v)
    u = np.zeros((n, rank))
    sigma = np.zeros(rank)
    vt = np.zeros((rank, m))

    time_bound = time.time() * 1000 + time_limit
    for i in range(rank):
        v, eigenvalue = find_eigenvector_and_value(channel, v, time_bound, eps)
        vt[i, :] = v
        u[:, i] = np.dot(channel, v) / eigenvalue
        sigma[i] = eigenvalue
        channel = channel - eigenvalue * np.outer(u[:, i], v)

    compressed_data = np.concatenate((u.flatten(), sigma, vt.flatten()))
    return compressed_data.astype(np.float32).tobytes()

def advanced_svd_compression(channel_data, rank, time_limit):
    n, m = channel_data.shape
    u = np.zeros((n, rank))
    sigma = np.zeros(rank)
    v = np.zeros((m, rank))

    eps = 1e-10
    time_bound = time.time() * 1000 + time_limit

    i = 0
    while time.time() * 1000 < time_bound:
        q, r = np.linalg.qr(np.dot(channel_data, v))
        u = q[:, :rank]

        q, r = np.linalg.qr(np.dot(channel_data.T, u))
        v = q[:, :rank]

        sigma = np.diag(r[:rank, :rank])
        i += 1
        if (np.allclose(np.dot(channel_data, v), np.dot(u, r[:rank, :rank]), eps) and i % 10 == 0):
            break 

    # Flatten U, S, and V and concatenate them into a single bytearray
    compressed_data = np.concatenate((u.flatten(), sigma, v.T.flatten()))
    return compressed_data.astype(np.float32).tobytes()

def compress_image(in_file, out_file, method, compression):
    # Load the image to compress
    image_to_compress = Image.open(in_file)
    n, m = image_to_compress.height, image_to_compress.width
    k = np.floor(n * m / (4 * compression * (n + m + 1))).astype(int)

    # Process the image, compress it and save to the custom format
    image_data = np.asarray(image_to_compress)
    compressed_data = bytes()
    for color_channel in range(3):
        channel_data = image_data[..., color_channel]
        if method == "numpy":
            compressed_data += compress_numpy(channel_data, k)
        elif method == "simple":
            compressed_data += simplified_channel_compression(channel_data, k, 1000)
        elif method == "advanced":
            compressed_data += advanced_svd_compression(channel_data, k, 1000)
        else:
            raise NotImplementedError('There is no another methods yet')
    with open(out_file, 'wb') as file:
        header_info = SVD_HEADER + struct.pack('<III', n, m, k)
        file.write(header_info)
        file.write(compressed_data)

def decompress_image(input_file, result_image_name):
    with open(input_file, 'rb') as f:
        header_data = f.read(HEADER_SIZE)
        if header_data[:len(SVD_HEADER)] != SVD_HEADER:
            raise ValueError(f'Incorrect format of {input_file}')
        
        # Unpack the header information
        n, m, k = struct.unpack('<III', header_data[4:])

        # Initialize a list to hold the image channels
        image_channels = []

        for _ in range(3):
            # Read and unpack the channel data
            channel_data = f.read(4 * k * (n + m + 1))
            u = np.frombuffer(channel_data[:4 * n * k], dtype=np.float32).reshape(n, k)
            sigma = np.frombuffer(channel_data[4 * n * k: 4 * n * k + 4 * k], dtype=np.float32)
            vt = np.frombuffer(channel_data[4 * n * k + 4 * k:], dtype=np.float32).reshape(k, m)
            
            # Reconstruct the channel by multiplying the matrices and add it to the list
            channel_channel = np.dot(u, np.diag(sigma)).dot(vt)
            image_channels.append(channel_channel)

        # Stack the channels, clip to valid range and convert to uint8
        image_channel = np.stack(image_channels, axis=2).clip(0, 255).astype(np.uint8)

        # Create and save the image
        result_image = Image.fromarray(image_channel)
        result_image.save(result_image_name)

def main():
    parser = argparse.ArgumentParser(description="Compress or decompress an image.")
    parser.add_argument("--mode", type=str, choices=["compress", "decompress"], required=True)
    parser.add_argument("--method", type=str, choices=["numpy", "simple", "advanced"])
    parser.add_argument("--compression", type=int)
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "compress":
        compress_image(args.in_file, args.out_file, args.method, args.compression)
    elif args.mode == "decompress":
        decompress_image(args.in_file, args.out_file)

if __name__ == "__main__":
    main()