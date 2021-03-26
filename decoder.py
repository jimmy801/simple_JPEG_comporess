import cv2
import numpy as np
from utils import special_bin_str2int, read_binstr_frome_file, get_quantization_table_by_factor, get_block_iterator, \
    zigzag_value_table, zigzag_index_table, _DC_table, _AC_table_rs2cw as _AC_table, \
    _DC, _EOB, _ZRL, channel_select


def decode_AC_DC(bin_str, img_shape, st):
    """
        Decode signal and value of DC, and decode run size and codeword of AC from encode binary string
        :param bin_str: encode binary string
        :param img_shape: shape of image
        :param st: type of subsampling
        :return: decode zigzag blocks
        """
    row, col, c = img_shape
    block_size = (row // 8) * (col // 8)
    read_stream = ''
    isDC = True
    last_DC = 0
    block = []
    zigzag_blocks = []
    img_chanels = []
    sc = 0

    from io import StringIO
    bs = StringIO(bin_str)
    while True:
        read_stream += bs.read(1)

        if not read_stream:
            break

        if isDC:
            try:
                idx = _DC_table[channel_select[sc]].index(read_stream)
                v = 0
                if idx == 0:
                    next_bits = bs.read(1)
                    if next_bits != '0':
                        v = special_bin_str2int(next_bits)
                else:
                    next_bits = bs.read(idx)
                    v = special_bin_str2int(next_bits)
                last_DC += v
                block.append((_DC, v))
                isDC = False
                read_stream = ''
            except:
                pass
        else:
            try:
                run, size = _AC_table[channel_select[sc]][read_stream]
                if run == 0 and size == 0:
                    block.append((_EOB, None))
                    zigzag_blocks.append(block)
                    if len(zigzag_blocks) == block_size:
                        img_chanels.append(zigzag_blocks)
                        zigzag_blocks = []
                        sc = 1
                        if len(img_chanels) == c:
                            break
                        if st == (4, 1, 1):
                            block_size = (row // 8 // 2) * (col // 8 // 2)
                        elif st == (4, 2, 2):
                            block_size = (row // 8) * (col // 8 // 2)
                    block = []
                    isDC = True
                elif run == 15 and size == 0:
                    block.append((_ZRL, None))
                else:  # AC
                    next_bits = bs.read(size)
                    v = special_bin_str2int(next_bits)
                    block.append((run, v))
                read_stream = ''
            except:
                pass
    return img_chanels


def zigzag(arr):
    """
        Go through zigzag path
        :param arr: array to go zigzag
        :return: zigzag array
        """
    block_row, block_col = 8, 8
    z = np.zeros((block_row, block_col), np.int16)
    for i in range(block_row):
        for j in range(block_col):
            z[i, j] = arr[zigzag_value_table[i][j]]
    return z


def de_zigzag(zigzag_blocks):
    """
        Decode from zigzag blocks
        :param zigzag_blocks: decode zigzag blocks
        :return: decoded blocks
        """
    blocks = []
    zigzag_arr = []
    last_DC = 0
    for zb in zigzag_blocks:
        for s, v in zb:
            if s == _ZRL:
                zigzag_arr.extend([0] * 15)
            elif s == _EOB:
                zigzag_arr.extend([0] * (64 - len(zigzag_arr)))
                blocks.append(zigzag(zigzag_arr))
                zigzag_arr.clear()
            elif s == _DC:
                last_DC += v
                zigzag_arr.append(last_DC)
            else:  # AC
                zigzag_arr.extend([0] * s)
                zigzag_arr.append(v)
    return blocks


def get_dequantization_img_blocks(q_blocks, q_t):
    """
        get deQuantization image blocks form given quantization blocks and quantization table
        :param q_blocks: quantization blocks
        :param q_t: quantization table
        :return:  deQuantization image blocks
        """
    dct_blocks = [block * q_t * 1. for block in q_blocks]
    img_blocks = [np.round(cv2.idct(block)).astype(np.int8) for block in dct_blocks]
    return img_blocks


def decode(qf, st, en_filename, jpg_filename, img_shape):
    """
        Decode to jpeg file from encode file, and return numpy array of jpeg file
        :param qf: quantization factor
        :param st: type of subsample
        :param en_filename: encode file name
        :param jpg_filename: jpeg file name
        :param img_shape: image size
        :return: numpy array of output jpeg file
        """
    row, col, ch = img_shape
    en_bin = read_binstr_frome_file(en_filename)
    zbs = decode_AC_DC(en_bin, img_shape, st)
    sample_size = (row, col)
    if st == (4, 1, 1):
        sample_size = ((row // 2), (col // 2))
    elif st == (4, 2, 2):
        sample_size = (row, (col // 2))
    img = np.zeros((row, col, ch), np.uint8)
    for c, zb in enumerate(zbs):
        blocks = de_zigzag(zb)
        q_t = get_quantization_table_by_factor(qf, channel_select[c % len(zbs)])
        img_blocks = get_dequantization_img_blocks(blocks, q_t)

        b_r, b_c = (row, col) if c == 0 else sample_size

        tmp = np.ones((b_r, b_c), np.int8) * 128
        for i, (row_offset, col_offset) in enumerate(get_block_iterator(b_r, b_c)):
            tmp[row_offset:row_offset + 8 if row_offset + 8 <= b_r else b_r,
                col_offset:col_offset + 8 if col_offset + 8 <= b_c else b_c] += img_blocks[i]

        # inverse subsample
        img_blocks = cv2.resize(tmp, (row, col))

        img[:, :, c] = np.round(img_blocks)

    if ch == 3:
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(jpg_filename, img)

    return img
