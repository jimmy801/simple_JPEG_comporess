import cv2
import numpy as np
from utils import special_int2bin_str, write_bit2file, get_quantization_table_by_factor, get_block_iterator, \
    zigzag_index_table, _DC_table, _AC_table_cw2rs as _AC_table, \
    _DC, _EOB, _ZRL, channel_select


def padding_img(img):
    """
        Pad zero util row and row of image are multiples of 8
        :param img: origin image
        :return: padded image
        """
    row, col, c = img.shape
    if row % 8 == 0 and col % 8 == 0:
        return img
    row = (row // 8 + 1) * 8
    col = (col // 8 + 1) * 8
    pad_img = np.zeros((row, col, img.shape[-1]))
    pad_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img
    return pad_img


def convert_color_space(img):
    """
        Convert RGB image to YCbCr image, return original image if image is grey level
        :param img: image will be converted
        :return: converted Image
        """
    return img if img.shape[-1] == 1 else cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)


def subsampling(img, sample_type):
    """
        CbCr channel subsample
        :param img: origin YCbCr image
        :param sample_type: tuple of subsampling type
        :return: subsample YCbCr image
        """
    row, col, c = img.shape

    if c != 3 or (sample_type == (4, 4, 4) and c == 3):
        return img
    Cr_blocks = cv2.boxFilter(img[:, :, 1], ddepth=-1, ksize=(2, 2))
    Cb_blocks = cv2.boxFilter(img[:, :, 2], ddepth=-1, ksize=(2, 2))
    if sample_type not in [(4, 1, 1), (4, 2, 2)]:
        raise ValueError("{} is not support.".format(sample_type))
    if sample_type == (4, 1, 1):
        SSV = 2
        SSH = 2
    else:
        SSV = 1
        SSH = 2
    Cr_blocks = Cr_blocks[::SSV, ::SSH]
    Cb_blocks = Cb_blocks[::SSV, ::SSH]
    img = [img[:, :, 0], Cr_blocks, Cb_blocks]
    return img


def get_quantization_blocks(img, qf, q_type):
    """
        Convert  image to 8*8 DCT quantization blocks
        :param img: original image
        :param qf: quantize qf
        :param q_type: luminance or chrominance table
        :return: quantization blocks
        """
    row, col = img.shape
    img = img - 128.
    dct_blocks = [cv2.dct(img[i:i + 8, j:j + 8]) for (i, j) in get_block_iterator(row, col)]
    q_t = get_quantization_table_by_factor(qf, q_type)
    quantization_blocks = [np.round(b / q_t).astype(np.int16) for b in dct_blocks]
    return quantization_blocks


def get_run_length(arr):
    """
        Get AC run-length coding
        :param arr: origin quantization block
        :return: AC run-length coding array
        """

    # Remove last zero(s)
    if arr == [0] * (8 * 8):
        arr = [0]
    else:
        for i, v in enumerate(arr[::-1]):
            if v != 0:
                arr = arr[:len(arr) - i]
                break

    # save (signal, origin value) or (count, value) (for AC)
    rl = []
    count = 0
    for i, v in enumerate(arr):
        if i == 0:
            rl.append((_DC, v))
        elif v == 0:
            count += 1
            if count > 15:
                rl.append((_ZRL, None))
                count = 0
        else:  # AC
            rl.append((count, v))
            count = 0
    rl.append((_EOB, None))

    return rl


def get_zigzag_data(blocks):
    """
        Get zigzag quantization data of all blocks
        :param blocks: quantization blocks of image
        :return: zigzag quantization data
        """
    zd = []
    for block in blocks:
        z_arr = [block[i, j] for i, j in zigzag_index_table]
        zd.append(get_run_length(z_arr))
    return zd


def get_category(val):
    """
        Get DC category
        :param val: difference of DC
        :return: category of DC
        """
    if val == 0:
        return 0

    val = abs(val)
    power = 0
    while True:
        res = 2 ** power
        if res > val:
            return power
        elif res == val:
            return power + 1
        else:
            power += 1


def encode(qf, st, img, en_filename):
    """
        Encode image to encoding file
        :param qf: quantization factor
        :param st: type of subsampling
        :param img: numpy array of image
        :param en_filename: encode file name
        """
    row, col, ch = img.shape
    img = padding_img(img)
    img = convert_color_space(img)
    img_channels = subsampling(img, st)

    encode_value = ''
    for c in range(ch):
        sc = channel_select[c % ch]
        try:  # grey level or no subsample image
            q_b = get_quantization_blocks(img_channels[:, :, c], qf, sc)
        except:
            q_b = get_quantization_blocks(img_channels[c], qf, sc)
        z_data = get_zigzag_data(q_b)

        last_DC = 0
        for block_data in z_data:
            for signal, value in block_data:
                if signal == _EOB:
                    codeword = _AC_table[sc][(0, 0)]
                elif signal == _DC:
                    diff_dc = value - last_DC
                    category_codeword = _DC_table[sc][get_category(diff_dc)]
                    diff_codeword = special_int2bin_str(diff_dc)
                    codeword = category_codeword + diff_codeword
                    last_DC = value
                elif signal == _ZRL:
                    codeword = _AC_table[sc][(15, 0)]
                else:  # AC
                    count = get_category(value)
                    run_size_code = _AC_table[sc][(signal, count)]
                    AC_coefficient_codeword = special_int2bin_str(value)
                    codeword = run_size_code + AC_coefficient_codeword
                encode_value += codeword

    write_bit2file(encode_value, en_filename)
