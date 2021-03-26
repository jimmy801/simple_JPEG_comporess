import argparse
import errno
import math
import ntpath
import os
import time

import numpy as np
from encoder import encode
from decoder import decode


def size_of_tuple(s):
    try:
        return tuple([int(_.strip()) for _ in s.split(',')])
    except:
        raise argparse.ArgumentTypeError("Input must be 'a,b,c,...'.")


def parse_arguments():
    """
        Parse the command line arguments of the program.
        """

    parser = argparse.ArgumentParser(
        description="Simple jpeg compression."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="out/"
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="Origin image (relative path or absolute path) file(s). \
        A single file name or multiple file names split with '|'",
        default="Test Images/GrayImages/Baboon.raw|Test Images/GrayImages/Lena.raw|Test Images/ColorImages/BaboonRGB.raw|Test Images/ColorImages/LenaRGB.raw",
    )
    parser.add_argument(
        "-f",
        "--factor",
        type=str,
        nargs="?",
        help="The quantize factor(s). A int value in [5, 10, 20, 50, 80, 90] or several value in previous array split \
             with ','",
        default="5,10,20,50,80,90"
    )
    parser.add_argument(
        "-s",
        "--size",
        type=size_of_tuple,
        nargs="?",
        help="Size of image. Int values of the format 'height,width'",
        default=(512, 512)
    )
    parser.add_argument(
        "-ss",
        "--subsample",
        type=str,
        nargs="?",
        help="Subsampling type of color space (only work on RGB image). Just support 4,1,1 or 4,2,2 or 4,4,4.\
             Single tuple like '4,2,2' or multiple file names split with '|'(like '4,2,2|4,4,1)",
        default='4,1,1|4,2,2|4,4,4'
    )
    return parser.parse_args()


def main():
    """
        Description: Main function
        """

    # Argument parsing
    args = parse_arguments()

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    col, row = args.size

    # Check all input file(s)
    files = args.input_file.split('|')
    for file in files:
        if not os.path.isfile(file):
            raise OSError("Cannot open file {}".format(file))

    # Check all factor(s)
    factors = list(map(int, sorted(set(args.factor.split(',')), key=int)))
    for f in factors:
        if f not in [5, 10, 20, 50, 80, 90]:
            raise ValueError("Factor {} is illegal.".format(f))

    # Check all subsampling type
    sts = [size_of_tuple(_) for _ in args.subsample.split('|')]
    for st in sts:
        if st not in [(4, 4, 4), (4, 1, 1), (4, 2, 2)]:
            raise ValueError("Color sampling is not support {}.".format(st))

    for file in files:
        for factor in factors:
            source = np.fromfile(file, np.uint8)
            if source.shape[0] != row * col and source.shape[0] != row * col * 3:
                raise ValueError(
                    "Shape of file {} is illegal. Image size must {} * {}, given shape is {}".format(file, row, col,
                                                                                                     source.shape[0]))
            source = source.reshape((row, col, -1))

            for st in sts:
                pure_file_name = ntpath.basename(file)
                dot_pos = pure_file_name.rfind('.')
                pure_file_name = pure_file_name[0:len(pure_file_name) if dot_pos < 0 else dot_pos]
                if source.shape[-1] == 1:
                    out_file = os.path.join(args.output_dir, '{}_{}.jpg'.format(pure_file_name, factor))
                    encode_file = os.path.join(args.output_dir, '{}.en{}'.format(pure_file_name, factor))
                else:
                    out_file = os.path.join(args.output_dir, '{}_{}_{}.jpg'.format(pure_file_name, factor, st))
                    encode_file = os.path.join(args.output_dir, '{}_{}.en{}'.format(pure_file_name, st, factor))
                print('*' * 60)
                print('- Input file:', file)
                print('- Encode file:', encode_file)
                print('- Output file:', out_file)
                print('- Quantize factor:', factor)

                # start encode
                start = time.time()
                encode(factor, st, source, encode_file)
                end = time.time()
                print('- Encode spent {} seconds.'.format(end - start))

                # start decode
                start = time.time()
                jpg = decode(factor, st, encode_file, out_file, source.shape)
                end = time.time()
                print('- Decode spent {} seconds.'.format(end - start))

                print('-' * 60)
                print('- PSNR: ', cal_PSNR(source, jpg))
                print('*' * 60)
                print()
                if source.shape[-1] == 1:
                    break


def cal_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 100 if mse == 0 else 20 * math.log10(255. / math.sqrt(mse))


if __name__ == "__main__":
    main()
