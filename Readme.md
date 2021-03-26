# Simple JPEG compress
## Environment
- python 3.X
- numpy==1.16.1
- opencv-python==4.1.2.30

or install require packages
`pip install -r requirement.txt`

## Usage
```
python main.py [-o [OUTPUT_DIR]] [-i [INPUT_FILE]] [-f [FACTOR]]
               [-s [SIZE]] [-ss [SUBSAMPLE]]
```

- `-o  [--output_dir]`: output encode file directory
- `-i  [--input_file]`: input file(s), relative path or absolute path. A single file name or multiple file names split with '|' 
					(e.g. -i "Test Images/GrayImages/Baboon.raw|Test Images/GrayImages/Lena.raw")
- `-f  [--factor]`    : The quantize factor(s). A int value in [5, 10, 20, 50, 80, 90] or several value in previous array split with ','
					(e.g. -f "5,10,20,50")
- `-s  [--size]`	  : Size of image. Int values of the format 'height,width'. If you input multiple images, every image must has the same size.
- `-ss [--subsample]` : Subsampling type of color space (only work on RGB image). Just support 4,1,1 or 4,2,2 or 4,4,4. Single tuple like '4,2,2' or multiple file names split with '|'
					(e.g. -s "4,2,2|4,4,1")
