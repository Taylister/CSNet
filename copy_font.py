import os
import shutil
from glob import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

INVISIBLE_CHARS = [' ', '　', '\n', '\r', '\t', '\a', '\b', '\f', '\v']
AVOIDED_CHARS = ['\\', '\0', '/', ':', '*', '?', '"', '<', '>', '|']
#FONT_EXTS = ['ttf', 'ttc', 'otf', 'otc']
FONT_EXTS = ['ttf']
ALPHABET_CAPS = [chr(i) for i in range(65, 65 + 26)]

class CopyFont():
    def __init__(self, src_font_dir_path, dst_dir_path):

        self.src_font_dir_path = os.path.normpath(src_font_dir_path)
        self.dst_dir_path = os.path.normpath(dst_dir_path)

        self._get_font_paths()

    def _get_font_paths(self):
        '''
        フォントパスの取得
        FONT_EXTSに含まれる拡張子のファイルを全て取得
        '''
        self.font_paths = list()
        for ext in FONT_EXTS:
            #if self.is_recursive:
            tmp = glob(self.src_font_dir_path + '/**/*.' + ext, recursive=True)
            # else:
            #     tmp = glob(self.src_font_dir_path + '/*.' + ext)
            self.font_paths.extend(tmp)

    def run(self):
        '''
        フォントファイルをコピーする
        '''
        pbar_font_paths = tqdm(self.font_paths)
        for font_path in pbar_font_paths:
            pbar_font_paths.set_description('{: <30}'.format(os.path.basename(font_path)))
            font_name = os.path.basename(os.path.splitext(font_path)[0])
            if os.path.exists(font_path):
                shutil.copy(font_path, self.dst_dir_path)
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy font files in the directory to Destinetion path.')
    parser.add_argument('src_font_dir_path', action='store', type=str, help='Directory path where source files are located.')
    parser.add_argument('dst_dir_path', action='store', type=str, help='Directory path of destination.')
    args = parser.parse_args()

    copy_font = CopyFont(args.src_font_dir_path, args.dst_dir_path)
    copy_font.run()
