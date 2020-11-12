import os
import cv2
import cfg
from Synthtext.gen_i_t import datagen

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    i_t_dir = os.path.join(cfg.data_dir, cfg.i_t_dir)
    makedirs(i_t_dir)
    gen = datagen()
    gen.run_generate(i_t_dir)

if __name__ == '__main__':
    main()
