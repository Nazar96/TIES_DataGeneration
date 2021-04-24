import warnings

warnings.filterwarnings("ignore")

# import tensorflow as tf
import numpy as np
import traceback
import cv2
import os
import string
import pickle
from multiprocessing import Process, Lock
from TableGeneration.Table import Table
from multiprocessing import Process, Pool, cpu_count
import random
import argparse
from TableGeneration.tools import *
import numpy as np
from selenium.webdriver import Firefox
from selenium.webdriver import PhantomJS
import warnings
from TableGeneration.Transformation import *
from uuid import uuid4


def warn(*args, **kwargs):
    pass


class Logger:
    def __init__(self):
        pass
        # self.file=open('logtxt.txt','a+')

    def write(self, txt):
        file = open('logfile.txt', 'a+')
        file.write(txt)
        file.close()


class GenerateTFRecord:
    def __init__(self, outpath, filesize, unlvimagespath, unlvocrpath, unlvtablepath, visualizeimgs, visualizebboxes,
                 distributionfilepath):
        self.outtfpath = outpath  # directory to store tfrecords
        self.filesize = filesize  # number of images in each tfrecord
        self.unlvocrpath = unlvocrpath  # unlv ocr ground truth files
        self.unlvimagespath = unlvimagespath  # unlv images
        self.unlvtablepath = unlvtablepath  # unlv ground truth of tabls
        self.visualizeimgs = visualizeimgs  # wheter to store images separately or not
        self.distributionfile = distributionfilepath  # pickle file containing UNLV distribution
        self.logger = Logger()  # if we want to use logger and store output to file
        # self.logdir = 'logdir/'
        # self.create_dir(self.logdir)
        # logging.basicConfig(filename=os.path.join(self.logdir,'Log.log'), filemode='a+', format='%(name)s - %(levelname)s - %(message)s')
        self.num_of_max_vertices = 10_000  # number of vertices (maximum number of words in any table)
        self.max_length_of_word = 100  # max possible length of each word
        self.row_min = 3  # minimum number of rows in a table (includes headers)
        self.row_max = 20  # maximum number of rows in a table
        self.col_min = 2  # minimum number of columns in a table
        self.col_max = 15  # maximum number of columns in a table
        self.minshearval = -0.1  # minimum value of shear to apply to images
        self.maxshearval = 0.1  # maxmimum value of shear to apply to images
        self.minrotval = -0.05  # minimum rotation applied to images
        self.maxrotval = 0.05  # maximum rotation applied to images
        self.num_data_dims = 5  # data dimensions to store in tfrecord
        # self.max_height = 2500  # max image height
        # self.max_width = 2500  # max image width
        self.tables_cat_dist = self.get_category_distribution(self.filesize)
        self.visualizebboxes = visualizebboxes

    def get_category_distribution(self, filesize):
        tables_cat_dist = [0, 0, 0, 0]
        firstdiv = filesize // 2
        tables_cat_dist[0] = firstdiv // 2
        tables_cat_dist[1] = firstdiv - tables_cat_dist[0]

        seconddiv = filesize - firstdiv
        tables_cat_dist[2] = seconddiv // 2
        tables_cat_dist[3] = seconddiv - tables_cat_dist[2]
        return tables_cat_dist

    def create_dir(self, fpath):  # creates directory fpath if it does not exist
        if (not os.path.exists(fpath)):
            os.mkdir(fpath)

    def str_to_int(self, str):  # converts each character in a word to equivalent int
        intsarr = np.array([ord(chr) for chr in str])
        padded_arr = np.zeros(shape=(self.max_length_of_word), dtype=np.int64)
        padded_arr[:len(intsarr)] = intsarr
        return padded_arr

    def convert_to_int(self, arr):  # simply converts array to a string
        return [int(val) for val in arr]

    def pad_with_zeros(self, arr, shape):  # will pad the input array with zeros to make it equal to 'shape'
        dummy = np.zeros(shape, dtype=np.int64)
        dummy[:arr.shape[0], :arr.shape[1]] = arr
        return dummy

    def draw_matrices(self, img, arr, matrices, imgindex, output_file_name):
        '''Call this fucntion to draw visualizations of a matrix on image'''
        no_of_words = len(arr)
        colors = np.random.randint(0, 255, (no_of_words, 3))
        arr = arr[:, 2:]

        img = img.astype(np.uint8)
        img = np.dstack((img, img, img))

        mat_names = ['row', 'col', 'cell']
        output_file_name = output_file_name.replace('.tfrecord', '')

        for matname, matrix in zip(mat_names, matrices):
            im = img.copy()
            x = 1
            indices = np.argwhere(matrix[x] == 1)
            for index in indices:
                cv2.rectangle(im, (int(arr[index, 0]) - 3, int(arr[index, 1]) - 3),
                              (int(arr[index, 2]) + 3, int(arr[index, 3]) + 3),
                              (0, 255, 0), 1)

            x = 4
            indices = np.argwhere(matrix[x] == 1)
            for index in indices:
                cv2.rectangle(im, (int(arr[index, 0]) - 3, int(arr[index, 1]) - 3),
                              (int(arr[index, 2]) + 3, int(arr[index, 3]) + 3),
                              (0, 0, 255), 1)

            img_name = os.path.join('bboxes/', output_file_name + '_' + str(imgindex) + '_' + matname + '.jpg')
            cv2.imwrite(img_name, im)

    def generate_img(self, driver, output_file_name, N=10):
        for _ in range(N):
            try:
                rows = np.random.randint(self.row_min, self.row_max)
                cols = np.random.randint(self.col_min, self.col_max)
                max_text_length = np.random.choice([1, 3, 5, 15])

                driver.set_window_size((cols*20 + max_text_length*10) * 6, (rows*15 + max_text_length*10) * 3)
                # This loop is to repeat and retry generating image if some an exception is encountered.
                # initialize table class
                table = Table(rows, cols, self.unlvimagespath, self.unlvocrpath, self.unlvtablepath,
                              3, self.distributionfile, max_text_length)
                # get table of rows and cols based on unlv distribution and get features of this table
                # (same row, col and cell matrices, total unique ids, html conversion of table and its category)
                html_1, html_2, id_count = table.create()

                # convert this html code to image using selenium webdriver. Get equivalent bounding boxes
                # for each word in the table. This will generate ground truth for our problem

                image_name = str(uuid4()) + '.png'
                path_1 = os.path.join(output_file_name, 'type_1', image_name)
                path_2 = os.path.join(output_file_name, 'type_2', image_name)

                im_1, _ = html_to_img(driver, html_1, id_count)
                im_2, _ = html_to_img(driver, html_2, id_count)

                im_1.save(path_1, )
                im_2.save(path_2, )
            except:
                pass

    def write_img(self, N=100):
        options = Options()
        options.add_argument("--headless")
        driver = Firefox(options=options)
        # driver.set_window_size(200, 200)

        self.create_dir('test')
        self.create_dir('test/type_1/')
        self.create_dir('test/type_2/')

        self.generate_img(driver, './test/', N)

        driver.stop_client()
        driver.quit()
