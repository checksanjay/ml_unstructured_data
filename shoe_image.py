# %% [markdown]
import math
import csv
import time   
import pandas as pd
import os
from pandas import DataFrame
import numpy as np
import collections
from sklearn.model_selection import train_test_split
import random
import re
import numpy as np
import sklearn.cluster
import distance
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 
import os
import base64
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import os

# Set seed for reproducibility
np.random.seed(seed=1234)

# Import the csv file
# script_dir = os.getcwd() + '/ml/datasets/ebay/'
script_dir = '/Users/sgupta/Desktop/ebay/2021/'

training_set_file = 'mlchallenge_set_2021.tsv'

df = pd.read_csv(os.path.normcase(os.path.join(script_dir, training_set_file)), sep="\t", names= ['category', 'primary_image_url', 'all_image_urls', 'attributes', 'index'])
df = df.head(13000)
# print(df)


#  Row Ids:[315, 3792]
#  Row Ids:[451, 5354, 5975]
#  Row Ids:[467, 3345]
#  Row Ids:[567, 666, 4026, 4673, 4768, 4866, 5466, 5958, 7840, 8064, 9028]
#  Row Ids:[1826, 5094]
#  Row Ids:[2470, 8011]
#  Row Ids:[2529, 8800]
#  Row Ids:[4591, 5678]
#  Row Ids:[4744, 5409]
#  Row Ids:[6261, 6344, 6611]

def printImage(index):
    row = df.loc[df['index'] == index]
    all_image_urls = row.iloc[0]['all_image_urls']
    arr = all_image_urls.split(';')
    first_image = arr[0]
    print(first_image)
    image = Image.open(urllib.request.urlopen(first_image))
    plt.imshow(image)
    plt.show()

    hash = imagehash.average_hash(image)
    return hash


# printImage(315)
# printImage(3792)

hash1 = printImage(3752)
hash2 = printImage(10698)

# print(hash1 == hash2)
# print(hash1 - hash2)

input()