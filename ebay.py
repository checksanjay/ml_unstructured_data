# # %% [markdown]
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

# Set seed for reproducibility
np.random.seed(seed=1234)

# Import the csv file
# script_dir = os.getcwd() + '/ml/datasets/ebay/'
script_dir = '/Users/sgupta/Desktop/ebay/2021/'

training_set_file = 'mlchallenge_set_2021.tsv'

df = pd.read_csv(os.path.normcase(os.path.join(script_dir, training_set_file)), sep="\t", names= ['category', 'primary_image_url', 'all_image_urls', 'attributes', 'index'])
df = df.head(60000)
# print(df)

all_brands = [];
all_colors = [];
all_styles = [];
all_size = [];
all_material = [];

matDict = {}
colorDict = {}
brandDict = {}
styleDict = {}
sizeDict = {}

rowDict = {} # main dictionary

matchDict = {}
rowIdToClusterDict = {}
cluster_to_row_ids_dict = {}
allRowIdsToClusterIdsDict = {}
val_cluster_to_row_dict = {}
val_row_to_cluster_dict = {}
subsetDict = {}
submissionDict = {}

def get_unique_values(data):    
    lst = []
    unique_numbers = set(data)
    for number in unique_numbers:
        lst.append(number)
    return lst

def printSummary(type, data, dict):   
    unique_data = get_unique_values(data)
    unique_data.sort()  
    index = 0
    for val in unique_data:
        index = index + 1
        print(index, ': ', val)
    
    print()
    print('#######')
    print('Total unique' , type, ':', len(unique_data))
    print('#######')
    print()
    for i in dict:
        print(i, dict[i])
        print('#######')
        print()

def create_one_to_one_dict(key, value, dict ):
        dict[key] = value

def create_dict(value, rowId, dict ):
    if value in dict:
        curr_ids = dict[value]
        if rowId not in curr_ids:
            curr_ids.append(rowId)
            dict[value] = curr_ids
    else:
        curr_ids = []
        curr_ids.append(rowId)
        dict[value] = curr_ids

def parse_brand(rowId, rowData):
    # find brand
    index = rowData.find('brand:')
    if (index > -1):
        index = index + len('brand:')
        commaIndex = rowData.find(',', index, len(rowData))
        brand_val = rowData[index: commaIndex]
        if (len(brand_val) < 25):
            brand_val = brand_val.strip()
            create_dict(brand_val, rowId, brandDict)
            all_brands.append(brand_val)        
    else:
        brand_val = ''
        all_brands.append(brand_val)
    return brand_val 

def parse_style(rowId, rowData):
    # find style
    rowData = rowData.replace('shoe\'s', ' ')
    rowData = rowData.replace('shoes', ' ')
    rowData = rowData.replace('shoe', ' ')
    index = rowData.find('style:')
    if (index > -1):
        index = index + len('style:')
        commaIndex = rowData.find(',', index, len(rowData))
        style_val = rowData[index: commaIndex]
        if (len(style_val) < 25):
            style_val = style_val.strip()
            create_dict(style_val, rowId, styleDict)
            all_styles.append(style_val)
    else:
        style_val = ''
        all_styles.append(style_val)
    return style_val 

def parse_color(rowId, rowData):
    # find color
    index = rowData.find('color:')
    if (index > -1):
        index = index + len('color:')
        commaIndex = rowData.find(',', index, len(rowData))
        color_val = rowData[index: commaIndex]
        # print('color:', color_val)
        if (len(color_val) < 30):
            strings = color_val.split("/")
            for color in strings:
                color = color.strip()                
                create_dict(color, rowId, colorDict)
                all_colors.append(color)
    else:
        color_val = ''
        all_colors.append(color_val)
    return color_val 

def parse_size(rowId, rowData):
    rowData = rowData.replace('euro size', '')
      # find size
    index = rowData.find('size')
    if (index > -1):
        index = index + len('size')
        colonIndex = rowData.find(':', index, len(rowData))
        commaIndex = rowData.find(',', colonIndex, len(rowData))
        size_val = rowData[colonIndex+1: commaIndex]
        # print('size:', size_val)
        numbers = sum(c.isdigit() for c in size_val)
        if (numbers > 0) :
            size_val = size_val.replace('us', '')
            size_val = size_val.replace('us ', '')

            size_val = size_val.replace('uk', '')
            size_val = size_val.replace('uk ', '')

            size_val = size_val.replace('eur', '')
            size_val = size_val.replace('eur ', '')
            size_val = size_val.replace('eu', '')
            size_val = size_val.replace('eu ', '')

            size_val = size_val.replace('size', '')
            size_val = size_val.replace('size ', '')

            size_val = size_val.replace('men ', '')
            size_val = size_val.replace('mens ', '')
            size_val = size_val.replace('men\'s ', '')

            size_val = size_val.replace('women ', '')
            size_val = size_val.replace('womens ', '')
            size_val = size_val.replace('women\'s ', '')
            size_val = size_val.replace('wo', '')

            if ( size_val.find('&') == -1 ):
               size_val = size_val_calc(size_val, rowId)
            else:
                arr = size_val.split('&')
                # create list of size values
                lst = []
                for size_val in arr:                    
                    size_val = size_val_calc("".join(size_val.split()), rowId)
                    lst.append(size_val)
                return lst
    else:
        size_val = 0
        all_size.append(str(size_val))

    if (size_val == 'queen'):
        size_val = 100
        create_dict(size_val, rowId, sizeDict)
    if (size_val == 'king'):
        size_val = 200
        create_dict(size_val, rowId, sizeDict)
    return size_val 

def size_val_calc(size_val, rowId):
    size_val = size_val.split(' ', 1)[0]
    index = 0
    for letter in size_val:
        if (letter.isnumeric() or letter == '.' ):
            index = index + 1
        else:
            break
    if (index < len(size_val)):
        size_val = size_val[0: index]
    if (len(size_val) > 0):
        try:
            size_val = float(size_val)
            if (size_val >= 1 and size_val < 20):
                create_dict(size_val, rowId, sizeDict)
                all_size.append(str(size_val))
        except ValueError:
            size_val = 0
        all_size.append(str(size_val))
    return size_val

def parse_material(rowId, rowData):
    rowData = rowData.replace('does', ' ')
     # find material
    index = rowData.find('material:')
    if (index > -1):
        index = index + len('material:')
        commaIndex = rowData.find(',', index, len(rowData))
        mat_val = rowData[index: commaIndex]
        mat_val = mat_val.replace('%', ' ')
        mat_val = mat_val.replace('#', ' ')
        index = 0
        for letter in mat_val:
            if (letter.isdigit()):
                index = index + 1
            else:
                break
        if (index < len(mat_val)):
            spaceIndex = mat_val.find(' ', index, len(mat_val))
            if (spaceIndex >= 0):
                mat_val = mat_val[index:spaceIndex]
            slashIndex = mat_val.find('/', 0, len(mat_val))
            if (slashIndex >= 0):
                mat_val = mat_val[index:slashIndex]
            hypenIndex = mat_val.find('-', 0, len(mat_val))
            if (hypenIndex >= 0):
                mat_val = mat_val[index:hypenIndex]
        if (len(mat_val) > 3 and len(mat_val) < 25):
            mat_val = mat_val.strip() 
            create_dict(mat_val, rowId, matDict)
            all_material.append(mat_val)
    else:
        mat_val = ''
        all_material.append(mat_val)
    return mat_val 

def printRowData(rowId, rowData):
    print(rowId, ' : ', rowData)
    print('#######')

def parseRowData(rowId, rowData):
    rowData = rowData.lower() 
    # rowIds = [8528, 11726]
    # for id in rowIds:
    #     if (rowId == id):            
    #         printRowData(rowId, rowData)
           
    brand_val = parse_brand(rowId, rowData)
    style_val = parse_style(rowId, rowData)

    color_val = parse_color(rowId, rowData)
    mat_val = '' # parse_material(rowId, rowData)
    size_val = parse_size(rowId, rowData)
    # if size_val is list, create 2 rowData
    # if (str(size_val).find(',') == -1):
    rowData =	{
        "brand": brand_val,
        "style": style_val,
        "size": size_val,
        "color": color_val,
        "material": mat_val
    }
   
    # create dict
    rowDict[rowId] = rowData

def isColorSame(first, second):

    if ( second == first or second == 'multicolor' or second == 'multi-color' or
                    first== 'multicolor' or first== 'multi-color'):
        return True

    if (len(first) > 0 and len(second) > 0):
        # replace / by space 
        firstItems = first.replace('/', ' ')
        secondItems = second.replace('/', ' ')
        if ( set(firstItems) == set(secondItems) ):
            return True
        if (first.find(second) != -1 or second.find(first) != -1):
                return True   
    return False

def createMatchDict():
    # first parameter, size, find all unique keys in sizeDict
    cluster = 0
    for index in sizeDict:
        # debug
        if ( index == 12.0):
            # for key in sizeDict[index]:
            #     print('key', key)
            k = 1
        # find all rowIds having same size    
        rowIds = sizeDict[index]
        i = 0
        while i < len(rowIds):
            first = rowIds[i]
            # debug
            if ( first == 4768):
                k = 1
            # get first origData
            firstRowData = rowDict[first]
            j = i+1
            while j < len(rowIds):
                # get next id
                next = rowIds[j]
                if ( next == 10396):
                    k = 1
                # find next row data
                nextRowData = rowDict[next]
                j += 1
                if (nextRowData['brand'] == firstRowData['brand'] and 
                    nextRowData['style'] == firstRowData['style'] and
                    ( isColorSame(firstRowData['color'] , nextRowData['color']) == True)):
                        create_dict(first, next, matchDict) 
                else:
                    # comment
                    k = 1
            i += 1

def markSameCluster(rowId, clusterID):
    if rowId not in rowIdToClusterDict:
        rowIdToClusterDict[rowId] = clusterID
        if rowId in matchDict:
            allRowIds = matchDict[rowId]
            for id in allRowIds:
                # # debug
                # if (id == 4026):
                #     k = 1
                rowIdToClusterDict[id] = clusterID

def createFinalCluster():
    clusterID = 1
    for rowId in matchDict:
        # # debug
        # rowtest = 8528
        # if (rowId == rowtest):
        #     k = 1
        markSameCluster(rowId, clusterID)
        clusterID = clusterID + 1
   
def printUniqueClusters():   
    list_clusters = []
    for key, value in rowIdToClusterDict.items():
        # value is cluster id
        list_clusters.append(value)
    numClusters = np.unique(list_clusters)

    print('#######')
    print('Total Unique Cluster', len(numClusters))
    print('#######')

def create_orig_cluster_file(fileName, dict):
    fileObj = open(fileName, "w")  # append mode 
    # print cluster id and matching elements rowIds
    for key, value in rowIdToClusterDict.items():
        create_dict(value, key, dict) 

    for cluster in dict:         
        temp = 'Cluster:' + str(cluster) + ' Row Ids:' + str(dict[cluster]) + '\n'
        fileObj.write(temp)
    fileObj.close() 

def readValidateFileAndCreateClusters():
    validation_set_file = 'mlchallenge_set_validation.tsv'
    df_validated = pd.read_csv(os.path.normcase(os.path.join(script_dir, validation_set_file)), 
                        sep="\t", names= ['rowId', 'clusterId'])
    
    for index, row in df_validated.iterrows():
        rowId = row["rowId"]
        clusterId = row["clusterId"]
        # create 2 dict, cluster to row and row to cluster
        create_dict(clusterId, rowId, val_cluster_to_row_dict)
        create_dict(rowId, clusterId, val_row_to_cluster_dict)
        index = index + 1

    for cluster in val_cluster_to_row_dict:      
        rowIds =  val_cluster_to_row_dict[cluster]  
        if (len(rowIds) > 1):
            print('cluster id', cluster, 'row ids', val_cluster_to_row_dict[cluster])

def assignUniqueClusterToAllRows():
    # add unique cluster to single rowID
    clusterId = 1
    for rowId in rowDict:
        clusterId += 1
        # assign same cluster
        # clusterId = base64.b64encode(os.urandom(6)).decode('ascii')
        create_one_to_one_dict(rowId, clusterId, allRowIdsToClusterIdsDict)

def prepareExcelAllRowsDataToCluster():
    # assign unique cluster id to all row Ids, so all rows have different cluster id
    assignUniqueClusterToAllRows()
     
    clusterId = 20000000
    # now, assign same clustser id to those rows, which falls in one bucket
    for rowId in cluster_to_row_ids_dict:
        # clusterId = base64.b64encode(os.urandom(6)).decode('ascii')
        clusterId += 1
        for rowId in cluster_to_row_ids_dict[rowId]:  
            # delete old entry
            allRowIdsToClusterIdsDict[rowId] = []   
            create_one_to_one_dict(rowId, clusterId, allRowIdsToClusterIdsDict)

def createSubsetDictFromAllRowsDict():
    # test validate data with our data
    for rowId in allRowIdsToClusterIdsDict: 
        if (rowId in val_row_to_cluster_dict):
            create_dict(allRowIdsToClusterIdsDict[rowId], rowId, subsetDict)
            create_one_to_one_dict(rowId, allRowIdsToClusterIdsDict[rowId], submissionDict)

def printSubsetDict(fileName, dict):
    fileObj = open(fileName, "w")  # append mode 
    counterCluster = 0
    for clus in dict:
        lst = dict[clus]    
        if (len(lst) > 1):
            counterCluster += 1
            temp = 'Cluster:' + str(clus) + ' Row Ids:' + str(dict[clus]) + '\n' + ' Row Ids:' + str(lst) + '\n' + '#######' + '\n'
            fileObj.write(temp)
            for row in lst:
                temp = 'Row ID :' + str(rowDict[row]) + '\n'
                fileObj.write(temp)
            fileObj.write('\n')
    return counterCluster

def start_read_rows_create_row_dict():
    index = 0
    for index, nextRow in df.iterrows():
        attr = nextRow["attributes"]
        index = nextRow["index"]
        parseRowData(index, attr)
        index = index + 1

def create_submission_file(fileName, dict):
    with open(fileName, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for key, value in dict.items():
            tsv_writer.writerow([key, value])

start_read_rows_create_row_dict()

# row ids, maps to matching row ids
createMatchDict()

# test
print('32841', rowDict[32841])
print('52013', rowDict[52013])

# it creates rowIdToClusterDict, cluster get starts from 1
createFinalCluster()

# creates orig_cluster.txt file
create_orig_cluster_file("orig_cluster.txt", cluster_to_row_ids_dict)

# prints file how many unique clusters got created
printUniqueClusters()

# prepare allRowIdsToClusterIdsDict, assigns same cluster ids to different row ids, if they fall into same bucket
prepareExcelAllRowsDataToCluster()

# read validate file given by ebay
readValidateFileAndCreateClusters()

# creates clusters from specific row ids as given in validation file
# also creates submissionDict, file for submission
createSubsetDictFromAllRowsDict()

# print submission file
create_submission_file("ebay_submission.tsv", submissionDict)

# prints cluster and matching row ids
# subsetDict : validation cluster and row ids mapping
counterCluster = printSubsetDict("subset_cluster.txt", subsetDict)
print('counterCluster: ', counterCluster)

# test
print('567', rowDict[567])
print('353342', rowDict[353342])
print('#######')
print('2742', rowDict[2742])
print('12020', rowDict[12020])
print('#######')
print('4026', rowDict[4026])
print('8064', rowDict[8064])
print('#######')
print('8528', rowDict[8528])
print('11726', rowDict[11726])
print('#######')



# printSummary('materials', all_material, matDict)
# printSummary('colors', all_colors, colorDict)
# printSummary('sizes', all_size, sizeDict)
# printSummary('brands', all_brands, brandDict)
# printSummary('styles', all_styles, styleDict)


# lst = [4026, 8064]
# for row in lst:
#     print('row', row, ':', rowDict[row])
# print('#######')

# lst = [8528, 11726]
# for row in lst:
#     print('row', row, ':', rowDict[row])


# debug, make sure that matchDict does not contain any duplicate
# dictKeys = {}
# for key in matchDict:
#     if key in dictKeys:
#         print(key, 'alredy exist')
#     else:
#         dictKeys[key] = key
#     for val in matchDict[key]:
#          if val in dictKeys:
#             print(val, 'alredy exist')
#          else:
#             dictKeys[val] = val
# %%
