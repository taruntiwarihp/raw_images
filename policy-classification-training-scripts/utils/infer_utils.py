# -*- coding: utf-8 -*-
"""
Description: This file have pdf_to_text() func will convert pdf to text
and amazon_multithreading_ocr() will convert images docs to text. 
"""
__author__ = "Tarun Tiwari"
__copyright__ = "Copyright 2023, INJALA"
__credits__ = ["Tarun Tiwari", "Dhaval", "Prakash"]
__email__ = "tarun.tiwari@injala.com"
__version__ = "v2.0"

import json
import numpy as np
import boto3, os
import fitz, time
from PIL import Image
import pandas as pd
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm 
from threading import Thread
from queue import Queue

# Get the credentials values from the response
access_key = "AKIAV625SOF2MLXM3NFT"
secret_key = "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"
textract_client = boto3.client('textract', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name='us-east-2')
input_type = 'jpg'

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)

def pdf_to_text(pdf_file, abs_pth):

    data = {}
    data['Texts'] = []
    pdf_document = fitz.open(pdf_file)
    
    root = "{}/pdf_imgs".format(abs_pth)
    os.makedirs(root, exist_ok = "True")
    for idx, page_index in tqdm(enumerate(range(pdf_document.page_count))):

        page_pixmap = pdf_document.get_page_pixmap(page_index,matrix=fitz.Matrix(300/72,300/72),alpha=False)
        proc_img = Image.frombytes("RGB", [page_pixmap.width, page_pixmap.height], page_pixmap.samples)
        img_pth = "{}/{}.jpg".format(root, idx)
        proc_img.save(img_pth)

        with open(img_pth, 'rb') as file:
            file_bytes = file.read()
            
        response = textract_client.detect_document_text(Document={'Bytes': file_bytes})
        line_lst = []
        for pair in response['Blocks']:
            if pair['BlockType'] == "LINE":
                line_lst.append(pair['Text'])
            
        data['Texts'].append(''.join(line_lst))

        
    df = pd.DataFrame(data)
    df = df[pd.notnull(df['Texts'])]
    df=df.dropna(axis=1,how='all')
    df = df[df.astype(str)['Texts'] != 'This page intentionally left blank']
    
    df["lemm_sent"] = df["Texts"].apply(lambda text: lemmatize_words(text))
    df.to_csv("{}/inference.csv".format(root), index=False)

    return df


def wrapper(func, arg, arg1, queue):
    queue.put(func(arg, arg1))

def amazon_img_text(img, root):
    
    with open(img, 'rb') as file:
        file_bytes = file.read()
        
    response = textract_client.detect_document_text(Document={'Bytes': file_bytes})
    img_id = img.split("/")[-1].split(".")[0]
    save_file = open("{}/{}.json".format(root, img_id), "w")  
    json.dump(response, save_file, indent = 6)  
    save_file.close() 


def amazon_multithreading_ocr(pdf_file, abs_pth):
    data = {}
    data['Texts'] = []
    data['Img_pth'] = []

    start_thresh = 50
    st = 0

    pdf_document = fitz.open(pdf_file)
    root = "{}/pdf_imgs".format(abs_pth)

    os.makedirs(root, exist_ok = "True")

    for idx, page_index in tqdm(enumerate(range(pdf_document.page_count))):

        page_pixmap = pdf_document.get_page_pixmap(page_index,matrix=fitz.Matrix(300/72,300/72),alpha=False)
        proc_img = Image.frombytes("RGB", [page_pixmap.width, page_pixmap.height], page_pixmap.samples)
        img_pth = "{}/{}.jpg".format(root, idx)
        proc_img.save(img_pth)

        data['Img_pth'].append(img_pth)

    images_lst = list(data["Img_pth"])
    if len(images_lst) <= start_thresh:
        batch_images = None
        queue_list = []
        batch_images = images_lst

        for i in range(len(batch_images)):
            q  = Queue()
            queue_list.append(q)
            Thread(target=wrapper, args=(amazon_img_text, batch_images[i], root, queue_list[i])).start() 

        time.sleep(15)

    else:
        
        for bt in range(start_thresh, len(images_lst), start_thresh):
            batch_images = None
            queue_list = []
            batch_images = images_lst[st:bt]
            
            for i in range(len(batch_images)):
                q  = Queue()
                queue_list.append(q)
                Thread(target=wrapper, args=(amazon_img_text, batch_images[i], root, queue_list[i])).start() 

            time.sleep(10)

            if len(images_lst) - bt <= start_thresh:
                batch_images = images_lst[bt:]
                for i in range(len(batch_images)):
                    q  = Queue()
                    queue_list.append(q)
                    Thread(target=wrapper, args=(amazon_img_text, batch_images[i], root, queue_list[i])).start() 

                time.sleep(10)

            st = bt

    for imgs in data['Img_pth']:

        line_lst = []
        json_pth = imgs.replace("jpg", "json")
        with open(json_pth, encoding="utf8") as f:
            json_file = json.load(f)

        for pair in json_file['Blocks']:
            if pair['BlockType'] == "LINE":
                line_lst.append(pair['Text'])
            
        data['Texts'].append(" ".join(line_lst))

    df = pd.DataFrame(data)
    df.drop_duplicates(subset=['Texts'], keep='first', inplace=True)
    df['Texts'].replace('', np.nan, inplace=True)
    df.dropna(subset=['Texts'], inplace=True)
    
    df = df[pd.notnull(df['Texts'])]
    df=df.dropna(axis=1,how='all')
    df = df[df.astype(str)['Texts'] != 'This page intentionally left blank']
    df["lemm_sent"] = df["Texts"].apply(lambda text: lemmatize_words(text))
    df.to_csv("{}/amazon_ocr.csv".format(abs_pth), index=False)

    return df