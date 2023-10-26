# -*- coding: utf-8 -*-
"""
Description: Inference
"""
__author__ = "Tarun Tiwari"
__copyright__ = "Copyright 2023, INJALA"
__credits__ = ["Tarun Tiwari"]
__email__ = "tarun.tiwari@injala.com"
__version__ = "v1.0"

import torch
import time
import numpy as np

from transformers import BertTokenizer
from utils.config import parse_args
from tqdm import tqdm
from models.bert import BERT_Model
from utils.infer_utils import convert_lines, amazon_multithreading_ocr
import fitz, json
from utils.infer_utils import NpEncoder

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

class bert_inferencing_sentence():

    def __init__(self, device, logger):

        self.opts = parse_args()
        self.device = device
        self.logger = logger
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.opts.bert_config, do_lower_case=True)
        
        self.model = BERT_Model(
            bert_config=self.opts.bert_config,
            n_class=self.opts.n_class
        )
        self.model.load_state_dict(torch.load("latest_weights/bert_lemm_sent_150/best_model.pt", map_location=self.device)['model_dict'])
        self.model.to(device)
        self.logger.info("Bert model loading is done")

    def is_pdf_unscanned(self, pdf_path, abs_pth):
        text_data = []
        text_pages = 0
        text = ""

        with fitz.open(pdf_path) as doc:

            total_pages = len(doc)
            for i in range(total_pages):
                page = doc[i]
                text = page.get_text(sort=True)
                raw_text = " ".join(text.splitlines()).replace("–", "-")
                text_data.append(raw_text)
                if text:
                    text_pages += 1

            if (text_pages / total_pages) * 100 >= 90:  # here will check if pdf have more than 90% text data
                
                self.logger.info("Processing unscanned PDF for bert Model")
                del raw_text, doc
                return text_data
            else:

                self.logger.info("Processing scanned PDF for bert Model")
                test_df = amazon_multithreading_ocr(pdf_path, abs_pth) # this process can take several time
                test_df['lemm_sent'] = test_df['Texts'].astype(str) 
                return test_df['lemm_sent']

    def process(self, pdf_path, abs_pth):
        results, conf_l = [], []

        text_data = self.is_pdf_unscanned(pdf_path, abs_pth)

        X_test = convert_lines(text_data, self.opts.max_length, self.bert_tokenizer)
        test_data = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))
        test_loader = torch.utils.data.DataLoader(test_data, 
                        batch_size = self.opts.batch_size, shuffle=False)

        for _, (x_batch,) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                tic = time.time()
                
                pred = self.model(x_batch.to(self.device), 
                            mask=(x_batch > 0).to(self.device), 
                            token_type_ids=None)
                
                pred_prob = torch.softmax(pred, dim=1).cpu()
                probs = torch.nn.functional.softmax(pred, dim=1)
                conf, _ = torch.max(probs, 1)
                conf_l.append(list(conf.tolist()))
                results.append(list(pred_prob.argmax(-1).tolist()))
                torch.cuda.empty_cache()
                self.logger.info("Time taken by Bert Model is {}".format(time.time() -tic))

        conf_l = list(np.concatenate(conf_l))
        results = list(np.concatenate(results))
        predicted = [*map(self.opts.id2label.get, results)] 
        predicted = [ f'{x} {y:.3f}' for x,y in zip(predicted, conf_l) ]
        predicted = {int(i+1): v for i, v in enumerate(predicted)}
        
        output = {"BERT": predicted}
        output = json.dumps(output, cls=NpEncoder)

        return  output, results, text_data
    

