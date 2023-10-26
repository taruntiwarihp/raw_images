import torch
import shutil
from glob import glob
import gradio as gr
import uuid, os, json

from utils.config import parse_args
from utils.infer_utils import NpEncoder
from utils.logging import create_logger
import gradio as gr

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")



opts = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger, _ = create_logger(opts, phase = 'inference')

from bert_inference_sentence import bert_inferencing_sentence
bert_sentence = bert_inferencing_sentence(device, logger)


def zip_to_json(file_obj):

    abs_pth = "input_pdf/{}".format(uuid.uuid4().hex)
    os.makedirs(abs_pth, exist_ok=True)

    shutil.unpack_archive(file_obj.name, abs_pth)

    try:
        pdf_path = "{}".format(glob("{}/*.PDF".format(abs_pth))[0])
    except:
        pdf_path = "{}".format(glob("{}/*.pdf".format(abs_pth))[0])


    final_out, _, _ = bert_sentence.process(pdf_path, abs_pth)

    return final_out

inputs = gr.inputs.File(label="Please Upload Zip File", type="file")
outputs = gr.outputs.Textbox(label="json")
demo = gr.Interface(zip_to_json, title = "Policy Page Classification", inputs = inputs, outputs = 'json')

if __name__ == "__main__":
    demo.queue().launch(share=True, server_name='0.0.0.0', max_threads=20,
                        show_error=True, #enable_queue=True,
                        debug=True,)




    
