import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Document Classification')

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='latest_weights/bert_lemm_multi_150/best_model.pt') # 1st bert1 2nd New_bert(NEW)
    parser.add_argument('--id2label',
                        help='Index of the labels',
                        type=dict,
                        default={0: 'declaration', 1:'endorsement', 2:'others', 3:'acord'})
    parser.add_argument('--label2id',
                        help='labels of index',
                        type=dict,
                        default={'declaration': 0,'endorsement': 1,'others': 2, 'acord': 3})
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='logs')
    parser.add_argument('--bert_config',
                        help='Bert Model Config',
                        type=str,
                        default='bert-base-uncased',
                        choices=['bert-base-uncased', 'bert-large-uncased', 'distilbert-base-uncased']) # experiment
    parser.add_argument('--root',
                        help='Root Location',
                        type=str,
                        default='data/final_process_data.csv')
    parser.add_argument('--model_type',
                        help='Types of model',
                        type=str,
                        default='bert',
                        choices=['robert', 'bert', 'lstm']) 
    parser.add_argument('--n_class',
                        help='Total number of Labels',
                        type=int,
                        default=4) 
    parser.add_argument('--split_ratio',
                        help='train set valid set ratio',
                        type=float,
                        default=0.2)
    parser.add_argument('--max_length',
                        help='Max length in one description',
                        type=int,
                        default=512)
    parser.add_argument('--batch_size',
                        help='train set valid set Batch size',
                        type=int,
                        default=60)
    parser.add_argument('--finetune_epochs',
                        help='Total Epochs for Finetuning',
                        type=int,
                        default=80)
    parser.add_argument("--class_weights", 
                        type=int, 
                        default=0, 
                        choices=[0, 1])
    parser.add_argument('--learning_rate',
                        help='Learning Rate',
                        type=float,
                        default=1e-5) # 1e-4
    parser.add_argument('--lstm_hidden_layer',
                        help='LSTM Hidden Layer on RoBert Model',
                        type=int,
                        default=100)
    parser.add_argument('--weight_decay',
                        help='Weight decay if we apply some.',
                        type=float,
                        default=0.1)
    parser.add_argument('--warmup_ratio',
                        help='Linear warmup over warmup_ratio*total_steps.',
                        type=float,
                        default=0.1)
    parser.add_argument('--adam_epsilon',
                        help='Epsilon for Adam optimizer.',
                        type=float,
                        default=1e-8)                     

    args = parser.parse_args()

    return args