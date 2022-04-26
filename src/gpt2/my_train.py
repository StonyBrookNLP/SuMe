'''
Creating a new GPT model for training the dataset
'''
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import *
import logging
import random
import numpy as np
from tqdm import tqdm, trange
import os
import pickle 
import nltk


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()


parser.add_argument("--model_name", type=str, default="healx/gpt-2-pubmed-medium", help="pretrained model name")
parser.add_argument("--do_train", action="store_true", default="True", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", default="False", help="Whether to run eval on the dev set.")
parser.add_argument("--output_dir",type=str, default="./model_out",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument("--train_dataset", type=str, default="./data/train_v8.txt") 
parser.add_argument("--eval_dataset", type=str, default="./data/test_v8.txt")

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_train_epochs", type=int, default=10)
parser.add_argument("--do_lower", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", type=int, default=1)
parser.add_argument("--device", type=str, default='cuda')

parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training \
                    steps to perform. Override num_train_epochs.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=2,
    help="Number of updates steps to accumulate before\
                    performing a backward/update pass.",
)
parser.add_argument("--learning_rate", type=float, default=6.25e-5)#1.0e-4)#
parser.add_argument("--sample_len", type=int, default=256)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--lm_coef", type=float, default=0.9)
parser.add_argument("--n_valid", type=int, default=374)

parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
## load a pre-trained GPT2LM model
model_class = GPT2LMHeadModel
tokenizer_class = GPT2Tokenizer
pretrained_weights =args.model_name 
special_tokens = [ '<exp>',  # 50258
                    '<re>',  # 50259
                    '<er>',  # 50260
                    '<el>',  # 50261
                    '<le>',  # 50262
                    '<end>',#, # 50263
                    ]
all_tokens = {'eos_token':'<end>','cls_token': '<CLS>','sep_token':'<SEP>','pad_token':'<PAD>'}
tokenizer = tokenizer_class.from_pretrained(pretrained_weights,pad_token='<PAD>',padding_side='right',do_lower_case=args.do_lower,max_length=args.sample_len)
tokenizer.add_tokens(special_tokens)
tokenizer.add_special_tokens(all_tokens)

CLS_ID = tokenizer(['<CLS>'],add_special_tokens=False)['input_ids'][0]
SEP_ID = tokenizer(['<SEP>'],add_special_tokens=False)['input_ids'][0]
DOT_ID = tokenizer(['.'],add_special_tokens=False)['input_ids'][0]
re_ID = tokenizer(['<re>'],add_special_tokens=False)['input_ids'][0]
el_ID = tokenizer(['<el>'],add_special_tokens=False)['input_ids'][0]
pad_ID = tokenizer(['<PAD>'],add_special_tokens=False)['input_ids'][0]
eos_ID = tokenizer(['<end>'],add_special_tokens=False)['input_ids'][0]

tokenizer.save_pretrained(args.output_dir)
model = model_class.from_pretrained(pretrained_weights)
model.resize_token_embeddings(len(tokenizer))
from accelerate import Accelerator
accelerator = Accelerator()
if args.device =='cuda':
    device = 'cuda'
else:
    device = accelerator.device

model.to(device)
configuration = model.config

def prepare_data(filename,input_len):
    print('preparing dataset')
    all_samples = (open(filename)).readlines()
    out_saved = filename.split('/')[-1]
    n_batch = len(all_samples)
    ind = 0
    current_input = []
    sample_output = []
    labels_output = []
    
    nashod = 0
    shod = 0
    ind = 0
    print('all samples length:',len(all_samples))
    for sent in all_samples:
        if ind% 10000 ==0:
            print('processes %d items'%ind)
        if args.do_lower:
            sent = sent.lower()
        current_sent = sent.split('<exp>')[0]
        try:
            exp_sent = sent.split('<exp>')[1]
            
            current_pieces = nltk.sent_tokenize(current_sent)
            if len(current_pieces) <1:
                continue
        except Exception as e:
            print(e)
            continue
        sample = tokenizer.batch_encode_plus(current_pieces, add_special_tokens=False)#,pad_to_max_length=True,padding='max_length',\
        current_input = sample['input_ids']
        output = tokenizer.encode_plus('<exp> '+exp_sent+ ' <end>', add_special_tokens=True,padding='max_length',max_length=input_len,pad_to_max_length=True)
        actual_len = sum(output['attention_mask'])
        if actual_len > input_len-2:
            nashod += 1
            continue
        used_input_temp = []
        idd = 0
        while True :
            used_input_temp += current_input[idd]
            if len(used_input_temp) >= input_len-actual_len-5:
                break
            used_input = used_input_temp.copy()
            idd += 1
            if idd == len(current_input):
                shod += 1
                break

        padded_input = used_input + output['input_ids'][:actual_len] +eos_ID+ pad_ID*(input_len-len(used_input)-actual_len-1) 
        label_output = [-100] * (len(used_input)) + output['input_ids'][:actual_len]+eos_ID+[-100]*(input_len-len(used_input)-actual_len-1)
        assert len(label_output) == len(padded_input)
        sample_output.append(padded_input)
        labels_output.append(label_output)
        ind += 1


    with open(out_saved+'_v7_512.in.pk','wb') as f:
        pickle.dump(sample_output,f)
    with open(out_saved+'_v7_512.out.pk','wb') as f:
        pickle.dump(labels_output,f)

    return sample_output, labels_output

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

logger.info("Encoding dataset...")
try:

    out_saved = args.train_dataset.split('/')[-1]

    with open(out_saved+'_v7_512.in.pk','rb') as f:
        train_samples = pickle.load(f)
    with open(out_saved+'_v7_512.out.pk','rb') as f:
        label_train_samples = pickle.load(f)
    print('input files loaded')
except Exception as e:
    print('file can not be loaded: %s'% str(e))
    train_samples, label_train_samples= prepare_data(args.train_dataset,args.sample_len)
print('train samples length:' , len(train_samples))

logger.info("Dataset Encoded...")
# Prepare optimizer
if args.do_train:
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_samples) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_samples) // args.gradient_accumulation_steps * args.num_train_epochs
    print('train totall steps:',t_total)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if device != 'cuda':
        model, optimizer, train_samples = accelerator.prepare(model, optimizer, train_samples)

if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        print('train epochs:' , int(args.num_train_epochs))
        print('train samples size: ',len(train_samples))
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            import gc

            gc.collect()

            torch.cuda.empty_cache()
            tr_loss = 0
            nb_tr_steps = 0

            batch_idxs = np.random.permutation(len(train_samples)//args.train_batch_size)

            line_tqdm = tqdm(batch_idxs, dynamic_ncols=True)
            
            for batch_idx in line_tqdm:
                batch = train_samples[batch_idx*args.train_batch_size:min((batch_idx+1)*args.train_batch_size, len(train_samples))]
                labels = label_train_samples[batch_idx*args.train_batch_size:min((batch_idx+1)*args.train_batch_size, len(train_samples))]
                
                batch = torch.tensor(batch)
                label_batch = torch.tensor(labels)
                input_ids = batch.to(device)
                label_batch = label_batch.to(device)

                losses = model(input_ids,labels=label_batch)

                #losses: a tuple with 3 items, 
                # the first one is the loss value, 
                # the second one is the output of dixr batch * seq length * vocab size, 
                # the third one is a tuple with item of size [2,8,12,256,64] I don't know how many items it has and what it is :D
                loss = losses[0]
                if device == 'cuda':
                    loss.backward()
                else:
                    accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = (
                    loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
                )
                nb_tr_steps += 1
                
                
                line_tqdm.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, scheduler.get_lr()[0])

            if True:
                    model_to_save = model.module if hasattr(model, "module") else model  # Only save the model itself                
                    ep_path = os.path.join(args.output_dir,str(ep))
                    output_model_file = os.path.join(args.output_dir,str(ep), WEIGHTS_NAME)
                    output_config_file = os.path.join(args.output_dir,str(ep), CONFIG_NAME)
                    if not os.path.exists(ep_path):
                        os.makedirs(ep_path)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_pretrained(ep_path)

                    
                
                
# Save a trained model
if args.do_train:
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, "module") else model  # Only save the model itself

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    model = model_class.from_pretrained(args.output_dir)
    model.to(device)
    

    
