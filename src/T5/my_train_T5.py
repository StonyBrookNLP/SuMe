'''
Creating a new GPT model for training the dataset
'''
import argparse
import torch
from transformers import *
import logging
import random
import numpy as np
from tqdm import tqdm, trange
import os
from accelerate import Accelerator
import nltk
import pickle

# nltk.download('punkt')
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="razent/SciFive-base-Pubmed_PMC", help="pretrained model name") # razent/SciFive-large-Pubmed_PMC
parser.add_argument("--do_train", action="store_true", default="True", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", default="False", help="Whether to run eval on the dev set.")
parser.add_argument("--output_dir",type=str, default="./model_out",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument("--encoding_dir", type=str, default="./encoding")

parser.add_argument("--train_dataset", type=str, default="./data/train_v8.txt")
parser.add_argument("--do_freez",type=bool, default=False)
parser.add_argument("--num_freez_layers",type=int, default=5)


parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_train_epochs", type=int, default=20)
parser.add_argument("--do_lower", type=bool, default=False)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", type=int, default=1)
parser.add_argument("--input_index", type=int, default=None,help="To just use this number of training set")

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
parser.add_argument("--learning_rate", type=float, default=6.25e-5)
parser.add_argument("--sample_len", type=int, default=512)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--lr_schedule", type=str, default="warmup_linear")
parser.add_argument("--weight_decay", type=float, default=0.01)

parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
args = parser.parse_args()

print('model:', args.model_name)
print('outputdir:', args.output_dir)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
## load a pre-trained GPT2LM model
model_class = T5ForConditionalGeneration
tokenizer_class = T5Tokenizer

pretrained_weights = args.model_name
model = model_class.from_pretrained(pretrained_weights)

tokenizer = tokenizer_class.from_pretrained(pretrained_weights,pad_token='<PAD>',padding_side='right',do_lower_case=args.do_lower,max_length=args.sample_len)





special_tokens = [ '<exp>',  # 32101
                    '<re>',  # 32102
                    '<er>',  # 32103
                    '<el>',  # 32104
                    '<le>',  # 32105
                    '<end>'
                    ]


special_tokens_dict = {'additional_special_tokens': special_tokens}

num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

CLS_ID = tokenizer(['<CLS>'],add_special_tokens=False)['input_ids'][0]
SEP_ID = tokenizer(['<SEP>'],add_special_tokens=False)['input_ids'][0]
EOS_ID = tokenizer([tokenizer.eos_token],add_special_tokens=False)['input_ids'][0]

RE_ID = tokenizer(['<re>'],add_special_tokens=False)['input_ids'][0]

model.resize_token_embeddings(len(tokenizer))


device = 'cuda'
model.to(device)
configuration = model.config

def prepare_data(filename,input_len):
    print('preparing dataset')
    out_saved = filename.split('/')[-1]
    all_samples = (open(filename)).readlines()
    print('total data length: %s'%len(all_samples))
    n_batch = len(all_samples)
    ind = 0
    current_input = []
    padded_inputs = []
    padded_targets = []
    labels_output = []
    input_masks = []
    supporting_sum = 0
    shod = 0
    for idd, sent in enumerate(all_samples):
        if idd%10000 == 0:
            print('processed: %d samples'%idd)
        if args.do_lower:
            sent = sent.lower()
        try:
            current_sent = sent.split('<exp>')[0]
            exp_sent = sent.split('<exp>')[1]
            
            current_pieces = nltk.sent_tokenize(current_sent)
            if len(current_pieces) <1:
                continue
        except:
            continue
        sample = tokenizer.batch_encode_plus(current_pieces, add_special_tokens=False)#,pad_to_max_length=True,padding='max_length',\
        current_input = sample['input_ids']
        output = tokenizer.encode_plus('<exp> '+ exp_sent + ' <end> ', add_special_tokens=True,padding='max_length',max_length=input_len,pad_to_max_length=True)
        actual_len = sum(output['attention_mask'])+1 # add one for eos token
        if  actual_len > input_len-2:
            print('skipped this sample because explanation is too large')
            continue

        used_input_temp = []
        idd = 0
        while True :
            used_input_temp += current_input[idd]
            if len(used_input_temp) >= input_len-actual_len -5:
                break
            used_input = used_input_temp.copy()
            idd += 1
            if idd == len(current_input):
                shod += 1
                break
        if len(used_input_temp) <1:
            print('no supporting added because mechanism is too long')
            continue

        padded_input = [21603,    10]+used_input+ EOS_ID +[0]*(input_len-len(used_input)-3)

        input_attention_mask = [1] * (3+len(used_input)) + [0]*(input_len-len(used_input)-3)

        ## mask out all input tokens and <exp> sign and just compute loss for the output token
        label_output = output['input_ids'][:actual_len]+ [-100]* (input_len - actual_len) 
        padded_target= output['input_ids'][:actual_len]+ [0]* (input_len - actual_len)
        assert len(label_output) == len(padded_input)
        assert len(padded_input) == len(input_attention_mask)
        padded_targets.append(padded_target)
        padded_inputs.append(padded_input)
        labels_output.append(label_output)
        input_masks.append(input_attention_mask)


    
    with open(os.path.join(args.encoding_dir,out_saved+'_padded_targets.in.pk'),'wb') as f:
        pickle.dump(padded_inputs,f, protocol=4)
    with open(os.path.join(args.encoding_dir,out_saved+'_input_masks.out.pk'),'wb') as f:
        pickle.dump(input_masks,f, protocol=4)
    with open(os.path.join(args.encoding_dir,out_saved+'_padded_targets.out.pk'),'wb') as f:
        pickle.dump(padded_targets,f, protocol=4)
    with open(os.path.join(args.encoding_dir,out_saved+'_labels_output.out.pk'),'wb') as f:
        pickle.dump(labels_output,f, protocol=4)

    print('completed sentences: %d'%shod)
    return padded_inputs, input_masks,padded_targets,labels_output

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

## freeze some layers of being trained
if args.do_freez:
    modules = [model.encoder.embed_tokens, *model.encoder.block[:args.num_freez_layers]] #Replace 5 by what you want
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False

logger.info("Encoding dataset...")
try:
    # a = b
    print('loading encoded data')
    out_saved = args.train_dataset.split('/')[-1]
    with open(os.path.join(args.encoding_dir,out_saved+'_padded_targets.in.pk'),'rb') as f:
        train_samples = pickle.load(f)
    with open(os.path.join(args.encoding_dir,out_saved+'_input_masks.out.pk'),'rb') as f:
        train_attention_mask = pickle.load(f)
    with open(os.path.join(args.encoding_dir,out_saved+'_padded_targets.out.pk'),'rb') as f:
        train_targets = pickle.load(f)
    with open(os.path.join(args.encoding_dir,out_saved+'_labels_output.out.pk'),'rb') as f:
        label_train_samples = pickle.load(f)
    print('encoded data loaded successfully')
except Exception as e:
    print('error in loading data:',e)
    print('data encoding started')
    train_samples,train_attention_mask,train_targets, label_train_samples= prepare_data(args.train_dataset,args.sample_len)#,args.output_len)#model.config.n_positions)
if args.input_index:
    train_samples = train_samples[:args.input_index]
    train_attention_mask = train_attention_mask[:args.input_index]
    train_targets = train_targets[:args.input_index]
    label_train_samples = label_train_samples[:args.input_index]
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
        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            import gc

            gc.collect()
            tr_loss = 0
            nb_tr_steps = 0
            print('train samples length:' , len(train_samples))
            print('train batch size:',args.train_batch_size)
            batch_idxs = np.random.permutation(len(train_samples)//args.train_batch_size)
            line_tqdm = tqdm(batch_idxs, dynamic_ncols=True)
            
            for batch_idx in line_tqdm:
                batch = train_samples[batch_idx*args.train_batch_size:min((batch_idx+1)*args.train_batch_size, len(train_samples))]
                labels = label_train_samples[batch_idx*args.train_batch_size:min((batch_idx+1)*args.train_batch_size, len(train_samples))]
                targets = train_targets[batch_idx*args.train_batch_size:min((batch_idx+1)*args.train_batch_size, len(train_samples))]
                input_mask = train_attention_mask[batch_idx*args.train_batch_size:min((batch_idx+1)*args.train_batch_size, len(train_samples))]

                batch = torch.tensor(batch)
                label_batch = torch.tensor(labels)
                input_ids = batch.to(device)
                label_batch = label_batch.to(device)
                targets_batch =  torch.tensor(targets).to(device)
                input_mask_batch = torch.tensor(input_mask).to(device)

                losses = model(input_ids = input_ids, labels=label_batch,attention_mask=input_mask_batch)
                
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

                
                ## save every 10000 steps
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
                    
                
