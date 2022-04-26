'''
Generate schemas using the seed events
'''
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from transformers import *
from tqdm import tqdm, trange
import logging
import os
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",type=str, default="./model_out")
parser.add_argument("--seeds_file",type=str, default="./data/test_v8.txt")
parser.add_argument("--generation_path",type=str, default="./generation_out/v6")
parser.add_argument("--generation_type", type=str, default="beam",help="it can be greedy, beam, sampling")
parser.add_argument("--generation_type2", type=str, default="greedy",help="it can be greedy, beam, sampling")
parser.add_argument("--generation_type3", type=str, default="samppling",help="it can be greedy, beam, sampling")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_length", type=int, default=256)
parser.add_argument("--generation_length", type=int, default=512)
parser.add_argument("--do_lower", type=bool, default=True)
parser.add_argument("--batch_size", type=int, default=1)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

## load a pre-trained GPT2LM model_old
model_class = GPT2LMHeadModel
tokenizer_class = GPT2Tokenizer
# pretrained_weights = 'gpt2'

# Load a trained model_old and vocabulary that you have fine-tuned
tokenizer = tokenizer_class.from_pretrained(args.model_dir)
print(len(tokenizer))
# exit()
model = model_class.from_pretrained(args.model_dir)

model.to(device)
# model.parallelize()
END_TOKEN = tokenizer.encode(['<end>'])[0]
EXP_TOKEN = tokenizer.encode(['<exp>'])[0]
CLS_TOKEN = tokenizer.encode(['<CLS>'])[0]
SEP_TOKEN = tokenizer.encode(['<SEP>'])[0]
PAD_TOKEN = tokenizer.encode(['<PAD>'])[0]
DOT_ID = tokenizer(['.'],add_special_tokens=False)['input_ids'][0]
re_ID = tokenizer(['<re>'],add_special_tokens=False)['input_ids'][0]
el_ID = tokenizer(['<el>'],add_special_tokens=False)['input_ids'][0]
# print('end token is: ', END_TOKEN)
# print('exp token is: ', EXP_TOKEN)
# print(tokenizer.sep_token)
# exit()
def prepare_data(filename,input_len,f):#,output_len):
    print('preparing dataset')
    all_samples =  (open(filename)).readlines()#[:300]
    n_batch = len(all_samples)
    ind = 0
    current_input = []
    sample_output = []
    supporting_sum = 0
    skipped = 0
    
    shod = 0
    for sent in all_samples:
        if args.do_lower:
            sent = sent.lower()
        current_sent = sent.split('<exp>')[0]

        try:
            exp_sent = sent.split('<exp>')[1]
            import nltk
            current_pieces = nltk.sent_tokenize(current_sent)
            if len(current_pieces) <1:
                continue
        except Exception as e:
            print(e)
            continue

        # sample = tokenizer.encode_plus(current_sent, add_special_tokens=True)#,pad_to_max_length=True,padding='max_length',\
        sample = tokenizer.batch_encode_plus(current_pieces, add_special_tokens=False)
        current_input = sample['input_ids']
    

        used_input_temp = []
        idd = 0
        while True :
            used_input_temp += current_input[idd]
            if len(used_input_temp) >= input_len-5:
                break
            used_input = used_input_temp.copy()
            idd += 1
            if idd == len(current_input):
                shod += 1
                break
        padded_input = used_input+ [PAD_TOKEN]*(input_len-len(used_input)-1) + [EXP_TOKEN] + [tokenizer.eos_token_id]

        
        sample_output.append(padded_input)
       
        fi.write(tokenizer.decode(used_input, skip_special_tokens=True))
        fi.write(' <exp> '+exp_sent)
        

    
    print('skipped:' , skipped)
    return sample_output 


## read samples from the seeds file

fi =open(os.path.join(args.generation_path,'generation_used_sents.txt'), 'w', encoding='utf-8')
samples = prepare_data(args.seeds_file,args.output_length,fi)
print('remaining:' , len(samples))
import pickle
print('data encoded')
## go over all the samples and generate the output with respect to the seed event.
greedy_file =  open(os.path.join(args.generation_path,'generated_schemas_beam.txt'), 'w') 
beam_file = open(os.path.join(args.generation_path,'generated_schemas_greedy.txt'), 'w') 
sampling_file = open(os.path.join(args.generation_path,'generated_schemas_sampling.txt'), 'w') 
batch_idxs = list(range(len(samples)//args.batch_size))
print('total samples length: %s'%len(samples))
print('batch to process: %s' %batch_idxs)
line_tqdm = tqdm(batch_idxs, dynamic_ncols=True) 
for batch_idx in line_tqdm:
        batch = samples[batch_idx*args.batch_size:min((batch_idx+1)*args.batch_size, len(samples))]
        batch = torch.tensor(batch)
        input_ids = batch.to(device)

        
        if args.generation_type == "beam":
            beam_output = model.generate(
                    input_ids, 
                    max_length=args.output_length+ args.generation_length, 
                    num_beams=5, 
                    # no_repeat_ngram_size=4, ##to remove the repetitive n-grams
                    # early_stopping=True,
                    eos_token_id=END_TOKEN,
                    pad_token_id=PAD_TOKEN
                )
            

            for outp in beam_output:
                tokenized = tokenizer.decode(outp, skip_special_tokens=True)
                beam_file.write(tokenized.strip())
                
                beam_file.write('\n')
            
        
        if args.generation_type2 == "greedy":
            ## for greedy search 
            
            greedy_output = model.generate(
                input_ids, 
                max_length=args.output_length+args.generation_length,
                # length_penalty=2.0, ## exponential penalty on length
                no_repeat_ngram_size=4,
                # early_stopping=True,
                eos_token_id=END_TOKEN,
                pad_token_id=PAD_TOKEN
                )

            for outp in greedy_output:
                tokenized = tokenizer.decode(outp, skip_special_tokens=True)
                greedy_file.write(tokenized.strip())
                
                greedy_file.write('\n')
        if args.generation_type3 == "sampling":
    
            # activate sampling and deactivate top_k by setting top_k sampling to 0
            sample_output = model.generate(
                input_ids, 
                do_sample=True, 
                max_length=args.output_length+args.generation_length, 
                top_k=0,
                eos_token_id=END_TOKEN,
                temperature=0.7, ##use temperature to decrease the sensitivity to low probability candidates
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                pad_token_id=PAD_TOKEN
            )
            for outp in sample_output:
                sampling_file.write(tokenizer.decode(outp, skip_special_tokens=True))
                
                sampling_file.write('>>>>>>\n<<<<<<')


fi.close()
            
            
