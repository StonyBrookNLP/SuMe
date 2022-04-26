'''
Generate schemas using the seed events
'''
import random
import numpy as np
import argparse
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import  *
from tqdm import tqdm, trange
import logging
import time
import os
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("--model_dir",type=str, default="./model_out")
parser.add_argument("--seeds_file",type=str, default="./data/test_v8.txt")
parser.add_argument("--generation_dir",type=str, default="./generation_out")

parser.add_argument("--generation_type", type=str, default="beam",help="it can be greedy, beam, sampling")
parser.add_argument("--generation_type2", type=str, default="greedy",help="it can be greedy, beam, sampling")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_length", type=int, default=512)
parser.add_argument("--do_lower", type=bool, default=False)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--generation_length", type=int, default=128)
parser.add_argument("--number_generation", type=int, default=1)
parser.add_argument("--num_beam_groups", type=int, default=5)
parser.add_argument("--num_beams", type=int, default=5)
parser.add_argument("--diversity_penalty", type=float, default=0.5)
parser.add_argument("--temperature", type=float, default=None)#1.5)
parser.add_argument("--topk", type=float, default=None)#20)
parser.add_argument("--topp", type=float, default=0)#0.9)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

## load a pre-trained GPT2LM model_old
model_class = T5ForConditionalGeneration
tokenizer_class = T5Tokenizer


# Load a trained model_old and vocabulary that you have fine-tuned
tokenizer = tokenizer_class.from_pretrained(args.model_dir)
model = model_class.from_pretrained(args.model_dir)

model.to(device)
END_TOKEN = tokenizer.encode(['<end>'],add_special_tokens=False)[0]
EXP_TOKEN = tokenizer.encode(['<exp>'],add_special_tokens=False)[0]
CLS_TOKEN = tokenizer.encode(['<CLS>'],add_special_tokens=False)[0]
SEP_TOKEN = tokenizer.encode(['<SEP>'],add_special_tokens=False)[0]
EOS_ID = tokenizer([tokenizer.eos_token],add_special_tokens=False)['input_ids'][0]

DOT_ID = tokenizer.encode(['.'],add_special_tokens=False)
re_ID = tokenizer.encode(['<re>'],add_special_tokens=False)
el_ID = tokenizer.encode(['<el>'],add_special_tokens=False)


def prepare_data(filename,input_len,f):#,output_len):
    print('preparing dataset')
    all_samples =  (open(filename)).readlines()
    n_batch = len(all_samples)
    ind = 0
    current_input = []
    sample_output = []
    sample_mask = []
    supporting_sum = 0
    skipped = 0
    shod = 0
    nashod = 0

    for sent in all_samples:
        if args.do_lower:
            sent = sent.lower()
        try:
            current_sent=sent.strip()
            import nltk
            current_pieces = nltk.sent_tokenize(current_sent)
            if len(current_pieces) <1:
                continue
        except Exception as e:
            print(e)
            continue
        sample = tokenizer.batch_encode_plus(current_pieces, add_special_tokens=False)#,pad_to_max_length=True,padding='max_length',\
           
        current_input = sample['input_ids']
        used_input_temp = []
        used_input = ''
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
        if len(used_input) < 2:
            print('input sentence is long')
            continue
        padded_input = [21603,    10]+used_input+ EOS_ID+[0]*(input_len-len(used_input)-3)#+ EOS_ID#+[0]*(input_len-len(used_inputs)-3)

        input_attention_mask = [1] * (3+len(used_input)) + [0]*(input_len-len(used_input)-3) #+ [0]*(input_len-len(used_inputs)-3)
        assert len(padded_input) == len(input_attention_mask) == input_len
        
        sample_output.append(padded_input)
        
        sample_mask.append(input_attention_mask)
        f.write(tokenizer.decode(used_input, skip_special_tokens=True))

    
    print('completed:', shod)
    print('all succedded:' , len(sample_output))
    return sample_output,sample_mask


if not os.path.exists(args.generation_dir):
        os.makedirs(args.generation_dir)
with open(os.path.join(args.generation_dir,'generation_used_sents.txt'), 'w', encoding='utf-8') as fi:
    samples,mask = prepare_data(args.seeds_file,args.output_length,fi)
import pickle
pickle.dump(samples,open('t5_samples.pkl','wb'))
print('data encoded')

## go over all the samples and generate the output with respect to the seed event.
beam_file = open(os.path.join(args.generation_dir, 'generated_schemas_beam.txt') , 'w') 
greedy_file = open(os.path.join(args.generation_dir, 'generated_schemas_greedy.txt'), 'w')
import math
batch_idxs = np.array(range(math.ceil(len(samples)/args.batch_size)))

###generation parameters:

line_tqdm = tqdm(batch_idxs, dynamic_ncols=True)
for batch_idx in line_tqdm:
        
        sample_tokenized = samples[batch_idx*args.batch_size:min((batch_idx+1)*args.batch_size, len(samples))]
        attention_mask = mask[batch_idx*args.batch_size:min((batch_idx+1)*args.batch_size, len(samples))]
        input_ids = torch.tensor(sample_tokenized) 
        mask_ids = torch.tensor(attention_mask)
        
        input_ids = input_ids.to(device) 
        mask_ids =  mask_ids.to(device) 
        if args.generation_type == "beam":

            beam_output=model.generate(
                             input_ids=input_ids,
                             attention_mask=mask_ids,
                             decoder_start_token_id=tokenizer.bos_token_id,
                             min_length=128,
                             max_length=256,
                             num_beams=50,
                             no_repeat_ngram_size=3,
                             length_penalty=0.1,
                             early_stopping=1.5)
            decoded = tokenizer.batch_decode(beam_output, skip_special_tokens=False)

            for b in decoded:
                beam_file.write(b)
                beam_file.write('\n')
            exit()

        if args.generation_type2 == "greedy":
            ## for greedy search 
            
            greedy_output = model.generate(input_ids = input_ids, 
                                attention_mask= mask_ids,
                                max_length=args.output_length+args.generation_length,
                                eos_token_id=END_TOKEN,
                                early_stopping=True,
                                repetition_penalty=2.5,
                                temperature=0.7)
            decoded = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)
            for go in decoded:
                greedy_file.write(go)
                greedy_file.write('\n')

        if args.generation_type == "sampling":
            outputs = model.generate(input_ids=input_ids,\
                attention_mask=mask_ids,\
                num_beams=1,\
                max_length=512,\
                do_sample=True,\
                temperature=1.5,\
                top_k=20,\
                top_p=0.9,\
                num_return_sequences=args.number_generation)

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            # print(decoded)
            # exit()
            for go in decoded:
                greedy_file.write(go)
                greedy_file.write('\n')
                beam_file.write(go)
                beam_file.write('\n')
beam_file.close()
greedy_file.close()



            
