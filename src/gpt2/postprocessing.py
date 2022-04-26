import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from bleurt import score
from rouge_score import rouge_scorer
import argparse
import os
parser = argparse.ArgumentParser()

parser.add_argument("--generated_path", type=str, default="./generation_out")
args = parser.parse_args()



r_scorer = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)
import re
def find_str(tag,sent):
    try:
        str_found = re.findall("%s(.*?)%s"%(tag,tag), sent)[0].strip()
    except:
        str_found=''
    return str_found

def entity_agreement(true_exp, generated_exp):

    true_search = re.search('<re>(.+?)<er>',true_exp)
    try:
        regulator_true = true_search.group(1).strip()
    except:

        regulator_true = ''

    true_search = re.search('<el>(.+?)<le>',true_exp)
    try:
        element_true = true_search.group(1).strip()
    except:

        element_true = ''

    grd_search = re.search('<re>(.+?)<er>',generated_exp)
    try:
        regulator_pred = grd_search.group(1).strip()
    except:
        # print('GE',generated_exp)
        regulator_pred = ''

    grd_search = re.search('<el>(.+?)<le>',generated_exp)
    try:
        element_pred = grd_search.group(1).strip()
    except:
        # print('GE',generated_exp)
        element_pred = ''
    # exit()
    return [regulator_true,regulator_pred, element_true,element_pred]

path = args.generated_path

generated_file_beam = open(os.path.join(path ,"generated_schemas_beam.txt"))
generated_file_greedy = open(os.path.join(path ,"generated_schemas_greedy.txt"))

used_sentences = open(os.path.join(path ,"generation_used_sents.txt"))
output_file = open(os.path.join(path ,"all_generated_filterlong.txt"),'w')




generated_lines_beam = generated_file_beam.readlines()
generated_lines_greedy = generated_file_greedy.readlines()

used_sents_lines = used_sentences.readlines()
df = pd.DataFrame([],columns=['Sup_Sen','True_Exp','True_lbl','True_reg','True_ele',\
    'GD_Exp','GD_lbl','GD_reg','GD_ele','GD_scr',\
        'BM_Exp','BM_lbl','BM_reg','BM_ele','BM_scr',\
            'SM_Exp','SM_lbl','SM_reg','SM_ele','SM_scr'])
true_labels = []
grd_labels = []
bm_labels = []
sm_labels=[]
regs_grd_binary = []
regs_bm_binary = []
regs_sm_binary = []
ele_grd_binary = []
ele_bm_binary = []
ele_sm_binary = []

bleurt_checkpoint = "./bleurt/BLEURT-20" 

grd_scores = []
bm_scores = []
sm_scores = []
true_exps = []
bm_exps = []
grd_exps = []
sm_exps = []
scores_l = []
scores_1 = []
scores_2 = []
scorer = score.BleurtScorer(bleurt_checkpoint)
for i in range(len(generated_lines_beam)):
    generated_line_beam = generated_lines_beam[i]
    generated_line_greedy = generated_lines_greedy[i]

    try:
        use_sent_line = used_sents_lines[i].split('<exp>')[0]
    except Exception as e:
        print(e)
        continue

    if use_sent_line.strip() == '':

        continue

    true_explan = used_sents_lines[i].split('<exp>')[1].strip()

    if true_explan.strip().find('<re>') < 0 or true_explan.strip().find('<el>')<0 or true_explan.strip().find('++++')>0:

        continue
    new_row = {}

    grd_line = generated_line_greedy[generated_line_greedy.find('<exp>'):].replace('<PAD>','')

    bm_line = generated_line_beam[generated_line_beam.find('<exp>'):].replace('<PAD>','')

    #### regulator and element ####
    [reg_tru,reg_grd,ele_tru,ele_grd]=entity_agreement(true_explan,grd_line)
    [_,reg_bm,_,ele_bm]=entity_agreement(true_explan,bm_line)
 
    reg_tru = reg_tru.lower()
    reg_grd = reg_grd.lower()
    ele_tru = ele_tru.lower()
    ele_grd = ele_grd.lower()
    reg_bm = reg_bm.lower()
    ele_bm = ele_bm.lower()

    new_row['True_reg'] = reg_tru
    new_row['True_ele'] = ele_tru
    new_row['GD_reg'] = reg_grd
    new_row['GD_ele'] =ele_grd
    new_row['BM_reg'] =reg_bm
    new_row['BM_ele'] =ele_bm

    if reg_tru == ele_tru or reg_tru=='' or ele_tru=='':
        continue
    regs_grd_binary.append(reg_tru==reg_grd)
    regs_bm_binary.append(reg_tru==reg_bm)


    ele_grd_binary.append(ele_tru==ele_grd)
    ele_bm_binary.append(ele_tru==ele_bm)



    output_file.write('----- Processing Example %d ----\n'%i)
    output_file.write('----- Supporting Senteces ----\n')
    output_file.write(use_sent_line)
    new_row['Sup_Sen']=use_sent_line
    output_file.write('----- True Explanation  and Label -----\n')
    output_file.write(true_explan)
    true_end_ind = true_explan.rfind('.')
    new_row['True_Exp'] = true_explan[:true_end_ind+1].strip()#[:-15]

    output_file.write('----- Greedy Generated Explanation and Label -----\n')

    output_file.write(grd_line)
    grd_end_ind = grd_line.rfind('.')
    new_row['GD_Exp'] = grd_line[:grd_end_ind+1]#[:-9]#[:-15]

    output_file.write('----- Beam Generated Explanation and Label -----\n')
    output_file.write(bm_line)   
    bm_end_ind = bm_line.rfind('.') 
    new_row['BM_Exp'] = bm_line[:bm_end_ind+1]#[:-9]

    


    # ###### BLEURT SCORE ######
    try:

        true_exp_bleurt = true_explan.strip()
        grd_exp_bleurt = grd_line[5:].strip()
        bm_exp_bleurt = bm_line[5:].strip()



        true_exp_bleurt = true_explan.strip()
        grd_exp_bleurt = grd_line[5:].strip()
        bm_exp_bleurt = bm_line[5:].strip()


        true_exps.append(true_exp_bleurt)
        bm_exps.append(bm_exp_bleurt)
        grd_exps.append(grd_exp_bleurt)

    except Exception as e: 

        pass

    hyp =bm_line[5:].strip()
    target = true_explan.strip()

    scores = r_scorer.score(hyp,target)
    scores_l.append(scores['rougeL'].fmeasure)
    scores_1.append(scores['rouge1'].fmeasure)
    scores_2.append(scores['rouge2'].fmeasure)
    df = df.append(new_row,ignore_index=True)
    

scores_grd = scorer.score(true_exps, grd_exps)
# scores_sm = scorer.score(true_exps, sm_exps)
scores_bm = scorer.score(true_exps, bm_exps)
df['GD_scr'] = scores_grd
df['BM_scr'] = scores_bm

with open(os.path.join(path,'test_2k_filterlong.csv'),'w') as f:
        df.to_csv(f,index=False)
output_file.close()
with open(os.path.join(path,'results.txt')) as f:
    print('Accuracy for greedy regulator prediction',file=f)
    print(sum(regs_grd_binary)/len(regs_grd_binary),file=f)
    print('Accuracy for greedy element prediction',file=f)
    print(sum(ele_grd_binary)/len(ele_grd_binary),file=f)
    print('Accuracy for beam regulator prediction',file=f)
    print(sum(regs_bm_binary)/len(regs_bm_binary),file=f)
    print('Accuracy for beam element prediction',file=f)
    print(sum(ele_bm_binary)/len(ele_bm_binary),file=f)




    
    print(sum(scores_grd)/len(scores_grd),file=f)
    print(sum(scores_bm)/len(scores_bm),file=f)


    print(sum(scores_l)/len(scores_l),file=f)
    print(sum(scores_1)/len(scores_1),file=f)
    print(sum(scores_2)/len(scores_2),file=f)
