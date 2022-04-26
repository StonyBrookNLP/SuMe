import os
import pandas as pd


path='./generation_out_dev_v8'
results_csv = pd.DataFrame(columns=['mdoel_name','reged_pred','reger_pred','bleurt','rougL','roug1','roug2'])
data_dicts = []
for folder in os.listdir(path=path):
    for subfolder in os.listdir(os.path.join(path,folder)):
        try:
            for each_files in os.listdir(os.path.join(path,folder,subfolder)):
                if not os.path.exists(os.path.join(path,folder,subfolder,'results.txt')):
                    print('no results found here:',os.path.join(path,folder,subfolder))
                    continue
                if 'result' in each_files:

                    with open(os.path.join(path,folder,subfolder,each_files)) as f:
                        results_lines = f.readlines()
                        beam_regulator_prediction_acc =results_lines[5]
                        beam_element_prediction_acc =results_lines[7]
                        weighted_F1 = results_lines[26].strip().split()[4]
                        Bleurt=float(results_lines[31])*100
                        Rouge_l=float(results_lines[33] )*100
                        Rouge_1=float(results_lines[35] ) *100
                        Rouge_2=float(results_lines[37] ) *100
                        avg = (Bleurt+Rouge_l)/2
                        model_name = '%s_%s'%(folder,subfolder)
                        row = {'model_name':model_name,'reged_pred':beam_regulator_prediction_acc,\
                            'reger_pred':beam_element_prediction_acc,\
                                'bleurt':Bleurt,'rougL':Rouge_l,'roug1':Rouge_1,\
                                    'roug2':Rouge_2,'RGF1':weighted_F1,'Avg':avg}
                        data_dicts.append(row)
                        results_csv = pd.DataFrame(data_dicts)
        except:
            pass
with open('all_results_dev_v8.csv','w') as f:        
    results_csv.to_csv(f,index=False)
