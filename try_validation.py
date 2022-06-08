import os
import numpy as np

path = 'logs_new/lm.sa.sta.bs.1t5dim.1024lstm.wd.0.0.md80120/seed0/log_static_2022-06-07.txt'

batch_valid = []
batch_test = []

with open(path,'r') as f:
    all_lines = f.readlines()

i = 0
while i < len(all_lines):
    line = all_lines[i]
    try:
        if 'Evaluate DEV' in line.split('-')[-1]:
            s_v = float(all_lines[i+2].split(':')[-1])
            n_v = float(all_lines[i+3].split(':')[-1])
            r_v = float(all_lines[i+4].split(':')[-1])
            f_v = float(all_lines[i+5].split(':')[-1])
            s_v_ = float(all_lines[i+7].split(':')[-1])
            n_v_ = float(all_lines[i+8].split(':')[-1])
            r_v_ = float(all_lines[i+9].split(':')[-1])
            f_v_ = float(all_lines[i+10].split(':')[-1])
            batch_valid.append([s_v,n_v,r_v,f_v,s_v_,n_v_,r_v_,f_v_])
            i += 11
            continue
        
        if 'Evaluate TEST' in line.split('-')[-1]:
            s_v = float(all_lines[i+2].split(':')[-1])
            n_v = float(all_lines[i+3].split(':')[-1])
            r_v = float(all_lines[i+4].split(':')[-1])
            f_v = float(all_lines[i+5].split(':')[-1])
            s_v_ = float(all_lines[i+7].split(':')[-1])
            n_v_ = float(all_lines[i+8].split(':')[-1])
            r_v_ = float(all_lines[i+9].split(':')[-1])
            f_v_ = float(all_lines[i+10].split(':')[-1])
            batch_test.append([s_v,n_v,r_v,f_v,s_v_,n_v_,r_v_,f_v_])
            i += 11
            continue
    except:
        break

    i+=1

batch_valid.pop() # final presentation of acc should be removed
array_valid = np.array(batch_valid)
array_test = np.array(batch_test)

max_value = np.max(array_valid,axis=0)
max_idx = np.argmax(array_valid,axis=0)

max_value_mean = np.max(np.mean(array_valid,axis=1),axis=0)
max_idx_mean = np.argmax(np.mean(array_valid,axis=1),axis=0)
max_value_mean_1 = np.max(np.mean(array_valid[:,0:4],axis=1),axis=0)
max_idx_mean_1 = np.argmax(np.mean(array_valid[:,0:4],axis=1),axis=0)
max_value_mean_2 = np.max(np.mean(array_valid[:,4:8],axis=1),axis=0)
max_idx_mean_2 = np.argmax(np.mean(array_valid[:,4:8],axis=1),axis=0)


max_idx_list = max_idx.tolist()
max_idx_list += [max_idx_mean_1,max_idx_mean_2,max_idx_mean]



model_name = path.split('/')[1]
with open('acc_record.txt','a') as f:
    f.write(model_name+'\n')
    for idx in max_idx_list:
        print(idx,array_test[idx],sum(array_test[idx]))
        f.write(str(idx)+'\t'+str(array_test[idx])+'\t'+str(sum(array_test[idx]))+'\n')
    f.write('\n')
    
pass