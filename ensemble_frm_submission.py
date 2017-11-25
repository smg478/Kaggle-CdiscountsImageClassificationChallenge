import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from statistics import mode
import gzip, csv

#######################################################


folder_dir = 'submit/lb_submit/'
df_test = pd.read_csv('input/sample_submission.csv')
df_test = df_test.sort_values('_id', ascending=True)
ids_test = df_test['_id']

names = []
for id in ids_test:
    names.append('{}'.format(id))

print("Start loading candidate submissions:")
cand1 = pd.read_csv(folder_dir + 'lb_0743_submit_ensemble.csv')
cand1 = cand1.sort_values('_id', ascending=True)
cand1 = cand1["category_id"]

cand2 = pd.read_csv(folder_dir + 'lb_0737_enhao_xception_my_submission.csv.gz',compression='gzip')
cand2 = cand2.sort_values('_id', ascending=True)
cand2 = cand2["category_id"]

cand3 = pd.read_csv(folder_dir + 'lb_0745_enhao_my_submission_inception_resnet_v2.csv.gz',compression='gzip')
cand3 = cand3.sort_values('_id', ascending=True)
cand3 = cand3["category_id"]

cand4 = pd.read_csv(folder_dir + 'lb_0706_incep_res_0001_idx0_batch500_zoom0_shift0_adam_99-1.3947-0.7928-0.7105.csv.gz',compression='gzip')
cand4 = cand4.sort_values('_id', ascending=True)
cand4 = cand4["category_id"]

cand5 = pd.read_csv(folder_dir + 'lb_0696_seIncep3_180_train_aug_test_noAug_pyTorch.csv.gz',compression='gzip')
cand5 = cand5.sort_values('_id', ascending=True)
cand5 = cand5["category_id"]

cand6 = pd.read_csv(folder_dir + 'lb_07099_xception_0001_idx0_batch960_zoom0_shift0_adam_45-1.1024-0.7602-0.7291.csv.gz',compression='gzip')
cand6 = cand6.sort_values('_id', ascending=True)
cand6 = cand6["category_id"]

cand7 = pd.read_csv(folder_dir + 'lb_07170_incep3_0001_idx0_batch960_zoom0_shift0_adam_45-1.1022-0.7960-0.7313.csv.gz',compression='gzip')
cand7 = cand7.sort_values('_id', ascending=True)
cand7 = cand7["category_id"]

cand8 = pd.read_csv(folder_dir + 'lb_0740_incep_incepRes_xcep_avgScore.csv.gz',compression='gzip')
cand8 = cand8.sort_values('_id', ascending=True)
cand8 = cand8["category_id"]

cand9 = pd.read_csv(folder_dir + 'lb_0728_ensem_7mdl.csv.gz',compression='gzip')
cand9 = cand9.sort_values('_id', ascending=True)
cand9 = cand9["category_id"]

cand10 = pd.read_csv(folder_dir + 'lb_0706_incep_res_0001_idx0_batch500_zoom0_shift0_adam_99-1.3947-0.7928-0.7105.csv.gz',compression='gzip')
cand10 = cand10.sort_values('_id', ascending=True)
cand10 = cand10["category_id"]

cand11 = pd.read_csv(folder_dir + 'lb_0699_incepRes_180_train_test_noAug_7435.csv.gz',compression='gzip')
cand11 = cand11.sort_values('_id', ascending=True)
cand11 = cand11["category_id"]

cand12 = pd.read_csv(folder_dir + 'lb_0698_incepRes_180_train_test_idx0_shift_flip_7036.csv.gz',compression='gzip')
cand12 = cand12.sort_values('_id', ascending=True)
cand12 = cand12["category_id"]

#cand13 = pd.read_csv(folder_dir + 'xception_idx0_noAug_adam_ClsWt_13-1.1968-0.7353-0.7241.csv.gz',compression='gzip')
#cand13 = cand13.sort_values('_id', ascending=True)
#cand13 = cand13["category_id"]

#cand14 = pd.read_csv(folder_dir + 'xception_idx0_noAug_adam_ClsWt_13-1.1968-0.7353-0.7241.csv.gz',compression='gzip')
#cand14 = cand14.sort_values('_id', ascending=True)
#cand14 = cand14["category_id"]

print("Done loading candidate submissions")

##########################################################

preds = []

print('Start Doing Merged Submission Predictions:')
for i in tqdm(range(len(ids_test)), miniters=1000):
    pred_merge = [cand1[i],
                  cand2[i],
                  cand3[i],
                  cand4[i],
                  #cand5[i],
                  cand6[i],
                  cand7[i],
                  #cand8[i],
                  #cand9[i],
                  #cand10[i],
                  #cand11[i],
                  #cand12[i],
                  #cand13[i],
                  #cand14[i]
                  ]
    data = Counter(pred_merge)
    most_common = data.most_common(1)  # Returns the highest occurring item
    preds.append(most_common[0][0])

print("Generating submission file...")

df = pd.DataFrame({'_id': names, 'category_id': preds})
df.to_csv('submit/shai3_jandjensem1_enhao2.csv.gz', index=False, compression='gzip')
print("All done!")


#######################################################