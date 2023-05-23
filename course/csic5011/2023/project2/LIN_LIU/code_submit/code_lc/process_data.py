import scipy.io as sio
import numpy as np
import os
import pandas as pd
#############
data_type = "college"
if data_type == "college":
    filter_list = ["Close to my home and where I will work", "like american universities","Tsinghua university","I know little about both two university"
    ,"Tongji university is one of top universities in China."
    ,"I don't know both of them"
    ,"both", "Shanghai Jiao Tong University"]
    print(len(filter_list))
    # sss
    os.makedirs('college', exist_ok=True)
    data = pd.read_csv('data/college_new.csv')
    for k in filter_list:
        data = data[data['Winner Text']!=k]
        data = data[data['Loser Text']!=k]
    gt = pd.read_csv('data/college_gt.csv')['Idea Text'].values
    print(len(gt))
    gt = [i for i in gt if i not in filter_list]
    print(len(gt))
    x = np.unique(np.concatenate([data['Winner Text'].values, data['Loser Text'].values]))
    print([i for i in gt if i not in x])
    gt_dict = {}
    for k in range(len(gt)):
        gt_dict[gt[k]] = k
    gt = np.concatenate([np.reshape(gt,(len(gt), 1) ), np.reshape( np.array(list(range(len(gt)))), (len(gt), 1)) ],1)
    select_data = data[['Session ID', 'Winner Text', 'Loser Text']].values
    pref = np.zeros((len(data)))
    pref[data['Winner ID'].values==data['Left Choice ID'].values] = 1
    pref[data['Winner ID'].values!=data['Left Choice ID'].values] = -1
    # print(pref)
    # sss
    select_data[:, 1] = [gt_dict[i] for i in select_data[:, 1]]
    select_data[:, 2] = [gt_dict[i] for i in select_data[:, 2]]
    ann = np.unique(select_data[:, 0])
    ann_dict = {}
    for k in range(len(ann)):
        ann_dict[ann[k]] = k
    select_data[:, 0] =  [ann_dict[i] for i in select_data[:, 0]]
    college_data = np.ones((len(select_data), 1)).astype(np.int64)
    college_data = np.concatenate([select_data,college_data], 1)
    print(college_data)
    annotator = np.unique(college_data[:, 0])
    print('Number of Annotators M : ', len(annotator))
    print('Number of Pairs : ', len( select_data))
    print('Number of Nodes n : ', len(gt))
    # ###cal number of edges
    pairs = college_data[:,1:3]
    pairs = np.reshape([str(i) for i in pairs.ravel()], pairs.shape)
    pairs_string = ['_'.join(i) for i in pairs] 
    print('Number of Edges E : ', len(np.unique(pairs_string)))
    n_anno = len(np.unique( college_data[:,0]))
    # np.save('college/college_data.npy', college_data)
    # np.save('college/college_gt.npy', gt)
    fileonj = open('college_prefer.csv', 'w')
    fileonj.write('id,left,right,ratio')
    fileonj.write('\n')
    for k in range(n_anno):
        tmp = college_data[college_data[:,0]==k]
        fileonj.write(','.join([str(k), str(len([i for i in pref[college_data[:,0]==k] if i>0])), str(len([i for i in pref[college_data[:,0]==k]  if i<0])), str(len([i for i in pref[college_data[:,0]==k]  if i>0])/(len([i for i in pref[college_data[:,0]==k]  if i<0])+len([i for i in pref[college_data[:,0]==k]  if i>0])))]))
        fileonj.write('\n')
    fileonj.close()
    ssss
    
elif data_type == "age":
    gt_data_path  = "./data/age/Groundtruth.mat"
    ag_data_path  = "./data/age/Agedata.mat"
    gt_data = sio.loadmat(gt_data_path)
    ag_data = sio.loadmat(ag_data_path)
    ag_data["Pair_Compar"][:,0] = ag_data["Pair_Compar"][:,0]-1
    ag_data["Pair_Compar"][:,1:3] = ag_data["Pair_Compar"][:,1:3]-1
    

    # print(np.unique(ag_data["Pair_Compar"][:,3]))
    new_pair =  ag_data["Pair_Compar"]
    print(np.unique(new_pair[:,3]))
    select_ind = [i for i in range(len(new_pair)) if int(new_pair[i,3])!=0]
    new_pair = new_pair[select_ind]
    n_anno = len(np.unique( new_pair[:,0]))
    print(n_anno)
    fileonj = open('age_prefer.csv', 'w')
    fileonj.write('id,left,right,ratio')
    fileonj.write('\n')
    for k in range(n_anno):
        tmp = new_pair[new_pair[:,0]==k]
        fileonj.write(','.join([str(k), str(len([i for i in tmp[:,-1] if i>0])), str(len([i for i in tmp[:,-1] if i<0])), str(len([i for i in tmp[:,-1] if i>0])/(len([i for i in tmp[:,-1] if i<0])+len([i for i in tmp[:,-1] if i>0])))]))
        fileonj.write('\n')
    fileonj.close()
    ssss
   
    print(np.unique(new_pair[:,3]))


    for  k in range(len(new_pair)):
        if new_pair[k][3] <0 :
            new_pair[k,1:3] = [new_pair[k,2], new_pair[k,1]]
    new_pair[:,3] = 1
    # print(np.unique(new_pair[:,3]))
    # sss
    os.makedirs('age', exist_ok=True)
    np.save('age/age_data.npy', new_pair)
    np.save('age/age_gt.npy', gt_data['Age'])
    #### cal M N E n
    print('Number of Annotators M : ', len(np.unique( new_pair[:,0])))
    print('Number of Pairs : ', len( new_pair))
    print('Number of Nodes n : ', len(np.unique(new_pair[:,1:3].ravel())))
    ###cal number of edges
    pairs = new_pair[:,1:3]
    pairs = np.reshape([str(i) for i in pairs.ravel()], pairs.shape)
    pairs_string = ['_'.join(i) for i in pairs] 
    print('Number of Edges E : ', len(np.unique(pairs_string)))