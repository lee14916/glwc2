'''
nohup python3 -u sumHdf5Group.py > log/sumHdf5Group.out &

'''

import os, shutil, click, h5py, re
import numpy as np

def run(inPath,outPath,sumDic):
    t_group=sumDic[list(sumDic)[0]][0][1]
    t_group_len=len(t_group)
    todoList=[]
    datasetDic={}
    with h5py.File(inPath) as fr, h5py.File(outPath,'w') as fw:
        def visitor_func(name, node):
            if not isinstance(node, h5py.Dataset):
                return
            datasetDic[name]=0
            t=name.find(t_group)
            if t==-1:
                return
            todoList.append((name[:t],name[t+t_group_len:]))
        fr.visititems(visitor_func)

        for ele in todoList:
            for key_sum,val in sumDic.items():
                t_sum=0
                for (coe,key_sub) in val:
                    t_sum += coe * fr[ele[0]+key_sub+ele[1]][()]
                fw.create_dataset(ele[0]+key_sum+ele[1],data=t_sum)
    print(outPath,': Done.')
    return

# 2:u+d; 4:uu+ud+/du+dd
sumDic={}
sumDic['N_2pt']={
    'p_p':[(1,'NP')],
}

sumDic['T']={
    'p_n&pi+':[(1,'Tseq'+str(i)) for i in [11,12,13,14]],
    'p_p&pi0':[(1,'Tseq'+str(i)) for i in [21,22,23,24]]+[(1,'Tseq'+str(i)) for i in [25,26]],
}

sumDic['B_2pt']={
    'n&pi+_n&pi+':[(1,'B'+str(i)) for i in [13,14,15,16]],
    'n&pi+_p&pi0':[(1,'B'+str(i)) for i in [17,18,19,20]]+[(1,'B'+str(i)) for i in [33,34,35,36]],
    'p&pi0_n&pi+':[(1,'B'+str(i)) for i in [9,10,11,12]]+[(1,'B'+str(i)) for i in []],
    'p&pi0_p&pi0':[(1,'B'+str(i)) for i in [3,4,5,6]]+[(1,'B'+str(i)) for i in []]+\
        [(1,'B'+str(i)) for i in []]+[(1,'B'+str(i)) for i in [7,8]],
}
sumDic['W_2pt']={
    'n&pi+_n&pi+':[(1,'W'+str(i)) for i in [25,26,27,28]],
    'n&pi+_p&pi0':[(1,'W'+str(i)) for i in [29,30,31,32]]+[(1,'W'+str(i)) for i in []],
    'p&pi0_n&pi+':[(1,'W'+str(i)) for i in [17,18,19,20]]+[(1,'W'+str(i)) for i in [21,22,23,24]],
    'p&pi0_p&pi0':[(1,'W'+str(i)) for i in [5,6,7,8]]+[(1,'W'+str(i)) for i in [9,10,11,12]]+\
        [(1,'W'+str(i)) for i in [13,14,15,16]]+[(1,'W'+str(i)) for i in []],
}
sumDic['Z_2pt']={
    'n&pi+_n&pi+':[(1,'Z'+str(i)) for i in [15,16]],
    'n&pi+_p&pi0':[(1,'Z'+str(i)) for i in []]+[(1,'Z'+str(i)) for i in [17,18,19,20]],
    'p&pi0_n&pi+':[(1,'Z'+str(i)) for i in []]+[(1,'Z'+str(i)) for i in [11,12,13,14]],
    'p&pi0_p&pi0':[(1,'Z'+str(i)) for i in [5,6,7,8]]+[(1,'Z'+str(i)) for i in []]+\
        [(1,'Z'+str(i)) for i in []]+[(1,'Z'+str(i)) for i in [9,10]],
}
sumDic['M_correct_2pt']={
    'n&pi+_n&pi+':[(1,'MN0PP')],
    'n&pi+_p&pi0':[],
    'p&pi0_n&pi+':[],
    'p&pi0_p&pi0':[(1,'MNPP01')]+[(1,'MNPP02')],
}

ph=-1j # fix the not correct PLEGMA phase
sumDic['B']={
    'p_j+_n&pi+':[],
    'p_j-_n&pi+':[(1*ph,'B'+str(i)) for i in [9,10,11,12]]+[(-1*ph,'B'+str(i)) for i in []],

    'p_j+_p&pi0':[],
    'p_j-_p&pi0':[(1*ph,'B'+str(i)) for i in [3,4,5,6]]+[(1*ph,'B'+str(i)) for i in []]+\
        [(-1*ph,'B'+str(i)) for i in []]+[(-1*ph,'B'+str(i)) for i in [7,8]],
}
sumDic['W']={
    'p_j+_n&pi+':[],
    'p_j-_n&pi+':[(1*ph,'W'+str(i)) for i in [17,18,19,20]]+[(-1*ph,'W'+str(i)) for i in [21,22,23,24]],

    'p_j+_p&pi0':[],
    'p_j-_p&pi0':[(1*ph,'W'+str(i)) for i in [5,6,7,8]]+[(1*ph,'W'+str(i)) for i in [9,10,11,12]]+\
        [(-1*ph,'W'+str(i)) for i in [13,14,15,16]]+[(-1*ph,'W'+str(i)) for i in []],
}
sumDic['Z']={
    'p_j+_n&pi+':[],
    'p_j-_n&pi+':[(1*ph,'Z'+str(i)) for i in []]+[(-1*ph,'Z'+str(i)) for i in [11,12,13,14]],

    'p_j+_p&pi0':[],
    'p_j-_p&pi0':[(1*ph,'Z'+str(i)) for i in [5,6,7,8]]+[(1*ph,'Z'+str(i)) for i in []]+\
        [(-1*ph,'Z'+str(i)) for i in []]+[(-1*ph,'Z'+str(i)) for i in [9,10]],
}

inPath_pre='./proj/'
# for pat in ['N_2pt','T','M_correct_2pt','B_2pt','W_2pt','Z_2pt','B','W','Z']:
for pat in ['M_correct_2pt']:
    inPath=inPath_pre+pat+'.h5'
    outPath=inPath_pre[:-1]+'_sumHdf5Group/'+pat+'.h5'
    run(inPath,outPath,sumDic[pat])