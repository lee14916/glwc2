'''
nohup python3 -u setupdata.py > nohup.out  &
'''

import h5py,os,pickle
import numpy as np

basepath='/p/project/pines/li47/code/projectData/discNJN/'
path_data='/p/project/pines/li47/code/projectData/discNJN/temp/data.h5'
cfgs={}
data={}

ensList=['cB64','cC80','cD96']
ens2ensemble={'cA48':'cA2.09.48','cB64':'cB211.072.64','cC80':'cC211.060.80','cD96':'cD211.054.96'}

# if os.path.exists(path_data):
#     os.remove(path_data)

with h5py.File(path_data, 'w') as fw:
    for ens in ensList:
        data[ens]={}
        ensemble=ens2ensemble[ens]

        with h5py.File(basepath+ensemble+'/data_merge/N.h5') as f:
            cfgs[ens]=list(f['data'].keys())
        
        with h5py.File(basepath+'others/'+ens+'/thrp-conn.h5') as f:
            cfgs[ens]=list(set(cfgs[ens])&set(f['gA']['up']['dt10'])) 
        
        # cfgs[ens]=cfgs[ens][:5]
        print(ens,len(cfgs[ens]))
        
        for diag in ['N','N-j','N-jbw']:
            data[ens][diag]={}
            with h5py.File(basepath+ensemble+'/data_merge/'+diag+'.h5') as f:
                t=f['mvec'][:]
                fw.create_dataset(ens+'/'+diag+'/mvec',data=t)
                for ky in f['data'][cfgs[ens][0]].keys():
                    # data[ens][diag][ky]=np.array([f['data'][cfg][ky][:] for cfg in cfgs[ens]])
                    t=np.array([f['data'][cfg][ky][:] for cfg in cfgs[ens]])
                    fw.create_dataset(ens+'/'+diag+'/'+ky,data=t)
            print(ens,diag)
        
        for diag in ['NJN']:
            data[ens][diag]={}
            with h5py.File(basepath+'others/'+ens+'/thrp-conn.h5') as f:
                for ky in ['gS','gA','gT']:
                    for dt in f[ky]['up'].keys():
                        # data[ens]['NJN'][ky+'_j+_'+'_deltat_'+dt[2:]]=np.array([f[ky]['up'][dt][cfg][:]+f[ky]['dn'][dt][cfg][:] for cfg in cfgs[ens]])
                        t=np.array([f[ky]['up'][dt][cfg][:]+f[ky]['dn'][dt][cfg][:] for cfg in cfgs[ens]])
                        fw.create_dataset(ens+'/'+diag+'/'+ky+'_j+'+'_deltat_'+dt[2:],data=t)
                        t=np.array([f[ky]['up'][dt][cfg][:]-f[ky]['dn'][dt][cfg][:] for cfg in cfgs[ens]])
                        fw.create_dataset(ens+'/'+diag+'/'+ky+'_j-'+'_deltat_'+dt[2:],data=t)
            print(ens,diag)  
        print()        
    
print('Done!')