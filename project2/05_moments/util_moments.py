import util as yu
from util import *

j2j1={
    'jq':[[1,'j+'],[1,'js'],[1,'jc']],
    'jv1':[[1,'j-']],
    'jv2':[[1,'j+'],[-2,'js']],
    'jv3':[[1,'j+'],[1,'js'],[-3,'jc']]
}
fla2iso={
    'u':[(1/4,'q'),(1/2,'v1'),(1/6,'v2'),(1/12,'v3')],
    'd':[(1/4,'q'),(-1/2,'v1'),(1/6,'v2'),(1/12,'v3')],
    's':[(1/4,'q'),(-1/3,'v2'),(1/12,'v3')],
    'c':[(1/4,'q'),(-1/4,'v3')]
}
fla2iso_conn={
    'u':[(1/2,'q'),(1/2,'v1')],
    'd':[(1/2,'q'),(-1/2,'v1')],
}

lat_a2s_plt=np.arange(0,0.009,0.001)

def extendBare_avgx(key2bare):
    keys=list(key2bare)
    enss=yu.removeDuplicates([ens for ens,j in keys])
    stouts=yu.removeDuplicates([int(j.split('stout')[-1]) for ens,j in keys if 'stout' in j])
    
    for ens in enss:
        ZERO=key2bare[(ens,'j+;conn')]*0
        key2bare[(ens,'js;conn')]=ZERO
        key2bare[(ens,'jc;conn')]=ZERO
        key2bare[(ens,'j-;disc')]=ZERO
        for cd in ['conn','disc']:
            for j in j2j1.keys():
                key2bare[(ens,f'{j};{cd}')]=np.sum([factor*key2bare[(ens,f'{j1};{cd}')] for factor,j1 in j2j1[j]],axis=0)
        
def bareRC2phy_avgx(key2bare,ens2RCs,mn_conn='mu=nu',mn_disc='mu!=nu',mn_g='mu!=nu'):
    keys=list(key2bare)
    enss=yu.removeDuplicates([ens for ens,j in keys])
    stouts=yu.removeDuplicates([int(j.split('stout')[-1]) for ens,j in keys if 'stout' in j])
    
    key2phy={}
    for ens in enss:
        for j in ['jv1','jv2','jv3']:
            key2phy[(ens,f'{j};conn')]=ens2RCs[ens][f'Zqq({mn_conn})']*key2bare[(ens,f'{j};conn')]
            key2phy[(ens,f'{j};disc')]=ens2RCs[ens][f'Zqq({mn_disc})']*key2bare[(ens,f'{j};disc')]
            key2phy[(ens,f'{j}')]=key2phy[(ens,f'{j};conn')]+key2phy[(ens,f'{j};disc')]
            
        for stout in stouts:
            key2phy[(ens,f'jq;conn;stout{stout}')]=ens2RCs[ens][f'Zqq^s({mn_conn})']*key2bare[(ens,'jq;conn')]
            key2phy[(ens,f'jq;disc;stout{stout}')]=ens2RCs[ens][f'Zqq^s({mn_disc})']*key2bare[(ens,'jq;disc')]
            key2phy[(ens,f'jq;0mix;stout{stout}')]=key2phy[(ens,f'jq;conn;stout{stout}')]+key2phy[(ens,f'jq;disc;stout{stout}')]
            
            key2phy[(ens,f'jq;mix;stout{stout}')]=ens2RCs[ens][f'Zqg({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
            key2phy[(ens,f'jq;stout{stout}')]=key2phy[(ens,f'jq;0mix;stout{stout}')]+key2phy[(ens,f'jq;mix;stout{stout}')]
            
            key2phy[(ens,f'jg;0mix;stout{stout}')]=ens2RCs[ens][f'Zgg^{stout}({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
            key2phy[(ens,f'jg;mix;stout{stout}')]=ens2RCs[ens][f'Zgq({mn_conn})']*key2bare[(ens,'jq;conn')]+ens2RCs[ens][f'Zgq({mn_disc})']*key2bare[(ens,'jq;disc')]
            key2phy[(ens,f'jg;stout{stout}')]=key2phy[(ens,f'jg;mix;stout{stout}')]+key2phy[(ens,f'jg;0mix;stout{stout}')]
            
            key2phy[(ens,f'jtot;stout{stout}')]=key2phy[(ens,f'jq;stout{stout}')]+key2phy[(ens,f'jg;stout{stout}')]
            
            for fla in fla2iso.keys():
                key2phy[(ens,f'j{fla};stout{stout}')]=np.sum([factor*(key2phy[(ens,f'j{iso};stout{stout}')] if iso in ['q'] else key2phy[(ens,f'j{iso}')])  for factor,iso in fla2iso[fla]],axis=0)
    
    for j in ['jv1','jv2','jv3']+[f'jq;stout{stout}' for stout in stouts]+[f'jg;stout{stout}' for stout in stouts]:
        ens2dat={ens:key2phy[(ens,j)] for ens in enss}
        fits=yu.doFit_continuumExtrapolation(ens2dat,lat_a2s_plt=lat_a2s_plt)
        for fit in fits:
            fitlabel,pars_jk,chi2_jk,Ndof=fit
            key2phy[(f'a=#_{fitlabel}',j)]=pars_jk
        pars_jk,probs_jk=yu.jackMA(fits)
        key2phy[('a=#_MA',j)]=pars_jk
    
    for stout in stouts:
        for fitlabel in ['const','linear','MA']:
            for fla in fla2iso.keys():
                key2phy[(f'a=#_{fitlabel}',f'j{fla};stout{stout}')]=np.sum([factor*(key2phy[(f'a=#_{fitlabel}',f'j{iso};stout{stout}')] if iso in ['q'] else key2phy[(f'a=#_{fitlabel}',f'j{iso}')]) for factor,iso in fla2iso[fla]],axis=0)
            key2phy[(f'a=#_{fitlabel}',f'jtot;stout{stout}')]=key2phy[(f'a=#_{fitlabel}',f'jq;stout{stout}')]+key2phy[(f'a=#_{fitlabel}',f'jg;stout{stout}')]
    
    return key2phy

def bareRC2phy_avgx_pre(key2bare,ens2RCs,ens2RCs_pre,mn_conn='mu=nu',mn_disc='mu!=nu',mn_g='mu!=nu'):
    keys=list(key2bare)
    enss=yu.removeDuplicates([ens for ens,j in keys]); enss=['b']
    stouts=yu.removeDuplicates([int(j.split('stout')[-1]) for ens,j in keys if 'stout' in j])
    
    key2phy={}
    for ens in enss:
        for j in ['jv1','jv2','jv3']:
            key2phy[(ens,f'{j};conn')]=ens2RCs[ens][f'Zqq({mn_conn})']*key2bare[(ens,f'{j};conn')]
            key2phy[(ens,f'{j};disc')]=ens2RCs[ens][f'Zqq({mn_disc})']*key2bare[(ens,f'{j};disc')]
            key2phy[(ens,f'{j}')]=key2phy[(ens,f'{j};conn')]+key2phy[(ens,f'{j};disc')]
            
        for stout in stouts:
            key2phy[(ens,f'jq;conn;stout{stout}')]=ens2RCs[ens][f'Zqq^s({mn_conn})']*key2bare[(ens,'jq;conn')]
            key2phy[(ens,f'jq;disc;stout{stout}')]=ens2RCs_pre[ens][f'Zqq^s^{stout}({mn_disc})']*key2bare[(ens,'jq;disc')]
            key2phy[(ens,f'jq;0mix;stout{stout}')]=key2phy[(ens,f'jq;conn;stout{stout}')]+key2phy[(ens,f'jq;disc;stout{stout}')]
            
            key2phy[(ens,f'jq;mix;stout{stout}')]=ens2RCs_pre[ens][f'Zqg^{stout}({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
            key2phy[(ens,f'jq;stout{stout}')]=key2phy[(ens,f'jq;0mix;stout{stout}')]+key2phy[(ens,f'jq;mix;stout{stout}')]
            
            key2phy[(ens,f'jg;0mix;stout{stout}')]=ens2RCs_pre[ens][f'Zgg^{stout}({mn_g})']*key2bare[(ens,f'jg;stout{stout}')]
            key2phy[(ens,f'jg;mix;stout{stout}')]=ens2RCs[ens][f'Zgq({mn_conn})']*key2bare[(ens,'jq;conn')]+ens2RCs_pre[ens][f'Zgq^{stout}({mn_disc})']*key2bare[(ens,'jq;disc')]
            key2phy[(ens,f'jg;stout{stout}')]=key2phy[(ens,f'jg;mix;stout{stout}')]+key2phy[(ens,f'jg;0mix;stout{stout}')]
            
            key2phy[(ens,f'jtot;stout{stout}')]=key2phy[(ens,f'jq;stout{stout}')]+key2phy[(ens,f'jg;stout{stout}')]
            
            for fla in fla2iso.keys():
                key2phy[(ens,f'j{fla};stout{stout}')]=np.sum([factor*(key2phy[(ens,f'j{iso};stout{stout}')] if iso in ['q'] else key2phy[(ens,f'j{iso}')])  for factor,iso in fla2iso[fla]],axis=0)        
    
    return key2phy

def convert_key2phy_stout(key2phy_old,stout):
    key2phy={}
    for key in key2phy_old.keys():
        ens,j=key
        if j.endswith(f';stout{stout}'):
            key2phy[(ens,j.removesuffix(f';stout{stout}'))]=key2phy_old[key]
        if 'stout' not in j:
            key2phy[key]=key2phy_old[key]
    return key2phy
def convert_key2phy_stouts(key2phy_old,stouts):
    key2phy={}
    for key in key2phy_old.keys():
        ens,j=key
        if j.endswith(f';stout{stouts[0]}'):
            j0=j.removesuffix(f';stout{stouts[0]}')
            key2phy[(ens,j0)]=np.mean([key2phy_old[(ens,f'{j0};stout{stout}')] for stout in stouts],axis=0)
        if 'stout' not in j:
            key2phy[key]=key2phy_old[key]
    return key2phy                

def makePlot_a2dependence_avgx(list_dic):
    Ncol=len(list_dic)
    fig, axs = yu.getFigAxs(2,Ncol,Lrow=4,Lcol=6,sharex=True,sharey='row')
    ax=axs[0,0]
    ax.set_xlim([0,lat_a2s_plt[-1]])
    ax.set_xticks([0,0.003,0.006])
    ax.set_ylim([0.2,1.4])
    ax.set_yticks([0.4,0.6,0.8,1.0,1.2])
    ax.set_ylabel(r'$\langle \mathrm{x} \rangle_{q,g}$')
    ax=axs[1,0]
    ax.set_ylim([-0.1,0.5])
    ax.set_ylabel(r'$\langle \mathrm{x}\rangle_{q}$')
    for icol in range(Ncol):
        axs[0,icol].axhline(1,color='black',ls='--',marker='')
        axs[0,icol].axvline(0,color='black',ls='dotted',marker='')
        axs[1,icol].set_xlabel(r'$a^2$ [fm$^2$]')
        axs[1,icol].axhline(0,color='black',ls='--',marker='')
        
    for icol in range(Ncol):
        dic=list_dic[icol]
        def setParameter(default,key):
            return dic[key] if key in dic else default
        
        key2phy=dic['key2phy']
        key2phy_pre=setParameter(None,'key2phy_pre')
        
        keys=list(key2phy)
        enss=yu.removeDuplicates([ens for ens,j in keys if 'a=#' not in ens])
        
        def get(ens,j):
            return key2phy[(ens,j)]
        def get_pre(ens,j):
            return key2phy_pre[(ens,j)]
        
        j2color={'jq':'purple','jg':'cyan','jtot':'grey','ju':'r','jd':'g','js':'b','jc':'orange'}
        j2label={'jq':'q','jg':'g','jtot':'N','ju':'u','jd':'d','js':'s','jc':'c'}
        j2fmt={'jq':'d','jg':'s','jtot':'o','ju':'^','jd':'v','js':'<','jc':'>'}
        
        
        ax=axs[0,icol]
        js=['jq','jtot','jg']
        for ij,j in enumerate(js):
            color=j2color[j]
            mean,err=yu.jackme(get('a=#_MA',j))
            label=rf'$\langle x\rangle _{{{j2label[j]}}}=$'+yu.un2str(mean[0],err[0],forceResult=1)
            for iens,ens in enumerate(enss):
                plt_x=yu.ens2a[ens]**2+(ij-len(js)/2)*5e-5; plt_y,plt_yerr=yu.jackme(get(ens,j))
                ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],label=label if iens==0 else None)
                
                if key2phy_pre is None or ens not in ['b']:
                    continue
                plt_x=yu.ens2a[ens]**2+0.0001; plt_y,plt_yerr=yu.jackme(get_pre(ens,j))
                ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],mfc='white')
                    
            mean,err=yu.jackme(get('a=#_MA',j))
            x=lat_a2s_plt; ymin=mean-err; ymax=mean+err
            ax.plot(x,mean,color=color,linestyle='--',marker='')
            ax.fill_between(x, ymin, ymax, color=color, alpha=0.1)
        ax.legend(fontsize=10,ncol=2)

        ax=axs[1,icol]
        js=['ju','jd','js','jc']
        for ij,j in enumerate(js):
            color=j2color[j]
            mean,err=yu.jackme(get('a=#_MA',j))
            label=rf'$\langle x\rangle _{{{j2label[j]}}}=$'+yu.un2str(mean[0],err[0],forceResult=1)
            for iens,ens in enumerate(enss):
                plt_x=yu.ens2a[ens]**2+(ij-len(js)/2)*5e-5; plt_y,plt_yerr=yu.jackme(get(ens,j))
                ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],label=label if iens==0 else None)
                if key2phy_pre is None or ens not in ['b']:
                    continue
                plt_x=yu.ens2a[ens]**2+0.0001; plt_y,plt_yerr=yu.jackme(get_pre(ens,j))
                ax.errorbar(plt_x,plt_y,plt_yerr,color=color,fmt=j2fmt[j],mfc='white')

            mean,err=yu.jackme(get('a=#_MA',j))
            x=lat_a2s_plt; ymin=mean-err; ymax=mean+err
            ax.plot(x,mean,color=color,linestyle='--',marker='')
            ax.fill_between(x, ymin, ymax, color=color, alpha=0.1)
        ax.legend(fontsize=10,ncol=2)
    
    return fig,axs