from collections import defaultdict as ddict
import sympy as sp
import numpy as np
import pickle as pkl
import math
import copy 
from itertools import product

from GSorth import orth_rows
from Gammas import *

###    Dictionaries    #
########################
class rec_dd(ddict):
    def __init__(self, *args, **kwargs):
        super().__init__(rec_dd, *args, **kwargs)
    def __repr__(self):
        return repr(dict(self))
    def todict(self):
        out = { k:self[k].todict() if hasattr(self[k],"todict") else self[k] for k in self }
        return out
def copy_dict(d):
    return copy.deepcopy(d)

def merge_dicts(d1,d2):
    out = {}
    
    common_keys = list(set(d1.keys()).intersection(set(d2.keys())))
    excl_k1     = [ k for k in d1 if k not in common_keys ]
    excl_k2     = [ k for k in d2 if k not in common_keys ]
    
    for k in excl_k1:
        out[k] = copy_dict(d1[k])
    for k in excl_k2:
        out[k] = copy_dict(d2[k])
    for k in common_keys:
        assert(hasattr(d1[k],'keys') and hasattr(d2[k],'keys')),str(d1)
        out[k] = merge_dicts(d1[k],d2[k])
    return out

def dict_repl(in_d,repl):
    out_d = {}
    for k in in_d:
        newk = repl[k] if k in repl else k
        assert(newk not in out_d),newk
        if hasattr(in_d[k],'keys'):
            out_d[newk] = dict_repl(in_d[k],repl)
        else:
            out_d[newk] = in_d[k]
    return out_d

def d_isequal(d1,d2):
    out = len(d1)==len(d2)
    for k in d1:
        #print(k)
        if k not in d2:
            out = False
        else:
            if hasattr(d1[k],'keys') or hasattr(d2[k],'keys'):
                out = out and d_isequal(d1[k],d2[k])
            else:
                if isinstance(d1[k],np.ndarray):
                    out = out and np.all(d1[k]==d2[k])
                else:
                    out = out and (d1[k]==d2[k])
    return out

###  Handling momenta   #
#########################

# len 3 sp or np array to '(p[0],p[1],p[2])' string
def p_tostr(p):
    return "("+str(p[0])+","+str(p[1])+","+str(p[2])+")"

def momsq(p):
    return p[0]**2+p[1]**2+p[2]**2

# delta kroen product between arbitrary number of matrices, input order matters!
def __deltakroen(a,b):
    na  = int(math.sqrt(len(a))) #3
    nb  = int(math.sqrt(len(b))) #4
    out = sp.zeros(na*nb,na*nb)
    for i in range(na):
        for j in range(na):
            out[i*nb:(i+1)*nb,j*nb:(j+1)*nb] = a[i,j]*b[:,:]
    return out.applyfunc(sp.nsimplify)
def deltakroen(*args):
    assert(len(args)>1)
    largs = list(args)
    b = largs.pop(-1)
    while len(largs)>0:
        a = largs.pop(-1)
        b = __deltakroen(a,b)
    return b

# Calc p1,p2 lists of momenta for 2particles at fixed P_tot, and |p1|^2 = msq1, |p2|^2 = msq2
def momenta_1p(msq):
    P=[]
    p1max = math.floor(math.sqrt(msq))
    for p1 in range(-p1max,p1max+1):
        p2max = math.floor(math.sqrt(msq-p1**2))
        for p2 in range(-p2max,p2max+1):
            p3s = math.floor(math.sqrt(msq-p1**2-p2**2))
            p3s = [0,] if p3s==0 else [-p3s,p3s]
            for p3 in p3s:
                if p1**2+p2**2+p3**2!=msq:
                    continue
                P.append((p1,p2,p3))
    return P

# Calc p1,p2 lists of momenta for 2particles at fixed P_tot, and |p1|^2 = msq1, |p2|^2 = msq2
def momenta_2p(P_ref,msq1,msq2):
    P1, P2 = [],[]
    p1max = math.floor(math.sqrt(msq1))
    for p1 in range(-p1max,p1max+1):
        p2max = math.floor(math.sqrt(msq1-p1**2))
        for p2 in range(-p2max,p2max+1):
            p3s = math.floor(math.sqrt(msq1-p1**2-p2**2))
            p3s = [0,] if p3s==0 else [-p3s,p3s]
            for p3 in p3s:
                if p1**2+p2**2+p3**2!=msq1:
                    continue
                p1_tmp = (p1,p2,p3)
                p2_tmp = (P_ref[0]-p1,P_ref[1]-p2,P_ref[2]-p3)
                if (p2_tmp[0]**2+p2_tmp[1]**2+p2_tmp[2]**2)==msq2:
                    P1.append(p1_tmp)
                    P2.append(p2_tmp)
    return P1,P2

def compute_otherPref( pref, _repr ):
    l_moms = momenta_1p( momsq(pref) )
    pref   = sp.Matrix(pref)
    out = {}
    for mom in l_moms:
        mom_a = sp.Matrix(mom)
        if mom_a==pref:
            continue
        for G in _repr:
            if (_repr[G]*pref).applyfunc(sp.nsimplify) == mom_a:
                out[mom]=G
                break
    assert( len(out)==len(l_moms)-1 ),out
    return out

###    Interpolators    #
#########################
_aux_dict  = {'1':sp.eye(4),'i':1j*sp.eye(4),'c':C_MATRIX,'g1':gamma_1,
              'g2':gamma_2,'g3':gamma_3,'g4':gamma_4,'g5':gamma_5}

def list_interps(strlist):
    out = dict(map(lambda x: (x,compute_gmatrix(x,_aux_dict)), strlist))
    out['list'] = strlist
    return out

def split_gmatrix(st):
    assert(len(st)>0)
    out = []
    index0 = 0
    index1 = 0
    while (index1 < len(st)):
        if st[index0]=='g':
            index1 = index0+2
        else:
            index1 = index0+1
        out.append(st[index0:index1])
        index0 = index1
    return out
def compute_gmatrix(st,dikt):
    out = sp.eye(4)
    for g in split_gmatrix(st):
        out *= dikt[g]
    return out.applyfunc(sp.nsimplify)

###  Calculate repr of G #
##########################

def extract_expr( gmap, subs):
    list_subs_sym = [ (sym,sp.MatrixSymbol(k,*mat.shape)) for k, (sym,mat) in subs.items() ]
    list_subs     = [ (list_subs_sym[i][1],mat) if list_subs_sym[i][0]==sym else None  
                      for i,(k,(sym,mat)) in enumerate(subs.items()) ]
    assert(not None in list_subs)

    U_expr = { G : gmap[G].subs( list_subs_sym,simultaneous=True ) for G in gmap }
    U      = { G : U_expr[G].subs( list_subs ).doit().expand() for G in U_expr }

    return U_expr, U

def extract_repr( group, init ):
    l_init = list(init.keys())
    l_symb = sp.symbols(','.join(l_init), commutative=False)
    subs   = { name:symb for name, symb in zip(l_init,l_symb) }
    gmap   = group.build_groupmap( **subs )
    
    subs   = { name:(symb,init[name]) for name, symb in zip(l_init,l_symb) }
    _, out = extract_expr( gmap, subs )
    return out

####  Projectors #
##################

def computeProjs(Gammas,Us,beta,conj=False,trans=False):
    #dimension of irrep
    dim_I = Gammas['E'].shape[0]
    projs = {}
    assert(beta<dim_I)
    #for each row
    for lamda in range(dim_I):
        (a,b) = (beta,lamda) if trans else (lamda,beta)
        if conj:
            tmp_coll = [ ((Gammas[G][a,b]).conjugate()*Us[G]).doit().expand() for G in Gammas ]
        else:
            tmp_coll = [ (Gammas[G][a,b]*Us[G]).doit().expand() for G in Gammas ]
        projs['l%d'%(lamda+1)] = (sum( tmp_coll, sp.zeros(*Us['E'].shape) )*dim_I/len(tmp_coll)).applyfunc(sp.nsimplify)
    
    coeff = {'l%d'%(beta+1):orth_rows(projs['l%d'%(beta+1)])}
    for mu in range(dim_I):
        if mu==beta or coeff['l%d'%(beta+1)]==sp.zeros(0,0):
            continue
        coeff['l%d'%(mu+1)] = orth_rows(coeff['l%d'%(beta+1)]*projs['l%d'%(mu+1)])
    return projs, coeff

### Functions that write coefficients to file #
###############################################
#read pickle file
def read_pickle(file):
    content=[]
    with open(file,'rb') as infile:
        try:
            while True:
                content.append(pkl.load(infile,encoding='latin1'))
        except EOFError:
            pass
    return content

#save objects on pickle file
def write_pickle(obj,file,append=False):
    fmode = 'ab' if append else 'wb'
    with open(file,fmode) as outfile:
        pkl.dump(obj,outfile)
    return None

#out[LG][I][lamda][ptot][n][interp][s]
def output_1p(coeff, inters):
    out      = rec_dd()
    printstr = []
    for LG in coeff:
        for I in coeff[LG]:
            dim_I = len( coeff[LG][I][list(coeff[LG][I].keys())[0]] )
            for l in range(dim_I):
                row = str(l+1)
                for p in coeff[LG][I]:
                    nsize = coeff[LG][I][p]['l'+row].shape[0]
                    for n in range(nsize):
                        repl    = chr(ord('`')+n+1)
                        aux_str = 'out[%s][%s][%s][%s][%s]'%(LG,I,row,p,repl)
                        out[LG][I]['l'+row][p][repl], l_str = add_out(coeff[LG][I][p]['l'+row].row(n), inters['list'])
                        for s in l_str:
                            printstr.append(aux_str+s)
    return out.todict(), printstr

def add_out(row, linters):
    out   = {}
    l_str = []
    #'gs index expected'
        
    for i_i,i in enumerate(linters):
        out[i] = np.array( row.evalf()[i_i*4:(i_i+1)*4], dtype=np.complex128 )
    for i in out:
        for i_s,s in enumerate(out[i]):
            l_str.append('[%s][%s]=%s'%(i,i_s,str(s)))

    return out, l_str

def add_out2p(row, i_obj):
    out   = {}
    l_str = []
    p1, p2 = i_obj.name
    i_obj.dims
    #'gs index expected'
    for i_m, (mom1,mom2) in enumerate(zip(i_obj.ps[p1],i_obj.ps[p2])):
        m = ( tuple((mom1.T.tolist())[0]),tuple((mom2.T.tolist())[0]) )
        out[m]={}
        for i_i, (g1,g2) in enumerate(product(i_obj.l_inter[p1],i_obj.l_inter[p2])):
            i = g1+';'+g2
            assert(i_i==0)
            out[m][i] = np.array( row.evalf()[i_m*4:(i_m+1)*4], dtype=np.complex128 )
    for m in out:
        for i in out[m]:
            for i_s,s in enumerate(out[m][i]):
                l_str.append('[%s][%s][%s]=%s'%(m,i,i_s,str(s)))

    return out, l_str

DIM_I = {'A':1,'B':1,'E':2,'T':3,'G':2,'H':4,'F':1}
# out2p[LG][I][lamda][ptot][N?P?][n][([p1],[p2])][interp][s]
def output_2p(coeff, ps, inters1, inters2):
    out      = rec_dd()
    printstr = []
    for LG in coeff:
        for I in coeff[LG]:
            dim_I = DIM_I[I[:1]]
            for l in range(dim_I):
                row = str(l+1)
                for p in coeff[LG][I]:
                    for labNP in coeff[LG][I][p]:
                        aux_i = interpolator_2p(['N','\\pi'],(ps[p][labNP]['N'],ps[p][labNP]['\\pi']),['gs','g'],[inters1,inters2])

                        nsize = coeff[LG][I][p][labNP]['l'+row].shape[0]
                        for n in range(nsize):
                            repl    = chr(ord('`')+n+1)
                            aux_str = 'out[%s][%s][%s][%s][%s][%s]'%(LG,I,row,p,labNP,repl)
                            out[LG][I]['l'+row][p][labNP][repl], l_str = add_out2p(coeff[LG][I][p][labNP]['l'+row].row(n), aux_i)
                            for s in l_str:
                                printstr.append(aux_str+s)
    return out.todict(), printstr