from itertools import product
import numpy as np
import sympy as sp
import math
from Functions import deltakroen, p_tostr, momenta_1p, momenta_2p

#p_tostr,deltakroen,momenta_2p,interpolator_sp, interpolator_2p,computeR,computeMomR

# Matrix that encodes how different interpolators transforms under R in G
def computeR( list_interp, S, func):
    '''
    Computes R_ab(g) s.t. func( G_a, S) = R_ab G_b where G_b are different gammas used in Delta(or other hadronic) interpolators.  
    R_ab should be equal to Rot3d(g)_ba.
    - list_interp : list of Gammas
    - S : matrix that implement rotation on the single quark field (S(g)^-1 for sink, S(g)^T for src)
    - func : how S affect Gamma. For Delta func x,y: y.T*x*y
    '''
    R = []
    for i in list_interp:
        row = [0,]*len(list_interp)
        iprime = func(i,S)
        try:
            j = list_interp.index(iprime)
        except ValueError:
            try:
                j = list_interp.index(-iprime)
            except ValueError:
                print("No so general yet.")
                sp.pprint(iprime)
                raise ValueError
            else:
                row[j] = -1
        else:
            row[j] = 1
        R.append(row)
    return sp.Matrix(R)

# M matrix: how momenta transform under the action of R in G
# For 1particle not needed, R always in LG(p), p does not change under U
def computeMomR(Plist,Rot):
    R = []
    for p in Plist:
        row    = [0,]*len(Plist)
        pprime = (Rot*p).doit()
        try:
            j = Plist.index(pprime)
        except ValueError:
            print("No so general yet.")
            sp.pprint(pprime)
            raise ValueError
        else:
            row[j] = 1
        R.append(row)
    return sp.Matrix(R)


#Single particle interpolator
class interpolator_1p(object):
    def __init__(self,particle,p_list,index,g_inter=None):
        momenta    = [ p_tostr(m) for m in p_list ]
        tmp_list   = [ momenta,]
        
        for label in index:
            if label=='s':
                tmp_list.append([0,1,2,3])
            elif label=='g':
                tmp_list.append([i for i in range(len(g_inter)-1)])
            else:
                assert(False),'Error'
        aux = '%d,'*(len(tmp_list)-1)
        
        self.name  = particle
        self.dims  = list( map( len, tmp_list) )
        self.index = index
        if isinstance(p_list[0],(sp.MutableDenseMatrix,sp.ImmutableDenseMatrix)):
            self.ps = p_list
        else: 
            self.ps = list(map( sp.Matrix, p_list ))
        self.l_inter = g_inter['list']
        self.inter   = [ g_inter[g] for g in self.l_inter ]

        components = [ particle+('%s'+'('+aux[:-1]+')')%tuple(l) for l in product(*tmp_list) ]
        self.components = sp.Matrix(list( map( sp.Symbol, components) ))
        self.size  = len(components)
        
    #this is where I define my U_ij!!
    def Wt(self,Rot,S,f_i):
        M = computeMomR( self.ps, Rot )
        R = computeR( self.inter, S, f_i )
        tmp_list = [M,]
        for l in self.index:
            if l=='s':
                tmp_list.append(S)
            elif l=='g':
                tmp_list.append(R)
            else:
                assert(False),'Error'
        return deltakroen(*tmp_list)
    def rotate_moms(self,Rot):
        pprime = [ (Rot*p).doit() for p in self.ps ]
        d_int  = { name : g for name,g in zip(self.l_inter,self.inter) }
        d_int['list'] = self.l_inter
        return interpolator_1p(self.name,pprime,self.index,d_int)
    def apply_outR(self,Rot,S,f_i):
        M = sp.eye(len(self.ps))
        psprime = [ (Rot*p).doit() for p in self.ps ]
        R = computeR( self.inter, S, f_i ) 
        tmp_list = [M,]
        for l in self.index:
            if l=='s':
                tmp_list.append(S)
            elif l=='g':
                tmp_list.append(R)
            else:
                assert(False),'Error'
        return psprime, deltakroen(*tmp_list)

#Two particles interpolator
class interpolator_2p(object):
    def __init__(self,particles,p_lists,indexs,g_inters=[None,None]):
        part1, part2     = particles
        index1, index2   = indexs
        ginter1, ginter2 = g_inters
        
        momenta   = [ (p_tostr(m1),p_tostr(m2)) for (m1,m2) in zip(*p_lists) ]
        tmp_list1 = [ ] 
        tmp_list2 = [ ] #momenta, idx1, idx2
        
        for l in index1:
            if l=='s':
                tmp_list1.append([0,1,2,3])
            elif l=='g':
                tmp_list1.append([i for i in range(len(ginter1)-1)])
            else:
                assert(False),'Error'
        for l in index2:
            if l=='s':
                tmp_list2.append([0,1,2,3])
            elif l=='g':
                tmp_list2.append([i for i in range(len(ginter2)-1)])
            else:
                assert(False),'Error'
        
        self.name  = (part1, part2)
        self.dims  = list( map( len, [momenta,] + tmp_list1 + tmp_list2) )
        self.index = { part1:index1, part2:index2 }

        self.ps    = {}
        if isinstance(p_lists[0][0],(sp.MutableDenseMatrix,sp.ImmutableDenseMatrix)):
            ps = p_lists  
        else:
            ps = zip(*list(map( lambda x : [sp.Matrix(x[0]),sp.Matrix(x[1])], zip(*p_lists) )))
        self.ps[part1],self.ps[part2] = ps
        

        self.l_inter = { part1:ginter1['list'], part2:ginter2['list'] }
        self.inter   = { part1:[ginter1[g] for g in self.l_inter[part1]], part2:[ ginter2[g] for g in self.l_inter[part2]] }

        tmp_list1  = [ l for l in tmp_list1 if len(l)>1 ]
        tmp_list2  = [ l for l in tmp_list2 if len(l)>1 ]
        n_i1, n_i2 = len(tmp_list1), len(tmp_list2)
        tmp_list   = [ momenta, ] + tmp_list1 + tmp_list2
        aux        = part1+'%s('+('%d,'*n_i1)[:-1]+')' if n_i1>0 else part1+'%s'
        aux       += part2+'%s('+('%d,'*n_i2)[:-1]+')' if n_i2>0 else part2+'%s'
        
        components      = [ aux%tuple(l[0][:1]+l[1:n_i1+1]+l[0][1:]+l[n_i1+1:]) for l in product(*tmp_list) ]
        self.components = sp.Matrix(list( map( sp.Symbol, components) ))
        self.size  = len(components)
        
    def Wt(self,R,S,f_i1,f_i2):
        n1,n2 = self.name
        
        M  = computeMomR( self.ps[n1], R )
        R1 = computeR( self.inter[n1], S, func=f_i1 ) 
        R2 = computeR( self.inter[n2], S, func=f_i2 ) 
        tmp_list = [M,]
        for l in self.index[n1]:
            if l=='s':
                tmp_list.append(S)
            elif l=='g':
                tmp_list.append(R1)
            else:
                assert(False),'Error'
        for l in self.index[n2]:
            if l=='s':
                tmp_list.append(S)
            elif l=='g':
                tmp_list.append(R2)
            else:
                assert(False),'Error'
        return deltakroen(*tmp_list)
    
    def rotate_moms(self,Rot):
        p1, p2 = self.name
        
        pprime = [ [(Rot*p).doit() for p in self.ps[p1]], 
                   [(Rot*p).doit() for p in self.ps[p2]] ]
        
        idxs   = [ self.index[p1], self.index[p2] ]
        d_int1 = { name : g for name,g in zip(self.l_inter[p1],self.inter[p1]) }
        d_int1['list'] = self.l_inter[p1]
        d_int2 = { name : g for name,g in zip(self.l_inter[p2],self.inter[p2]) }
        d_int2['list'] = self.l_inter[p2]
        
        return interpolator_2p(self.name,pprime,idxs,[d_int1,d_int2])

    def apply_outR(self,Rot,S,f_i1,f_i2):
        n1,n2 = self.name
        psprime = { n1:[ (Rot*p).doit() for p in self.ps[n1] ],
                    n2:[ (Rot*p).doit() for p in self.ps[n2] ]}
        
        M  = sp.eye( len(self.ps[n1]) )
        R1 = computeR( self.inter[n1], S, func=f_i1 ) 
        R2 = computeR( self.inter[n2], S, func=f_i2 ) 
        tmp_list = [M,]
        for l in self.index[n1]:
            if l=='s':
                tmp_list.append(S)
            elif l=='g':
                tmp_list.append(R1)
            else:
                assert(False),'Error'
        for l in self.index[n2]:
            if l=='s':
                tmp_list.append(S)
            elif l=='g':
                tmp_list.append(R2)
            else:
                assert(False),'Error'
        return psprime, deltakroen(*tmp_list)
    