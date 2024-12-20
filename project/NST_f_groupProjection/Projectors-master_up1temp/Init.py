import sympy as sp
from Functions import rec_dd
from Gammas import *

#spin1212 matrix
_S_4y  = ((sp.eye(4)+gamma_1*gamma_3)/sp.sqrt(2)).applyfunc(sp.nsimplify)
_S_4z  = ((sp.eye(4)+gamma_2*gamma_1)/sp.sqrt(2)).applyfunc(sp.nsimplify)
_S_I   = gamma_4
init_S = {'C_4y':_S_4y,'C_4z':_S_4z,'I_sE':_S_I,}

#spin1 matrix
_Rot3d_4y = sp.Matrix([[0,0,1],[0,1,0],[-1,0,0]])
_Rot3d_4z = sp.Matrix([[0,-1,0],[1,0,0],[0,0,1]])
_Rot3d_I  = sp.Matrix([[-1,0,0],[0,-1,0],[0,0,-1]])
init_R    = {'C_4y':_Rot3d_4y,'C_4z':_Rot3d_4z,'I_sE':_Rot3d_I,}

### irrep_matrix
init_G = rec_dd()
### 2Oh
#A1g/u
_A1g_4y = sp.eye(1)
_A1g_4z = sp.eye(1)
_A1g_I  = sp.eye(1)
init_G['2Oh']['A1g'] = {'C_4y':_A1g_4y,'C_4z':_A1g_4z,'I_sE':_A1g_I,}
init_G['2Oh']['A1u'] = {'C_4y':_A1g_4y,'C_4z':_A1g_4z,'I_sE':-_A1g_I,}
#A2g/u
_A2g_4y = -sp.eye(1)
_A2g_4z = -sp.eye(1)
_A2g_I  = sp.eye(1)
init_G['2Oh']['A2g'] = {'C_4y':_A2g_4y,'C_4z':_A2g_4z,'I_sE':_A2g_I,}
init_G['2Oh']['A2u'] = {'C_4y':_A2g_4y,'C_4z':_A2g_4z,'I_sE':-_A2g_I,}
#Eg/u
_Eg_4y = sp.Matrix([[1,sp.sqrt(3)],[sp.sqrt(3),-1]])/2
_Eg_4z = sp.Matrix([[-1,0],[0,1]])
_Eg_I  = sp.eye(2)
init_G['2Oh']['Eg'] = {'C_4y':_Eg_4y,'C_4z':_Eg_4z,'I_sE':_Eg_I,}
init_G['2Oh']['Eu'] = {'C_4y':_Eg_4y,'C_4z':_Eg_4z,'I_sE':-_Eg_I,}
#T1g/u
_T1g_4y = sp.Matrix([[0,0,1],[0,1,0],[-1,0,0]])
_T1g_4z = sp.Matrix([[0,-1,0],[1,0,0],[0,0,1]])
_T1g_I  = sp.eye(3)
init_G['2Oh']['T1g'] = {'C_4y':_T1g_4y,'C_4z':_T1g_4z,'I_sE':_T1g_I,}
init_G['2Oh']['T1u'] = {'C_4y':_T1g_4y,'C_4z':_T1g_4z,'I_sE':-_T1g_I,}
#T2g/u
_T2g_4y = sp.Matrix([[0,0,-1],[0,-1,0],[1,0,0]])
_T2g_4z = sp.Matrix([[0,1,0],[-1,0,0],[0,0,-1]])
_T2g_I  = sp.eye(3)
init_G['2Oh']['T2g'] = {'C_4y':_T2g_4y,'C_4z':_T2g_4z,'I_sE':_T2g_I,}
init_G['2Oh']['T2u'] = {'C_4y':_T2g_4y,'C_4z':_T2g_4z,'I_sE':-_T2g_I,}
#G1g/u
_G1g_4y = sp.Matrix([[1,-1],[1,1]])/sp.sqrt(2)
_G1g_4z = sp.Matrix([[1-1j,0],[0,1+1j]])/sp.sqrt(2)
_G1g_I  = sp.eye(2)
init_G['2Oh']['G1g'] = {'C_4y':_G1g_4y,'C_4z':_G1g_4z,'I_sE':_G1g_I,}
init_G['2Oh']['G1u'] = {'C_4y':_G1g_4y,'C_4z':_G1g_4z,'I_sE':-_G1g_I,}
#G2g/u
_G2g_4y = -sp.Matrix([[1,-1],[1,1]])/sp.sqrt(2)
_G2g_4z = -sp.Matrix([[1-1j,0],[0,1+1j]])/sp.sqrt(2)
_G2g_I  = sp.eye(2)
init_G['2Oh']['G2g'] = {'C_4y':_G2g_4y,'C_4z':_G2g_4z,'I_sE':_G2g_I,}
init_G['2Oh']['G2u'] = {'C_4y':_G2g_4y,'C_4z':_G2g_4z,'I_sE':-_G2g_I,}
#Hg/u
_Hg_4y  = sp.Matrix([[1,-sp.sqrt(3),sp.sqrt(3),-1], [sp.sqrt(3),-1,-1,sp.sqrt(3)],
                     [sp.sqrt(3),1,-1,-sp.sqrt(3)], [1,sp.sqrt(3),sp.sqrt(3),1]])/2/sp.sqrt(2)
_Hg_4z  = sp.Matrix([[-1-1j, 0, 0, 0], [0, 1-1j, 0, 0], [0, 0, 1+1j, 0], [0, 0, 0, -1+1j]])/sp.sqrt(2)
_Hg_I   = sp.eye(4)
init_G['2Oh']['Hg'] = {'C_4y':_Hg_4y,'C_4z':_Hg_4z,'I_sE':_Hg_I,}
init_G['2Oh']['Hu'] = {'C_4y':_Hg_4y,'C_4z':_Hg_4z,'I_sE':-_Hg_I,}

### 2C4v
#A1
_A1_4z  = sp.eye(1)
_A1_i2y = sp.eye(1)
init_G['2C4v'][(0,0,1)]['A1'] = {'C_4z':_A1_4z,'I_sC_2y':_A1_i2y,}
#A2
_A2_4z  = sp.eye(1)
_A2_i2y = -sp.eye(1)
init_G['2C4v'][(0,0,1)]['A2'] = {'C_4z':_A2_4z,'I_sC_2y':_A2_i2y,}
#B1
_B1_4z  = -sp.eye(1)
_B1_i2y = sp.eye(1)
init_G['2C4v'][(0,0,1)]['B1'] = {'C_4z':_B1_4z,'I_sC_2y':_B1_i2y,}
#B2
_B2_4z  = -sp.eye(1)
_B2_i2y = -sp.eye(1)
init_G['2C4v'][(0,0,1)]['B2'] = {'C_4z':_B2_4z,'I_sC_2y':_B2_i2y,}
#E
_E_4z   = sp.Matrix([[0,-1],[1,0]])
_E_i2y  = sp.Matrix([[1,0],[0,-1]])
init_G['2C4v'][(0,0,1)]['E'] = {'C_4z':_E_4z,'I_sC_2y':_E_i2y,}
#G1
_G1_4z  = sp.Matrix([[1-1j,0],[0,1+1j]])/sp.sqrt(2)
_G1_i2y = sp.Matrix([[0,-1],[1,0]])
init_G['2C4v'][(0,0,1)]['G1'] = {'C_4z':_G1_4z,'I_sC_2y':_G1_i2y,}
#G2
_G2_4z  = -sp.Matrix([[1-1j,0],[0,1+1j]])/sp.sqrt(2)
_G2_i2y = sp.Matrix([[0,-1],[1,0]])
init_G['2C4v'][(0,0,1)]['G2'] = {'C_4z':_G2_4z,'I_sC_2y':_G2_i2y,}

### 2C2v
#A1
_A1_2e  = sp.eye(1)
_A1_i2f = sp.eye(1)
init_G['2C2v'][(0,1,1)]['A1'] = {'C_2e':_A1_2e,'I_sC_2f':_A1_i2f,}
#A2
_A2_2e  = sp.eye(1)
_A2_i2f = -sp.eye(1)
init_G['2C2v'][(0,1,1)]['A2'] = {'C_2e':_A2_2e,'I_sC_2f':_A2_i2f,}
#B1
_B1_2e  = -sp.eye(1)
_B1_i2f = sp.eye(1)
init_G['2C2v'][(0,1,1)]['B1'] = {'C_2e':_B1_2e,'I_sC_2f':_B1_i2f,}
#B2
_B2_2e  = -sp.eye(1)
_B2_i2f = -sp.eye(1)
init_G['2C2v'][(0,1,1)]['B2'] = {'C_2e':_B2_2e,'I_sC_2f':_B2_i2f,}
#G
_G_2e  = sp.Matrix([[-1j,-1],[1,1j]])/sp.sqrt(2)
_G_i2f = sp.Matrix([[1j,-1],[1,-1j]])/sp.sqrt(2)
init_G['2C2v'][(0,1,1)]['G'] = {'C_2e':_G_2e,'I_sC_2f':_G_i2f,}

### 2C3v
#A1
_A1_3dl = sp.eye(1)
_A1_i2b = sp.eye(1)
init_G['2C3v'][(1,1,1)]['A1'] = {'C_3delta':_A1_3dl,'I_sC_2b':_A1_i2b,}
#A2
_A2_3dl = sp.eye(1)
_A2_i2b = -sp.eye(1)
init_G['2C3v'][(1,1,1)]['A2'] = {'C_3delta':_A2_3dl,'I_sC_2b':_A2_i2b,}
#E
_E_3dl = sp.Matrix([[-1,sp.sqrt(3)],[-sp.sqrt(3),-1]])/2
_E_i2b = sp.Matrix([[-1,0],[0,1]])
init_G['2C3v'][(1,1,1)]['E'] = {'C_3delta':_E_3dl,'I_sC_2b':_E_i2b,}
#F1
_F1_3dl = sp.Matrix([[-1]])
_F1_i2b = sp.Matrix([[1j]])
init_G['2C3v'][(1,1,1)]['F1'] = {'C_3delta':_F1_3dl,'I_sC_2b':_F1_i2b,}
#F2
_F2_3dl = sp.Matrix([[-1]])
_F2_i2b = sp.Matrix([[-1j]])
init_G['2C3v'][(1,1,1)]['F2'] = {'C_3delta':_F2_3dl,'I_sC_2b':_F2_i2b,}
#G
_G_3dl = sp.Matrix([[1-1j,-1-1j],[1-1j,1+1j]])/2
_G_i2b = sp.Matrix([[0,1-1j],[-1-1j,0]])/sp.sqrt(2)
init_G['2C3v'][(1,1,1)]['G'] = {'C_3delta':_G_3dl,'I_sC_2b':_G_i2b,}

###
init_G = init_G.todict()