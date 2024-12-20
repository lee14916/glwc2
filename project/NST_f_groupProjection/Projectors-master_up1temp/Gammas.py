import sympy as sp
# DeGrandRossi BASE
# gamma_1=sp.Matrix([[0.,0.,0.,1j],[0.,0.,1j,0.],[0.,-1j,0.,0.],[-1j,0.,0.,0.]])
# gamma_2=sp.Matrix([[0.,0.,0.,-1.],[0.,0.,1.,0.],[0.,1.,0.,0.],[-1.,0.,0.,0.]])
# gamma_3=sp.Matrix([[0.,0.,1j,0.],[0.,0.,0.,-1j],[-1j,0.,0.,0.],[0.,1j,0.,0.]])
# gamma_4=sp.Matrix([[0.,0.,1.,0.],[0.,0.,0.,1.],[1.,0.,0.,0.],[0.,1.,0.,0.]])

# PLEGMA BASE
gamma_1=sp.Matrix([[0.,0.,0.,1j],[0.,0.,1j,0.],[0.,-1j,0.,0.],[-1j,0.,0.,0.]])
gamma_2=sp.Matrix([[0.,0.,0.,1.],[0.,0.,-1.,0.],[0.,-1.,0.,0.],[1.,0.,0.,0.]])
gamma_3=sp.Matrix([[0.,0.,1j,0.],[0.,0.,0.,-1j],[-1j,0.,0.,0.],[0.,1j,0.,0.]])
gamma_4=sp.Matrix([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,-1.,0.],[0.,0.,0.,-1.]])

gamma_5=(gamma_1*gamma_2*gamma_3*gamma_4).applyfunc(sp.nsimplify)

C_MATRIX  = 1j*gamma_2*gamma_4 # Other convention, Marcus uses a -1