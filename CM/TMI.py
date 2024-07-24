import deepquantum as dq
import deepquantum.photonic as dqp
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def get_sigma(xj, xk):
    return np.cov(xj, xk)

def f(x):
    return (x+1/2)*np.log(x+1/2) - (x-1/2)*np.log(x-1/2)

def S(sigma):
    from scipy import linalg
    eigvals, _ = linalg.eig(sigma)
    return np.sum(f(eigvals))

class TripartiteMutualInformation():
    def __init__(
            self,
            Ns: int = 1,
            Nm: int = 2,
            NL: int = 2,
            eta: float = 0,
            delta_k: float = 0,
            r_k: float = 0,
            phi_k: float = 0,
            shots: int = 1000,
            init_state: any = 'vac',
            s_r: float = 0,
            s_phi: float = 0,
            if_save_circuit: bool = False,
            if_odd_even: bool = False  ## if True, phi_k should be a list
        ) -> None:

        N_total = Ns + Nm + NL
        theta = 2*eta - np.pi

        cir = dq.QumodeCircuit(nmode=N_total, init_state=init_state, 
                            backend='gaussian', 
                            name='try', noise=False,
                            mu = 0, sigma = 0)

        # s0,m1, two-mode squeezed vacuum (TMSV) state 
        # # S(r,theta)
        cir.s2(wires=[0, 1], r=s_r, theta=s_phi)

        # m2, single-mode squeezing vacuum (SMSV) state
        if not if_odd_even:
            for i in range(Nm+NL-1):
                cir.s(wires=2+i, r=r_k, theta=phi_k)
        else:
            for i in range(Nm+NL-1):
                cir.s(wires=2+i, r=r_k, theta=phi_k[i % 2])

        for i in range(N_total-2):
            for j in range(N_total-i-2):
                cir.bs_ry(wires=[j+1,j+2], inputs=theta)
                cir.ps(j+1, [delta_k])

        #线路可视化
        if if_save_circuit:
            cir.draw('pic/CM_circ_numME{}.svg'.format(N_total))
    
        state = cir.forward()

        # 这里measure_homodyne测量对应的物理量是正交算符 $$\hat x$$ 和 $$\hat p$$ 的值，
        # photon_number_mean_var对应的是每个mode的光子数的平均值和方差。
        # Use ``xxpp`` convention
        # 第一个光子：x1，x2，... p1，p2...
        # 第二个光子：x1，x2，... p1，p2...

        sample = cir.measure_homodyne(shots=shots)

        # # ( [平均值], [方差] )
        # photon_number = cir.photon_number_mean_var()

        s_x = np.array(sample.T[:Ns]) 
        s_p = np.array(sample.T[Ns+Nm+NL:2*Ns+Nm+NL][:])
        m_x = np.array(sample.T[Ns:(Ns+Nm)][:]) 
        m_p = np.array(sample.T[2*Ns+Nm+NL:2*Ns+2*Nm+NL][:])
        e_x = np.array(sample.T[Ns+Nm:Ns+Nm+NL][:])
        e_p = np.array(sample.T[2*Ns+2*Nm+NL:][:])

        sigma_s = np.zeros((2,2))
        sigma_m1 = np.zeros((2,2))
        sigma_m2 = np.zeros((2,2))

        sigma_s = get_sigma(s_x, s_p)
        sigma_m1 = get_sigma(m_x[0], m_p[0])
        sigma_m2 = get_sigma(m_x[1], m_p[1])
                
        sigma_m12 = np.zeros((2*Nm,2*Nm))
        sigma_sm1 = np.zeros((2*(Ns+1),2*(Ns+1)))
        sigma_sm2 = np.zeros((2*(Ns+1),2*(Ns+1)))

        choose = [['xxaa', 'xpaa', 'xxab', 'xpab'],
                ['pxaa', 'ppaa', 'pxab', 'ppab'],
                ['xxba', 'xpba', 'xxbb', 'xpbb'],
                ['pxba', 'ppba', 'pxbb', 'ppbb']]
        index_m12 = {'xa':m_x[0], 'pa':m_p[0], 'xb':m_x[1], 'pb':m_p[1]}
        index_sm1 = {'xa':s_x, 'pa':s_p, 'xb':m_x[0], 'pb':m_p[0]}
        index_sm2 = {'xa':s_x, 'pa':s_p, 'xb':m_x[1], 'pb':m_p[1]}

        for i in range(Nm*2):
            for j in range(Nm*2):
                sigma_m12[i][j] = get_sigma( index_m12[choose[i][j][0]+choose[i][j][2]], 
                                    index_m12[choose[i][j][1]+choose[i][j][3]] )[0][1]
                sigma_sm1[i][j] = get_sigma( index_sm1[choose[i][j][0]+choose[i][j][2]], 
                                    index_sm1[choose[i][j][1]+choose[i][j][3]] )[0][1]
                sigma_sm2[i][j] = get_sigma( index_sm2[choose[i][j][0]+choose[i][j][2]],
                                        index_sm2[choose[i][j][1]+choose[i][j][3]] )[0][1]

        sigma_sm12 = np.zeros((2*(Ns+Nm), 2*(Ns+Nm)))

        choose = [['xxaa', 'xpaa', 'xxab', 'xpab', 'xxac', 'xpac'],
                ['pxaa', 'ppaa', 'pxab', 'ppab', 'pxac', 'ppac'],
                ['xxba', 'xpba', 'xxbb', 'xpbb', 'xxbc', 'xpbc'],
                ['pxba', 'ppba', 'pxbb', 'ppbb', 'pxbc', 'ppbc'],
                ['xxca', 'xpca', 'xxcb', 'xpcb', 'xxcc', 'xpcc'],
                ['pxca', 'ppca', 'pxcb', 'ppcb', 'pxcc', 'ppcc']]
        index_sm12 = {'xa':s_x, 'pa':s_p, 'xb':m_x[0], 'pb':m_p[0], 'xc':m_x[1], 'pc':m_p[1]}

        for i in range(2*(Ns+Nm)):
            for j in range(2*(Ns+Nm)):
                sigma_sm12[i][j] = get_sigma( index_sm12[choose[i][j][0]+choose[i][j][2]], 
                                    index_sm12[choose[i][j][1]+choose[i][j][3]] )[0][1]

        self.I2_s_m1 = S(sigma_s) + S(sigma_m1) - S(sigma_sm1)
        self.I2_s_m2 = S(sigma_s) + S(sigma_m2) - S(sigma_sm2)
        self.I2_s_m12 = S(sigma_s) + S(sigma_m12) - S(sigma_sm12)

        self.I3_s_m1_m2 = self.I2_s_m1 + self.I2_s_m2 - self.I2_s_m12

    def I3(self):
        return self.I3_s_m1_m2
    
    def I2_SM12(self):
        return self.I2_s_m12
    
    def I2_SM1(self):
        return self.I2_s_m1
    
    def I2_SM2(self):
        return self.I2_s_m2