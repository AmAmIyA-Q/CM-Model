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
            m1_r: float = 0,
            m1_phi: float = 0,
        ) -> None:

        N_total = Ns + Nm + NL

        # initial state
        cov = torch.eye(2*(N_total+1))
        mean = torch.zeros(2*(N_total+1))

        # cir = dq.QumodeCircuit(nmode=num_ME+1, init_state=[cov, mean], 
        #                        backend='gaussian', 
        #                        name='try', noise=False)

        cir = dq.QumodeCircuit(nmode=N_total, init_state='vac', 
                            backend='gaussian', 
                            name='try', noise=False,
                            mu = 0, sigma = 0)

        # s0,m1, two-mode squeezed vacuum (TMSV) state 
        # # S(r,theta)
        cir.s(0, [s_r, s_phi])
        cir.s(1, [m1_r, m1_phi])

        # m2, single-mode squeezing vacuum (SMSV) state 
        cir.s(2, [r_k, phi_k])

        for i in range(N_total-2):
            for j in range(N_total-i-2):
                cir.bs_real(wires=[j+1,j+2], inputs=[eta, np.nan])
                cir.ps(j+1, [delta_k])

        #线路可视化
        cir.draw('pic/CM_circ_num_ME{}.svg'.format(N_total))
    
        state = cir.forward()

        # 这里measure_homodyne测量对应的物理量是正交算符 $$\hat x$$ 和 $$\hat p$$ 的值，
        # photon_number_mean_var对应的是每个mode的光子数的平均值和方差。

        # 第一个光子：x1，p1，x2，p2...
        # 第二个光子：x1，p1，x2，p2...

        sample = cir.measure_homodyne(shots=shots)

        # ( [平均值], [方差] )
        photon_number = cir.photon_number_mean_var()

        s_xj = np.array(sample.T[0])
        s_xk = np.array(sample.T[1])
        m_xp = np.array(sample.T[2*1:2*(Ns+Nm)][:]) # m1x, m1p, m2x, m2p
        e_xp = np.array(sample.T[2*(Ns+Nm):][:])

        sigma_s = np.zeros((2,2))
        sigma_m1 = np.zeros((2,2))
        sigma_m2 = np.zeros((2,2))

        sigma_s = get_sigma(s_xj, s_xk)
        sigma_m1 = get_sigma(m_xp[0], m_xp[1])
        sigma_m2 = get_sigma(m_xp[2], m_xp[3])
                
        sigma_m12 = np.zeros((2*Nm,2*Nm))
        sigma_sm1 = np.zeros((2*(Ns+1),2*(Ns+1)))
        sigma_sm2 = np.zeros((2*(Ns+1),2*(Ns+1)))

        choose = [['xxaa', 'xpaa', 'xxab', 'xpab'],
                ['pxaa', 'ppaa', 'pxab', 'ppab'],
                ['xxba', 'xpba', 'xxbb', 'xpbb'],
                ['pxba', 'ppba', 'pxbb', 'ppbb']]
        index_m12 = {'xa':m_xp[0], 'pa':m_xp[1], 'xb':m_xp[2], 'pb':m_xp[3]}
        index_sm1 = {'xa':s_xj, 'pa':s_xk, 'xb':m_xp[0], 'pb':m_xp[1]}
        index_sm2 = {'xa':s_xj, 'pa':s_xk, 'xb':m_xp[2], 'pb':m_xp[3]}

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
        index_sm12 = {'xa':s_xj, 'pa':s_xk, 'xb':m_xp[0], 'pb':m_xp[1], 'xc':m_xp[2], 'pc':m_xp[3]}

        for i in range(2*(Ns+Nm)):
            for j in range(2*(Ns+Nm)):
                sigma_sm12[i][j] = get_sigma( index_sm12[choose[i][j][0]+choose[i][j][2]], 
                                    index_sm12[choose[i][j][1]+choose[i][j][3]] )[0][1]

        I2_s_m1 = S(sigma_s) + S(sigma_m1) - S(sigma_sm1)
        I2_s_m2 = S(sigma_s) + S(sigma_m2) - S(sigma_sm2)
        self.I2_s_m12 = S(sigma_s) + S(sigma_m12) - S(sigma_sm12)

        self.I3_s_m1_m2 = I2_s_m1 + I2_s_m2 - self.I2_s_m12

    def I3(self):
        return self.I3_s_m1_m2
    
    def I2_SM12(self):
        return self.I2_s_m12