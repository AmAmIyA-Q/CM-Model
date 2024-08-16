import numpy as np
from numpy import sin, cos, sinh, cosh, pi, nan


def symplectic_eigenvalues(sigma):
    
    V = sigma

    eigvals, eigvecs = np.linalg.eig(V)
    # print("Eigenvalues of V:\n", eigvals)

    n = V.shape[0] // 2
    In = np.eye(n)
    J = np.block([[np.zeros((n, n)), -In], [In, np.zeros((n, n))]])

    # print("Matrix J:\n", J)

    # Compute the spectral decomposition of iJV
    eigvals, eigvecs = np.linalg.eig(1j * J @ V)
    # print("Eigenvalues of iJV:\n", eigvals)

    return eigvals[eigvals > 0]

    # Compute the matrix B: iJV = B D B^(-1)
    B = eigvecs
    sigmaz_Dn = np.diag(eigvals)
    B_inv = np.linalg.inv(B)

    # print("sigmaz_Dn:\n", sigmaz_Dn)
    # print("Matrix B:\n", B)

    # Verify the decomposition
    check_matrix = B @ sigmaz_Dn @ B_inv
    # print("Matrix B D B^(-1):\n", check_matrix)
    # print("Matrix iJV:\n", 1j * J @ V)
    # print("Verification:", np.allclose(1j * J @ V, check_matrix))

    U2 = 1/np.sqrt(2) * np.array([[1, 1], [1j, -1j]])

    # Compute matrix S
    S = J.T @ B @ np.kron(np.conjugate(U2).T, In) @ J
    # print("Matrix S:\n", S)

    Dn = np.diag(eigvals[:n])
    # print("Matrix Dn:\n", Dn.real)

    # Verify result
    I2 = np.eye(2)
    check_matrix = S @ np.kron(I2, Dn) @ S.T
    # print("Matrix S (D ⊕ D) S^T:\n", check_matrix.real)
    # print("Matrix V:\n", V)
    # print("Verification:", np.allclose(V, check_matrix))

    # sigma_z = np.array([[1, 0], [0, -1]])
    # iJV = np.linalg.inv(S).T @ np.kron(U2, In) @ np.kron(sigma_z, Dn) @ np.kron(np.conjugate(U2).T, In) @ S.T
    # print("Matrix iJV:\n", iJV)
    # print("Verification:", np.allclose(iJV, 1j * J @ V))

def f(x):
    return (x+1/2)*np.log(x+1/2) - (x-1/2)*np.log(x-1/2)

def S(sigma):
    from scipy import linalg
    # eigvals, _ = linalg.eig(sigma)
    eigvals = symplectic_eigenvalues(sigma)
    return np.sum(f(eigvals))

class QumodeWithLoss():
    def __init__(self, circuit, loss, params, order, compute_layers):
        self.layers = len(circuit[0])
        self.n_modes = len(circuit)
        self.circuit = circuit
        self.loss = loss
        self.params = params
        self.loss_params = np.log(10) / 10
        self.order = order
        self.compute_layers = compute_layers

        # print('layers:', self.layers)
        # print('n_modes:', self.n_modes)
        # print('compute_layers:', self.compute_layers)

    def forward(self):
        cov = np.eye(self.n_modes*2, dtype=complex)
        for i in range(self.compute_layers):
            for j in range(self.n_modes):
                type = self.order[str(self.circuit[j][i])]
                params = self.params[j][i]
                matrix = self.get_matrix(type=type, params=params, 
                                    dimention=self.n_modes*2, 
                                    position=j)
                cov = matrix @ cov @ matrix.conj().T

                # add loss
                cov = np.exp(-self.loss[j][i]*self.loss_params) * cov + \
                        (1 - np.exp(-self.loss[j][i]*self.loss_params)) * np.eye(self.n_modes*2, dtype=complex)

        return cov

    
    def get_matrix(self, type, params, dimention, position):

        def bs_matrix(eta):
            matrix = np.eye(dimention, dtype=complex)
            phi = pi
            theta = pi/2 - eta
            bs =  np.array([[cos(theta), -cos(phi)*sin(theta),           0, -sin(phi)*sin(theta)],
                            [cos(phi)*sin(theta),  cos(theta), -sin(phi)*sin(theta),           0],
                            [         0,  sin(phi)*sin(theta),  cos(theta), -cos(phi)*sin(theta)],
                            [sin(phi)*sin(theta),           0,  cos(phi)*sin(theta),  cos(theta)]])
            # matrix[2*position+1:2*position+3, 2*position+1:2*position+3] = bs
            matrix[position][position] = bs[0,0]
            matrix[position][position+1] = bs[0,1]
            matrix[position][position+self.n_modes] = bs[0,2]
            matrix[position][position+self.n_modes+1] = bs[0,3]
            matrix[position+1][position] = bs[1,0]
            matrix[position+1][position+1] = bs[1,1]
            matrix[position+1][position+self.n_modes] = bs[1,2]
            matrix[position+1][position+self.n_modes+1] = bs[1,3]
            matrix[position+self.n_modes][position] = bs[2,0]
            matrix[position+self.n_modes][position+1] = bs[2,1]
            matrix[position+self.n_modes][position+self.n_modes] = bs[2,2]
            matrix[position+self.n_modes][position+self.n_modes+1] = bs[2,3]
            matrix[position+self.n_modes+1][position] = bs[3,0]
            matrix[position+self.n_modes+1][position+1] = bs[3,1]
            matrix[position+self.n_modes+1][position+self.n_modes] = bs[3,2]
            matrix[position+self.n_modes+1][position+self.n_modes+1] = bs[3,3]
            return matrix

        def s_matrix(xi):
            r = xi[0]
            theta = xi[1]
            matrix = np.eye(dimention, dtype=complex)
            s = np.array([[cosh(r)-cos(theta)*sinh(r), -sin(theta)*sinh(r)],
                            [-sin(theta)*sinh(r), cosh(r)+cos(theta)*sinh(r)]])
            # matrix[2*position:2*position+2, 2*position:2*position+2] = s
            matrix[position][position] = s[0,0]
            matrix[position][position+self.n_modes] = s[0,1]
            matrix[position+self.n_modes][position] = s[1,0]
            matrix[position+self.n_modes][position+self.n_modes] = s[1,1]
            return matrix
        
        def s2_matrix(xi):
            r = xi[0]
            theta = xi[1]
            matrix = np.eye(dimention, dtype=complex)
            s2 = np.array([[cosh(r), cos(theta)*sinh(r), 0, sin(theta)*sinh(r)],
                            [cos(theta)*sinh(r), cosh(r), sin(theta)*sinh(r), 0],
                            [0, sin(theta)*sinh(r), cosh(r), -cos(theta)*sinh(r)],
                            [sin(theta)*sinh(r), 0, -cos(theta)*sinh(r), cosh(r)]])
            # matrix[2*position:2*position+4, 2*position:2*position+4] = s2
            matrix[position][position] = s2[0,0]
            matrix[position][position+self.n_modes] = s2[0,2]
            matrix[position][position+1] = s2[0,1]
            matrix[position][position+self.n_modes+1] = s2[0,3]
            matrix[position+self.n_modes][position] = s2[2,0]
            matrix[position+self.n_modes][position+self.n_modes] = s2[2,2]
            matrix[position+self.n_modes][position+1] = s2[2,1]
            matrix[position+self.n_modes][position+self.n_modes+1] = s2[2,3]
            matrix[position+1][position] = s2[1,0]
            matrix[position+1][position+self.n_modes] = s2[1,2]
            matrix[position+1][position+1] = s2[1,1]
            matrix[position+1][position+self.n_modes+1] = s2[1,3]
            matrix[position+self.n_modes+1][position] = s2[3,0]
            matrix[position+self.n_modes+1][position+self.n_modes] = s2[3,2]
            matrix[position+self.n_modes+1][position+1] = s2[3,1]
            matrix[position+self.n_modes+1][position+self.n_modes+1] = s2[3,3]
            return matrix
        
        def ps_matrix(theta):
            matrix = np.eye(dimention, dtype=complex)
            ps = np.array([[cos(theta), -sin(theta)],
                            [sin(theta), cos(theta)]])
            # matrix[2*position+1, 2*position+1] = ps
            matrix[position, position] = ps[0,0]
            matrix[position, position+self.n_modes] = ps[0,1]
            matrix[position+self.n_modes, position] = ps[1,0]
            matrix[position+self.n_modes, position+self.n_modes] = ps[1,1]

            return matrix

        if type == 'BS' :
            # print('BS', bs_matrix(params).real)
            return bs_matrix(params)
        elif type == 'S' :
            # print('S', s_matrix(params).real)
            return s_matrix(params)
        elif type == 'S2' :
            # print('S2', s2_matrix(params).real)
            return s2_matrix(params)
        elif type == 'PS' :
            # print('PS', ps_matrix(params).real)
            return ps_matrix(params)
        else:
            return np.eye(dimention, dtype=complex)


class TripartiteMutualInformation():
    def __init__(
            self,
            # scale
            Ns: int = 1,
            Nm: int = 2,
            NL: int = 5,

            # BS
            eta: float = 1,

            # S2
            s_r: float = 0.6,
            s_phi: float = 0,

            # PS 
            delta_k: float = 0,

            # S
            r_k: float = 1.4,
            phi_k: float = 0,

            # loss
            if_add_loss: bool = False,
            if_loss_input : bool = False,
            loss_amplitude_bs : float = None,
            loss_input : np.array = None,

            # additonal features
            if_odd_even: bool = False,
            if_part_computing: bool = False,
            part_computing_layers: int = 0,
            if_uncertain_params: bool = False,
            uncertain_percentage: float = 0.05,
            if_loss_end: bool = False,

            ) -> None:
        
        if Ns != 1:
            raise ValueError('Ns should be 1')
        
        # initialize circuit_origin & params
        order = {'0': 'NULL', 
                '1': 'BS', '2': 'PS', 
                '3': 'S', '4': 'S2'}
        q = nan
        
        n_modes = Ns + Nm + NL
        layers = 4*(Nm+NL) - 5
        circuit = np.zeros((n_modes, layers), dtype=int)
        circuit_origin = np.zeros((n_modes, layers), dtype=int)
        params = []
        for i in range(n_modes):
            params.append([q]*layers)

        loss = np.ones((n_modes, layers)) * 0.15
        if not if_uncertain_params:

            # S2
            circuit[0][0] = 4
            params[0][0] = [s_r, s_phi]
            circuit_origin[0][0] = 4
            circuit_origin[1][0] = 4

            # S
            for i in range(2, Ns+Nm+NL):
                circuit[i][0] = 3
                if if_odd_even:
                    params[i][0] = [r_k, phi_k*(i%2)]
                else:
                    params[i][0] = [r_k, phi_k]
                circuit_origin[i][0] = 3

            # BS+PS
            for i in range(1,Ns+Nm+NL):
                for j in range(Nm+NL-i):
                    circuit[i][2*i-1+4*j] = 1
                    circuit[i][2*i+4*j] = 2
                    params[i][2*i-1+4*j] = eta
                    params[i][2*i+4*j] = delta_k
                    circuit_origin[i][2*i-1+4*j] = 1
                    circuit_origin[i+1][2*i-1+4*j] = 1
                    circuit_origin[i][2*i+4*j] = 2

                    loss[i][2*i-1+4*j] = loss_amplitude_bs
                    loss[i+1][2*i-1+4*j] = loss_amplitude_bs
        
        else:

            # S2
            circuit[0][0] = 4
            params[0][0] = [s_r, 
                            s_phi]
            circuit_origin[0][0] = 4
            circuit_origin[1][0] = 4

            # S
            for i in range(2, Ns+Nm+NL):
                circuit[i][0] = 3
                if if_odd_even:
                    params[i][0] = [r_k, phi_k*(i%2)]
                else:
                    params[i][0] = [r_k, phi_k]
                circuit_origin[i][0] = 3

            # BS+PS
            for i in range(1,Ns+Nm+NL):
                for j in range(Nm+NL-i):
                    circuit[i][2*i-1+4*j] = 1
                    circuit[i][2*i+4*j] = 2
                    params[i][2*i-1+4*j] = eta*np.random.uniform(1-uncertain_percentage, 1+uncertain_percentage)
                    params[i][2*i+4*j] = delta_k
                    circuit_origin[i][2*i-1+4*j] = 1
                    circuit_origin[i+1][2*i-1+4*j] = 1
                    circuit_origin[i][2*i+4*j] = 2

                    loss[i][2*i-1+4*j] = loss_amplitude_bs
                    loss[i+1][2*i-1+4*j] = loss_amplitude_bs

        loss[:, 0:-1:2] = 0

        # loss
        if not if_add_loss:
            loss = np.zeros((n_modes, layers))
        if if_loss_input:
            loss = loss_input


        self.n_modes = n_modes
        self.Ns = Ns
        self.Nm = Nm
        self.NL = NL
        self.layers = layers
        self.order = order
        self.if_add_loss = if_add_loss
        self.if_loss_end = if_loss_end
        self.part_computing_layers = part_computing_layers
        self.if_part_computing = if_part_computing

        self.circuit = circuit
        self.loss = loss
        self.params = params
        self.circuit_origin = circuit_origin

    
    def compute(self):

        if self.if_part_computing:
            # print('part computing')
            loss_with_end_loss = self.loss
            # print(self.loss)
            if self.if_add_loss and self.if_loss_end:
                loss_with_end_loss[:, self.part_computing_layers-1] = 1.5
                loss_with_end_loss[0:3, self.part_computing_layers-1] = 3
            # print(loss_with_end_loss)   
            a = QumodeWithLoss(self.circuit, loss_with_end_loss, self.params, 
                               self.order, self.part_computing_layers)
        else:
            # print('full computing')
            loss_with_end_loss = self.loss
            if self.if_add_loss and self.if_loss_end:
                loss_with_end_loss[:, -1] = 1.5
                loss_with_end_loss[0:3, -1] = 3
            # print(loss_with_end_loss)
            # print(self.circuit)
            # print(self.params)
            a = QumodeWithLoss(self.circuit, loss_with_end_loss, self.params,
                                 self.order, self.layers)
        self.cov = a.forward().real
        # print(self.cov)
        sigma_s = self.get_sigma_s()
        sigma_m1 = self.get_sigma_m1()
        sigma_m2 = self.get_sigma_m2()
        sigma_m12 = self.get_sigma_m12()
        sigma_sm1 = self.get_sigma_sm1()
        sigma_sm2 = self.get_sigma_sm2()
        sigma_sm12 = self.get_sigma_sm12()  

        self.I2_s_m1 = S(sigma_s) + S(sigma_m1) - S(sigma_sm1)
        self.I2_s_m2 = S(sigma_s) + S(sigma_m2) - S(sigma_sm2)
        self.I2_s_m12 = S(sigma_s) + S(sigma_m12) - S(sigma_sm12)

        self.I3_s_m1_m2 = self.I2_s_m1 + self.I2_s_m2 - self.I2_s_m12

    
    def max_layers(self):
        return self.layers

    def circuit(self):
        return self.circuit
    
    def loss(self):
        return self.loss
    
    def params(self):
        return self.params
    
    def circuit_origin(self):
        return self.circuit_origin
    
    def get_cov(self):
        return self.cov

    def get_sigma_s(self):
        sigma_s = np.zeros((self.Ns*2, self.Ns*2))
        for i in range(2):
            for j in range(2):
                sigma_s[i][j] = self.cov[0+self.n_modes*i][0+self.n_modes*j]
        return sigma_s
    
    def get_sigma_m1(self):
        sigma_m1 = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                sigma_m1[i][j] = self.cov[1+self.n_modes*i][1+self.n_modes*j]
        return sigma_m1
    
    def get_sigma_m2(self):
        sigma_m2 = np.zeros((2*(self.Nm-1), 2*(self.Nm-1)))
        for i in range(2):
            for j in range(2):
                sigma_m2[0+i*(self.Nm-1):self.Nm-1+i*(self.Nm-1), 
                         0+j*(self.Nm-1):self.Nm-1+j*(self.Nm-1)] = self.cov[2+self.n_modes*i:self.Ns+self.Nm+self.n_modes*i, 
                                                                             2+self.n_modes*j:self.Ns+self.Nm+self.n_modes*j]
        return sigma_m2
                
    def get_sigma_m12(self):
        sigma_m12 = np.zeros((2*self.Nm, 2*self.Nm))
        for i in range(2):
            for j in range(2):
                sigma_m12[0+i*self.Nm:self.Nm+i*self.Nm, 
                          0+j*self.Nm:self.Nm+j*self.Nm] = self.cov[1+self.n_modes*i:self.Ns+self.Nm+self.n_modes*i, 
                                                              1+self.n_modes*j:self.Ns+self.Nm+self.n_modes*j]
        return sigma_m12
    
    def get_sigma_sm1(self):
        sigma_sm1 = np.zeros((2*(self.Ns+1), 2*(self.Ns+1)))
        for i in range(2):
            for j in range(2):
                sigma_sm1[0+i*(self.Ns+1):self.Ns+1+i*(self.Ns+1), 
                          0+j*(self.Ns+1):self.Ns+1+j*(self.Ns+1)] = self.cov[0+self.n_modes*i:self.Ns+1+self.n_modes*i, 
                                                                              0+self.n_modes*j:self.Ns+1+self.n_modes*j]
        return sigma_sm1
    
    def get_sigma_sm2(self):
        sigma_sm2 = np.zeros((2*(self.Nm), 2*(self.Nm)))
        for i in range(2):
            for j in range(2):
                sigma_sm2[0+i*self.Nm:self.Nm+i*self.Nm, 
                          0+j*self.Nm:self.Nm+j*self.Nm] = np.delete(
                                                                np.delete(self.cov[0+self.n_modes*i:self.Ns+self.Nm+self.n_modes*i,
                                                                                   0+self.n_modes*j:self.Ns+self.Nm+self.n_modes*j],
                                                                           1, axis=0),
                                                                              1, axis=1)
        return sigma_sm2
    
    def get_sigma_sm12(self):
        sigma_sm12 = np.zeros((2*(self.Ns+self.Nm), 2*(self.Ns+self.Nm)))
        for i in range(2):
            for j in range(2):
                sigma_sm12[0+i*(self.Ns+self.Nm):self.Ns+self.Nm+i*(self.Ns+self.Nm), 
                           0+j*(self.Ns+self.Nm):self.Ns+self.Nm+j*(self.Ns+self.Nm)] = self.cov[0+self.n_modes*i:self.Ns+self.Nm+self.n_modes*i, 
                                                                                               0+self.n_modes*j:self.Ns+self.Nm+self.n_modes*j]
        return sigma_sm12

    def I3(self):
        return self.I3_s_m1_m2.real
    
    def I2_SM12(self):
        return self.I2_s_m12.real
    
    def I2_SM1(self):
        return self.I2_s_m1.real
    
    def I2_SM2(self):
        return self.I2_s_m2.real
