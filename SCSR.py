import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class SingleChainSlideRing:
    def __init__(self, n, Z, k):
        '''
        Initialize the slide-ring chain
        n: number of Kuhn segments per strand
        Z: number of strands
        k: number of beads in the virtual spring, k = [kx,ky,kz]
        '''
        if type(n) != int or type(Z) != int or type(k) != list:
            raise ValueError('n and Z should be integers, k should be a list')
        if len(k) != 3:
            raise ValueError('k should be a list of length 3')
        if n < 1 or Z < 1 or k[0] < 1 or k[1] < 1 or k[2] < 1:
            raise ValueError('n, Z and k should be positive integers')

        self.N = n
        self.Z = Z

        self.lam = [1,1,1]

        # Array of coordinates of the slide-rings
        self.cords = np.zeros((Z+1,3))

        # Generate a dirichlet distribution for the number of Kuhn segments in each strand
        indices = np.random.choice(np.arange(1,n*Z),Z-1,replace=False)
        indices = np.sort(indices)
        self.Nlist = np.insert(indices,Z-1,n*Z) - np.insert(indices,0,0)
        # self.Nlist = n * np.ones(Z)

        # Generate the coordinates of the slide-rings
        c = np.zeros(3)
        self.cords[0,:] = c
        for i in range(Z):
            vec = np.random.multivariate_normal(np.zeros(3),self.Nlist[i]/3*np.eye(3))
            c += vec
            self.cords[i+1,:] = c

        # Generate the attachment points
        self.k = np.array(k)
        self.attachments = np.zeros((Z-1,3))
        for i in range(Z-1):
            self.attachments[i,:] = self.cords[i+1,:] + np.random.multivariate_normal(np.zeros(3),np.diag(k)/3)
        
            
    def connect_matrix(self):
        '''
        Generate the connection matrix M and the vector b for the linear system
        '''
        M = np.zeros((3,self.Z-1,self.Z-1))
        
        for i in range(self.Z-1):
            M[:,i,i] += 1/self.k
            M[:,i,i] += 1/self.Nlist[i]
            M[:,i,i] += 1/self.Nlist[i+1]
        for i in range(self.Z-2):
            M[:,i,i+1] += -1/self.Nlist[i+1]
            M[:,i+1,i] += -1/self.Nlist[i+1]
        
        b = np.zeros((3,self.Z-1))
        
        for i in range(self.Z-1):
            b[:,i] += self.attachments[i,:]/self.k
        b[:,0] += self.cords[0,:]/self.Nlist[0]
        b[:,-1] += self.cords[-1,:]/self.Nlist[-1]
        return M,b
    
    # def connect_matrix_epsilon(self, epsilon):
    #     '''
    #     Generate the connection matrix M with a small perturbation epsilon
    #     '''
    #     M = np.zeros((3,self.Z-1,self.Z-1))
        
    #     for i in range(self.Z-1):
    #         M[:,i,i] += 1/self.k
    #         M[:,i,i] += (1+epsilon)/self.Nlist[i]
    #         M[:,i,i] += (1+epsilon)/self.Nlist[i+1]
    #     for i in range(self.Z-2):
    #         M[:,i,i+1] += -(1+epsilon)/self.Nlist[i+1]
    #         M[:,i+1,i] += -(1+epsilon)/self.Nlist[i+1]
    
    #     return M
    
    def solve(self):
        '''
        Solve the linear system to find the mean positions of the slide-rings
        '''
        M,b = self.connect_matrix()
        
        # Solve the coordinates of the slide-rings by solving the linear system
        self.cords[1:-1,:] = np.linalg.solve(M,b).T

    def freeE(self):
        '''
        Calculate the free energy of the slide-ring chain
        '''
        # Elastic free energy of the real network strands
        strands = self.strands()
        F = 0
        for i in range(self.Z):
            F += 3/2 * np.sum(strands[i]**2) / self.Nlist[i] 
            F += 3/2 * np.log(self.Nlist[i])

        # Elastic free energy of the virtual springs
        for i in range(self.Z-1):
            
            F += 3/2*np.sum((self.cords[i+1,:] - self.attachments[i,:])**2 / self.k)
        
        # Free energy from the coupling of the fluctuations of neighboring strands
        M,_ = self.connect_matrix()
        _, logabsdet = np.linalg.slogdet(M)
        F += 1/2 * np.sum(logabsdet)
        return F
    
    def stress(self):
        '''
        Calculate the stress tensor (per strand) of the slide-ring chain
        '''
        
        S = np.zeros(3)
        # Mp = self.connect_matrix_epsilon(1e-6)
        # Mm = self.connect_matrix_epsilon(-1e-6)
        strands = self.strands()

        # Stress from the elastic energy of the real network strands
        for i in range(self.Z):
            S += 3 * strands[i,:]**2/self.Nlist[i]

        # # Stress from the coupling of the fluctuations of neighboring strands
        # _, logabsdet_p = np.linalg.slogdet(Mp)
        # _, logabsdet_m = np.linalg.slogdet(Mm)
        # S += (logabsdet_p - logabsdet_m)/(2e-6)
        return S / self.Z
    
    def strands(self):
        '''
        Return the vectors of the strands
        '''
        return self.cords[1:,:] - self.cords[:-1,:]
    
    def affine_deform(self, lam): 
        '''
        Apply an affine deformation to the slide-ring chain
        '''
        if type(lam) != list:
            raise ValueError('lam should be a list')
        if len(lam) != 3:
            raise ValueError('lam should be a list of length 3')
        self.cords = self.cords / self.lam * lam
        self.attachments = self.attachments / self.lam * lam

        self.lam = lam
        
    def mc_step(self):
        '''
        Perform a Monte Carlo step by moving a bead from one strand to another, and accept the move with the Metropolis criterion
        '''
        self.solve()
        F = self.freeE()
        # S = self.stress()

        ind = np.random.randint(self.Z-1)
        direction = np.random.randint(2)

        if direction == 0:
            if self.Nlist[ind+1] == 1:
                return False, F
            self.Nlist[ind] += 1
            self.Nlist[ind+1] -= 1
        else:
            if self.Nlist[ind] == 1:
                return False, F
            self.Nlist[ind] -= 1
            self.Nlist[ind+1] += 1
        
        self.solve()
        Fnew = self.freeE()
        # Snew = self.stress()
        # if Fnew < F:
        if np.random.rand() < np.exp(-(Fnew-F)):
            return True, Fnew
        else:
            if direction == 0:
                self.Nlist[ind] -= 1
                self.Nlist[ind+1] += 1
            else:
                self.Nlist[ind] += 1
                self.Nlist[ind+1] -= 1
            return False, F
        
    def change_k(self, k):
        '''
        Change the number of Kuhn segments in the virtual springs
        '''
        if type(k) != list:
            raise ValueError('k should be a list')
        if len(k) != 3:
            raise ValueError('k should be a list of length 3')
        
        self.k = np.array(k)

    def equilibrate(self, nsteps, verbose=True):
        '''
        Equilibrate the slide-ring chain by performing nsteps Monte Carlo steps
        '''
        if type(nsteps) != int:
            raise ValueError('nsteps should be an integer')
        if nsteps < 1:
            raise ValueError('nsteps should be a positive integer')
        freeE = np.zeros(nsteps)
        if verbose:
            for i in tqdm(range(nsteps)):
                freeE[i] = self.mc_step()[1]
        else:
            for i in range(nsteps):
                freeE[i] = self.mc_step()[1]

        return freeE

    def production(self, nsteps, verbose=True):
        '''
        Perform nsteps Monte Carlo steps and calculate the average stress tensor
        '''
        if type(nsteps) != int:
            raise ValueError('nsteps should be an integer')
        if nsteps < 1:
            raise ValueError('nsteps should be a positive integer')

        stress = np.zeros(3)
        stress2 = np.zeros(3)

        if verbose:
            for i in tqdm(range(nsteps)):
                self.mc_step()
                
                curstress = self.stress()
                # print(curstress)
                stress += curstress
                stress2 += curstress**2
        else:
            for i in range(nsteps):
                self.mc_step()
                
                curstress = self.stress()
                # print(curstress)
                stress += curstress
                stress2 += curstress**2
        
        stress /= nsteps
        stress2 /= nsteps

        # stress /= self.Z
        # stress2 /= self.Z

        error = np.sqrt(stress2 - stress**2) / np.sqrt(nsteps)

        return stress, error
    

    def visualize(self):
        '''
        Visualize the slide-ring chain
        '''
        ax = plt.figure().add_subplot(projection='3d')
        ax.set_box_aspect((np.ptp(np.concatenate((self.cords[:,0],self.attachments[:,0]))),
                           np.ptp(np.concatenate((self.cords[:,1],self.attachments[:,1]))),
                           np.ptp(np.concatenate((self.cords[:,2],self.attachments[:,2])))))
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.plot(self.cords[:,0],self.cords[:,1],self.cords[:,2],c='b')
        ax.plot(self.attachments[:,0],self.attachments[:,1],self.attachments[:,2],'x',c='r')
        ax.plot(self.cords[0,0],self.cords[0,1],self.cords[0,2],'o',c='r')
        ax.plot(self.cords[-1,0],self.cords[-1,1],self.cords[-1,2],'o',c='r')
        ax.scatter(self.cords[1:-1,0],self.cords[1:-1,1],self.cords[1:-1,2],'o', 
                   facecolors='none', edgecolors='orange', s=25, linewidths=1.5,depthshade=False)

        for i in range(self.Z-1):
            ax.plot([self.cords[i+1,0],self.attachments[i,0]],
                    [self.cords[i+1,1],self.attachments[i,1]],
                    [self.cords[i+1,2],self.attachments[i,2]],c='green')
        plt.show()