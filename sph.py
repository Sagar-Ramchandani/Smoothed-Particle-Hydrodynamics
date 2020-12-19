'''
The basis of SPH code.
by Mykhailo Lobodin and Sagar Ramchandani
'''
import numpy as np
import matplotlib.pyplot as plt

#Universal Particle Class
class particle: 
    def __init__(self,mass,position,internalEnergy):
        self.mass=mass
        self.pos=position
        self.velocity=0
        self.accln=0
        self.internalEnergy=internalEnergy
        self.pressure=0
        self.neighbours=[]
        self.density=0

    def distance(self,otherParticle):
        relativePosition=self.pos - otherParticle.pos
        dist=np.sqrt(np.dot(relativePosition,relativePosition))
        return dist
    
    def vecDist(self,otherParticle):
        relPos=self.pos - otherParticle.pos
        return relPos
    
'''
Several options for different dimensions kernels,
since need to calculate normalization each time
'''

def M4Kernel1D(s,SmoothL):
    sMag=np.sqrt(np.dot(s,s))
    sUnit=s/sMag
    s=sMag/SmoothL
    if s>=0 and s<=1:
        return (2/(3*SmoothL))*(1-3*s**2/2 +3*s**3/4)
    elif s>1 and s<=2:
        return (2/(3*SmoothL))*((2-s)**3)/4
    elif s>2:
        return 0
    else:
        return 0

def GradM4Kernel1D(s,SmoothL):
    sMag=np.sqrt(np.dot(s,s))
    sUnit=s/sMag
    s=sMag/SmoothL
    if s>=0 and s<=1:
        grad=(2/(3*SmoothL**2))*(-3*s +9*s**2/4)
    elif s>1 and s<=2:
        grad=(2/(3*SmoothL**2))*(-3*(2-s)**2)/4
    else:
        grad=0
    return grad*sUnit

def M4Kernel2D(s,SmoothL):
    sMag=np.sqrt(np.dot(s,s))
    sUnit=s/sMag
    s=sMag/SmoothL
    if s>=0 and s<=1:
        return (10/(7*np.pi*(SmoothL)**2))*(1-3*s**2/2 +3*s**3/4)
    elif s>1 and s<=2:
        return (10/(7*np.pi*(SmoothL)**2))*((2-s)**3)/4
    elif s>2:
        return 0
    else:
        return None

def M4Kernel3D(s,SmoothL):
    sMag=np.sqrt(np.dot(s,s))
    sUnit=s/sMag
    s=sMag/SmoothL
    if s>=0 and s<=1:
        return (1/(np.pi*(SmoothL)**3))*(1-3*s**2/2 +3*s**3/4)
    elif s>1 and s<=2:
        return (1/(np.pi*(SmoothL)**3))*((2-s)**3)/4
    elif s>2:
        return 0
    else:
        return None 

def divergenceVec(vectors,particles,distances,gradKernel,smoothL):
    for j in range(len(particles)):
        result+=(particles[j].mass/particles[j].density)*np.dot(vectors[j],gradKernel(distances[j],smoothL))

def smoothingLength(particles,eta):
    minDistances=[]
    for i in range(len(particles)):
        minDist=np.inf
        for j in range(len(particles)):
            if i==j:
                pass
            else:
                dist=particles[i].distance(particles[j])
                if dist<minDist:
                    minDist=dist
        minDistances.append(minDist)
    h=eta*np.mean(minDistances)
    return h

def neighbourSearch(particles,smoothL):
    #  required resetting previous neighbours
    #  since using append
    for i in range(len(particles)):  
        particles[i].neighbours=[]

    for i in range(1,len(particles)):
        for j in range(i):
            # small const to avoid machine error 
            if particles[i].distance(particles[j])<=2*smoothL+1e-8:
                particles[i].neighbours.append(particles[j])
                particles[j].neighbours.append(particles[i])

def densityEstimation(particles,smoothL, Kernel):
    for i in particles:
        neighbours=i.neighbours
        density=Kernel(0,smoothL) 
        for j in neighbours:
            dist=i.vecDist(j)
            density+=j.mass*Kernel(dist,smoothL)
        i.density=density

def pressureCalc(particles,gamma):
    for i in particles:
        i.pressure=(gamma-1)*i.density*i.internalEnergy

def acclnCalc(particles,gradKernel):
    for i in particles:
        accln=0
        for j in i.neighbours:
            dist=i.vecDist(j)
            accln+=j.mass*(i.pressure/(i.density**2) + j.pressure/(j.density**2))*gradKernel(dist,smoothL)
        i.accln=accln

def workLoop(N,eta,plot,show,dimension=1): 
    if dimension==1:
        Kernel=M4Kernel1D
    elif dimension==2:
        Kernel=M4Kernel2D
    elif dimension==3:
        Kernel=M4Kernel3D
    gamma=5/3
    particles=list(map(lambda x: particle(1/N,x,1), np.linspace(0,1,N)))
    pressureCalc(particles,gamma)
    smoothL=smoothingLength(particles,eta)
    neighbourSearch(particles,smoothL)
    densityEstimation(particles,smoothL,Kernel)
    densities=list(map(lambda x: x.density, particles))
    positions=list(map(lambda x: x.pos, particles))
    if plot:
        plt.plot(positions,densities)
        if show:
            plt.xlabel('Position')
            plt.ylabel('Density')
            plt.title('Density vs Position')
            plt.show()
