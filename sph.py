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
        self.deltaU=0
        self.pressure=0
        self.soundSpeed=0
        self.density=0
        self.divV=0
        self.neighbours=[]

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
    s=sMag/SmoothL
    if s>=0 and s<=1:
        return (2/(3*SmoothL))*(1-3*s**2/2 +3*s**3/4)
    elif s>1 and s<=2:
        return (2/(3*SmoothL))*((2-s)**3)/4
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
    s=sMag/SmoothL
    if s>=0 and s<=1:
        return (10/(7*np.pi*(SmoothL)**2))*(1-3*s**2/2 +3*s**3/4)
    elif s>1 and s<=2:
        return (10/(7*np.pi*(SmoothL)**2))*((2-s)**3)/4
    else:
        return 0

def M4Kernel3D(s,SmoothL):
    sMag=np.sqrt(np.dot(s,s))
    s=sMag/SmoothL
    if s>=0 and s<=1:
        return (1/(np.pi*(SmoothL)**3))*(1-3*s**2/2 +3*s**3/4)
    elif s>1 and s<=2:
        return (1/(np.pi*(SmoothL)**3))*((2-s)**3)/4
    else:
        return 0 

def divergenceV(particles,gradKernel,smoothL):
    for i in particles:
        result=0
        for j in i.neighbours:
            result+=(j.mass/j.density)*np.dot(j.velocity,gradKernel(i.vecDist(j),smoothL))
        i.divV=result

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
        density=Kernel(0,smoothL)*i.mass 
        for j in neighbours:
            dist=i.vecDist(j)
            density+=j.mass*Kernel(dist,smoothL)
        i.density=density

def pressureCalc(particles,gamma):
    for i in particles:
        i.pressure=(gamma-1)*i.density*i.internalEnergy
        i.soundSpeed=np.sqrt(abs(gamma*(gamma-1)*i.internalEnergy))

def acclnCalc(particles,gradKernel,smoothL):
    for i in particles:
        accln=0
        for j in i.neighbours:
            dist=i.vecDist(j)
            accln-=j.mass*(i.pressure/(i.density**2) + j.pressure/(j.density**2))*gradKernel(dist,smoothL)
        i.accln=accln

def eulerIntegration(particles,timeStep):
    for i in particles:
        i.pos=i.pos+i.velocity*timeStep
        i.velocity=i.velocity+i.accln*timeStep
        i.internalEnergy+=i.deltaU*timeStep

#Change in internal energy
def deltaUCalc(particles,gradKernel,smoothL):
    for i in particles:
        dU=0
        for j in i.neighbours:
            dU+=j.mass*(i.velocity-j.velocity)*gradKernel(i.vecDist(j),smoothL)
        i.deltaU=i.pressure/(i.density**2)*dU

def timeStepCalc(CFL,particles,gradKernel,smoothL,epsilon):
    Tmax=[]
    for i in particles:
        TmaxI=CFL*min(smoothL/(smoothL*abs(i.divV) + i.soundSpeed),
                np.sqrt(smoothL/(abs(i.accln)+epsilon)))
        Tmax.append(TmaxI)
    Tglobal=np.min(Tmax)
    return Tglobal

def totalEnergy(particles):
    tE=0
    for i in particles:
        tE+=(i.mass*i.velocity**2)/2 + i.internalEnergy*i.mass
    return tE

def workLoop(N,eta,CFL,epsilon,endTime,nRecordInstants=3,dimension=1): 
    #Kernel determination
    if dimension==1:
        Kernel=M4Kernel1D
        gradKernel=GradM4Kernel1D
    elif dimension==2:
        Kernel=M4Kernel2D
    elif dimension==3:
        Kernel=M4Kernel3D

    #Particle generation
    particles=list(map(lambda x: particle(1/N,x,1), np.linspace(0,1,N)))

    time=0
    legend=[]
    PositionsT=[]
    DensitiesT=[]
    VelocitiesT=[]
    TE=[]
    timeT=[]
    recordInstants=np.linspace(0,(nRecordInstants+1)/nRecordInstants*endTime,nRecordInstants+2)
    print(recordInstants)
    recIndex=1 
    toRecord=True
    while time<=endTime:
            
        #Density Estimation
        smoothL=smoothingLength(particles,eta)
        neighbourSearch(particles,smoothL)
        densityEstimation(particles,smoothL,Kernel)
            
        #Parameter Calculation
        gamma=5/3
        pressureCalc(particles,gamma)
        acclnCalc(particles,gradKernel,smoothL)
        deltaUCalc(particles,gradKernel,smoothL)
        timeStep=timeStepCalc(CFL,particles,gradKernel,smoothL,epsilon)
        
        #Plotting
        if toRecord:
            PositionsT.append(list(map(lambda x: x.pos, particles)))
            DensitiesT.append(list(map(lambda x: x.density, particles)))
            VelocitiesT.append(list(map(lambda x: x.velocity, particles)))
            TE.append(totalEnergy(particles))
            legend.append('t='+str(round(time,2)))
            timeT.append(round(time,2))
            toRecord=False

        if time+timeStep>recordInstants[recIndex]:
            timeStep=recordInstants[recIndex]-time
            if recIndex+1<len(recordInstants):
                recIndex+=1
            toRecord=True

        #Integration over time
        eulerIntegration(particles,timeStep)
        time+=timeStep
        
    for pos,den in zip(PositionsT,DensitiesT):
        plt.plot(pos,den)
    plt.xlabel('Position')
    plt.ylabel('Density')
    plt.legend(legend)
    plt.show()
    for pos,vel in zip(PositionsT,VelocitiesT):
        plt.plot(pos,vel)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.legend(legend)
    plt.show()
    plt.xlabel('Time')

    plt.plot(timeT,TE)
    plt.show()
workLoop(100,5,.5,1e-10,0.3,3)
#for eta in range(2,5):
#    workLoop(100,eta,0.5,0,0.3)
#plt.legend(range(2,5))
#plt.show()
#for cfl in np.linspace(0.1,1,4):
#    workLoop(100,4,cfl,0.0000000001,0.35)
#    print(cfl)
#plt.legend(np.linspace(0.1,1,4))
#plt.show()
