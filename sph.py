'''
The basis of SPH code.
by Mykhailo Lobodin and Sagar Ramchandani
'''
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
from numba import jit
from numba.typed import List
from time import time as timer

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

'''
Several options for different dimensions kernels,
since need to calculate normalization each time
'''
@jit(nopython=True)
def M4Kernel1D(s,SmoothL):
    s=abs(s)/SmoothL
    if s>=0 and s<=1:
        return (2/(3*SmoothL))*(1-3*s**2/2 +3*s**3/4)
    elif s>1 and s<=2:
        return (2/(3*SmoothL))*((2-s)**3)/4
    else:
        return 0

@jit(nopython=True)
def GradM4Kernel1D(s,SmoothL):
    sMag=abs(s)
    sUnit=s/sMag
    s=sMag/SmoothL
    if s>=0 and s<=1:
        grad=(2/(3*SmoothL**2))*(-3*s +9*s**2/4)
    elif s>1 and s<=2:
        grad=(2/(3*SmoothL**2))*(-3*(2-s)**2)/4
    else:
        grad=0
    return grad*sUnit

@jit(nopython=True)
def M4Kernel2D(s,SmoothL):
    sMag=np.sqrt(np.dot(s,s))
    s=sMag/SmoothL
    if s>=0 and s<=1:
        return (10/(7*np.pi*(SmoothL)**2))*(1-3*s**2/2 +3*s**3/4)
    elif s>1 and s<=2:
        return (10/(7*np.pi*(SmoothL)**2))*((2-s)**3)/4
    else:
        return 0

@jit(nopython=True)
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
            result+=(j.mass/j.density)*np.dot(j.velocity,gradKernel(distanceCalc(i.pos,j.pos,vector=True),smoothL))
        i.divV=result

@jit(nopython=True)
def distanceCalc(pos1,pos2,vector=False,periodic=True):
    #Constant to avoid zero distance between particles at extremes
    nonZeroConstant=1e-8
    if vector:
        alpha=pos1-pos2
        if periodic:
            if abs(alpha)>intervalLength-abs(alpha):
                if alpha>0:
                    return abs(alpha)-intervalLength+nonZeroConstant
                else:
                    return intervalLength-abs(alpha)+nonZeroConstant
            else:
                return alpha+nonZeroConstant
        else:
            return alpha+nonZeroConstant
    else:
        alpha=abs(pos1-pos2)
        if periodic:
            return min(alpha,intervalLength-alpha+nonZeroConstant)
        else:
            return alpha+nonZeroConstant
 

@jit(nopython=True)
def smoothingLength(distance,eta):
    minDistances=[]
    for i in distance:
        minDist=np.inf
        for j in distance:
            dist=distanceCalc(i,j)
            if dist==0:
                pass
            else:
                if dist<minDist:
                    minDist=dist
        minDistances.append(minDist)
    minDistances=np.asarray(minDistances)
    h=eta*np.mean(minDistances)
    return h

@jit(nopython=True)
def neighbourSearch(N,distance,smoothL):
    neighbours=List()

    for i in range(N):
        neighboursI=[]
        for j in range(N):
            if i!=j:
                if distanceCalc(distance[i],distance[j])<=2*smoothL:
                    neighboursI.append(j)
        neighbours.append(np.asarray(neighboursI))
    return neighbours

@jit(nopython=True)
def densityEstimation(N,neighbours,mass,distance,smoothL):
    Kernel=M4Kernel1D
    densities=List()
    for i in range(N):
        neighI=neighbours[i]
        density=Kernel(0,smoothL)*mass[i]
        for j in neighI:
            dist=distanceCalc(distance[i],distance[j],vector=True)
            density+=mass[j]*Kernel(dist,smoothL)
        densities.append(density)
    return densities

def pressureCalc(particles,gamma):
    for i in particles:
#        i.pressure=i.density*i.internalEnergy
#        i.soundSpeed=np.sqrt(i.internalEnergy)

        i.pressure=(gamma-1)*i.density*i.internalEnergy
        i.soundSpeed=np.sqrt(abs(gamma*(gamma-1)*i.internalEnergy))

@jit(nopython=True)
def acclnCalc(N,neighbours,mass,position,velocity,pressure,soundSpeed,density,gradKernel,smoothL,alpha=1):
    acceleration=List()
    for i in range(N):
        accln=0.
        for j in neighbours[i]:
            relDist=distanceCalc(position[i],position[j],vector=True)
            relDistMag=abs(relDist)
            accln-=mass[j]*(pressure[i]/(density[i]**2) + pressure[j]/(density[j]**2))*gradKernel(relDist,smoothL)
            #Viscosity term
            relVel=velocity[i]-velocity[j]
            if relVel*relDist>0:
                pass
            else:
                beta=2*alpha
                avgDensity=(density[i]+density[j])/2
                vSig=soundSpeed[i]+soundSpeed[j]-beta*relVel*relDist/relDistMag
                accln+=mass[j]*gradKernel(relDist,smoothL)*alpha*vSig*relVel*relDist/(avgDensity*relDistMag)
        acceleration.append(accln)
    return acceleration

#Change in internal energy
@jit(nopython=True)
def deltaUCalc(N,neighbours,mass,position,velocity,pressure,soundSpeed,density,gradKernel,smoothL,alpha=1):
    deltaU=List()
    for i in range(N):
        dU=0
        dUv=0
        for j in neighbours[i]:
            relDist=distanceCalc(position[i],position[j],vector=True)
            relDistMag=abs(relDist)
            dU+=mass[j]*(velocity[i]-velocity[j])*gradKernel(relDist,smoothL)
            #Viscosity term
            relVel=velocity[i]-velocity[j]
            if relVel*relDist>0:
                pass
            else:
                beta=alpha*2
                avgDensity=(density[i]+density[j])/2
                vSig=soundSpeed[i]+soundSpeed[j]-beta*relVel*relDist/relDistMag
                dUv-=mass[j]*relDist/relDistMag*gradKernel(relDist,smoothL)*(
                        alpha*vSig*((relVel*relDist)**2)/(2*avgDensity*relDistMag**2))
        deltaU.append(pressure[i]/(density[i]**2)*dU+dUv)
    return deltaU

def timeStepCalc(CFL,particles,gradKernel,smoothL,epsilon,alpha=1):
    dTs=[]
    divergenceV(particles,gradKernel,smoothL)
    for i in particles:
        beta=2*alpha
        dT=CFL*min(smoothL/(smoothL*abs(i.divV) + i.soundSpeed),
                np.sqrt(smoothL/(abs(i.accln)+epsilon)),
                smoothL/((1+1.2*alpha)*i.soundSpeed+(1+1.2*beta)*smoothL*abs(i.divV)))
        dTs.append(dT)
    Tglobal=np.min(dTs)
    return Tglobal

def eulerIntegration(particles,timeStep):
    for i in particles:
        i.pos=i.pos+i.velocity*timeStep
        #Periodic conditions
        if i.pos>boundR:
            i.pos=boundL+(i.pos-boundR)
        if i.pos<boundL:
            i.pos=boundR-(boundL-i.pos)
        i.velocity=i.velocity+i.accln*timeStep
        i.internalEnergy+=i.deltaU*timeStep

def totalEnergy(particles):
    tE=0
    for i in particles:
        tE+=(i.mass*i.velocity**2)/2 + i.internalEnergy*i.mass
    return tE

def standardDeviation(density):
    return np.std(density)

def sodShockTube(densityL,densityR,pressureL,pressureR,shockPos,N):
    #Auxillary particles
    N=N*2
    particles=list(map(lambda x: particle(densityL/N,x,pressureL/(gamma-1)) if x<shockPos 
        else particle(densityR/N,x,pressureR/(gamma-1)), np.linspace(-0.5,1.5,N)))
    return particles

def glass(N,boundL,boundR,mass,internalEnergy):
    positions=(boundR-boundL)*np.random.random_sample((N,))-boundL
    particles=list(map(lambda x: particle(mass,x,internalEnergy), positions))
    return particles

def sedovBlast(N,boundL,boundR,boundCentre,density,UHigh,ULow):
    particles=list(map(lambda x: particle(intervalLength*density/N,x,UHigh) if abs(x)<=boundCentre
        else particle(intervalLength*density/N,x,ULow), np.linspace(boundL,boundR,N)))
    return particles


def workLoop(N,eta,CFL,epsilon,endTime,alpha=1,nRecordInstants=3,dimension=1,particles=None,isoThermal=False,pltAxis=None): 
    #Kernel determination 
    if dimension==1:
        Kernel=M4Kernel1D
        gradKernel=GradM4Kernel1D
    elif dimension==2:
        Kernel=M4Kernel2D
    elif dimension==3:
        Kernel=M4Kernel3D

    #Particle generation
    if type(particles)==type(None):
        particles=list(map(lambda x: particle(1/N,x,1), np.linspace(0,1,N)))
    else:
        N=len(particles)

    time=0
    legend=[]
    PositionsT=[]
    PressureT=[]
    DensitiesT=[]
    VelocitiesT=[]
    TE=[]
    SD=[]
    timeT=[]
    times=[]
    recordInstants=np.linspace(0,(nRecordInstants+1)/nRecordInstants*endTime,nRecordInstants+2)
    recIndex=1 
    toRecord=True
    onePercentSDTime=0
    onePercentTimeRecorded=False
    while time<=endTime:

        #Density Estimation
        mass=np.asarray([i.mass for i in particles])
        position=np.asarray([i.pos for i in particles])
        smoothL=smoothingLength(position,eta)
        
        NS=neighbourSearch(N,position,smoothL)
        for i in range(len(NS)):
            particles[i].neighbours=[particles[j] for j in NS[i]]

        densities=densityEstimation(N,NS,mass,position,smoothL)
        for i in range(N):
            particles[i].density=densities[i]

        #Parameter Calculation
        gamma=5/3
        pressureCalc(particles,gamma)

        velocity=np.asarray([i.velocity for i in particles])
        pressure=np.asarray([i.pressure for i in particles])
        soundSpeed=np.asarray([i.soundSpeed for i in particles])

        acceleration=acclnCalc(N,NS,mass,position,velocity,pressure,soundSpeed,densities,gradKernel,smoothL,alpha)
        for i in range(N):
            particles[i].accln=acceleration[i]

        if not isoThermal:
            deltaUs=deltaUCalc(N,NS,mass,position,velocity,pressure,soundSpeed,densities,gradKernel,smoothL,alpha)
            for i in range(N):
                particles[i].deltaU=deltaUs[i]

        timeStep=timeStepCalc(CFL,particles,gradKernel,smoothL,epsilon,alpha)

        SD.append(standardDeviation(densities))
        times.append(round(time,2))

        if standardDeviation(densities)/np.mean(densities)<0.01 and not onePercentTimeRecorded:
            onePercentSDTime=time
            onePercentTimeRecorded=True

        #Plotting
        if toRecord:
            PositionsT.append(list(map(lambda x: x.pos, particles)))
            PressureT.append(list(map(lambda x: x.pressure, particles)))
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

        print(str(time/endTime*100)+' %')
    

    if type(pltAxis)==type(None):
        #f,(ax1,ax2,ax3)=plt.subplots(3)
        f,ax1=plt.subplots(1)
        showPlot=True
    else:
        #ax1,ax2,ax3=pltAxis
        ax1=pltAxis
        showPlot=False

    for pos,den in zip(PositionsT,DensitiesT):
        den=[x for y,x in sorted(zip(pos,den))]
        pos=sorted(pos)
        ax1.plot(pos,den)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Density')
    ax1.set_xlim(boundL,boundR)
    if showPlot:
        ax1.legend(legend)
        plt.show()
      
#    for pos,pre in zip(PositionsT,PressureT):
#        ax2.plot(pos,pre)
#    ax2.set_xlabel('Position')
#    ax2.set_ylabel('Pressure')
#    ax2.set_xlim(boundL,boundR)
#    if showPlot:
#       ax2.legend(legend)
#
#    for pos,vel in zip(PositionsT,VelocitiesT):
#        ax3.plot(pos,vel)
#    ax3.set_xlabel('Time')
#    ax3.set_ylabel('Velocity')
#    ax3.set_xlim(boundL,boundR)
#    if showPlot:
#        ax3.legend(legend)
#        plt.show()

    plt.plot(times,SD)
    plt.show()
    plt.plot(timeT,TE)
    plt.show()
    return (onePercentSDTime)

#gamma=1.4
#N=100
#boundL,boundR=0,1
#intervalLength=abs(boundR-boundL)
#workLoop(N,5,.1,1e-10,5,nRecordInstants=3,alpha=5,particles=p,isoThermal=True)

#f,ax1=plt.subplots(1)
#onePercent=[]
#while len(onePercent)<5:
#    p=glass(N,0,1,0.01,1)
#    t=workLoop(N,5,.1,1e-10,5,nRecordInstants=3,alpha=5,particles=p,isoThermal=True,pltAxis=ax1)
#    onePercent.append(t)
#print(onePercent)
#print(np.mean(onePercent))
#print(np.std(onePercent))

N=100
boundL,boundR=-10,10
intervalLength=abs(boundR-boundL)
p=sedovBlast(N,boundL,boundR,1,1,250,2.5)
workLoop(N,5,.1,1e-10,.6,nRecordInstants=6,alpha=0,particles=p)
