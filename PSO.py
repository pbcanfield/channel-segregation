import numpy
'''
Includes a population of particles that are going to do the
Optimization.

The particles need to operate in n-dimensional space.

Particles have a locally known best position.
'''

class Particle:
    
    localBest = None
    position = []
    velocity = []
    
    def __init__(self, pos, vel):
        self.position = pos.copy()
        self.velocity = vel.copy()
        self.localBest = self.position

    # Update the particles current position based on it's
    # Velocity.
    def updatePosition(self):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]

    def setPosition(self, position):
        self.position = position.copy()

    def setVelocity(self, velocity):
        self.velocity = velocity.copy()        

'''
Includes the actual PSO algorithm. Takes an initial population size, 
inputs, then runs particle swarm optimzation given a cost function, 
which is evaluated outside of the optimizer.

The parameters:
    1) population size
    2) A list of lower and upper bounds for each parameter
    3) The global influence parameter
    4) The local influence parameter
    5) The current velocity parameter

Therefore, the parameters list is a 2D list, with each dimension containing
a lower and upper bound for the particle to optimize within.

Each 2D sublist is stored with the lower bound first and the upper bound second.
'''
class PSOOptimizer:
    particles = []
    globalBest = None
    bestParams = None
    
    globalWeight = None
    localWeight = None
    velocityWeight = None

    def __init__(self, populationSize, parameters, gw, lw, vw):
        #The population size is the length of the parameters list.
        #Now initilize the population.  
        self.populationSize = populationSize
        self.globalWeight = gw
        self.localWeight = lw
        self.velocityWeight = vw

        for i in range(populationSize):
            #Initilize the position based on bounds provided to the
            #opimizer.
            pos = [(numpy.random.uniform(bound[0],bound[1])) for bound in parameters]
            
            #Get the absolute value of the parameter range.
            velocityBound = [abs(bound[1] - bound[0]) for bound in parameters]
            vel = [(numpy.random.uniform(-bound,bound)) for bound in velocityBound]

            self.particles.append(Particle(pos,vel))
            
    #This just prints out the parameters from the population.
    def printParameters(self):
        print("The best global cost is: ", self.globalBest)
        print("The best global position is: ", self.bestParams)

    #Evaluates the particles based on an external cost function.
    def evaluateCost(self,costFunction):
        #Evaluate each particle in terms of its cost. Also find the best one.
        if len(self.particles) == 0:
                print("There are no particles in the population.")
                return

        #The population has candidates.
        #There could be an exception in the cost function itself
        #if there is then set the cost to a huge amount
        cost = 0
        try:
            cost = costFunction(self.particles[0].position)
        except:
            print("Internal error, setting cost to max")
            cost = float('inf')

        if cost < costFunction(self.particles[0].localBest):
            self.particles[0].localBest = self.particles[0].position

        self.globalBest = cost
        self.bestParams = self.particles[0].position

        for p in self.particles[0:]:
            cost = costFunction(p.position)
            
            localCost = costFunction(p.localBest)
            if cost < localCost:
                p.localBest = p.position
                if cost < globalBest:
                    self.globalBest = cost
                    self.bestParams = p.position
            else:
                #Check if the localBest is less than the global best.
                localCost = costFunction(p.localBest)
                if localCost < self.globalBest:
                    self.globalBest = localCost
                    self.bestParams = p.position

    #Calculates an optimazation step.
    def optimizationStep(self):
        #For each particle.
        for p in self.particles:
            
            #Cylcle through each dimenstion in the parameter space.
            #Update the velocity value.
            for index,vel,pos,lb,gb in zip(range(len(p.velocity)),p.velocity,p.position,p.localBest,self.bestParams):
                #Pick some random values to move the particle.
                rl = numpy.random.uniform(0,1)
                rg = numpy.random.uniform(0,1)
                newVel = vel * self.velocityWeight + self.localWeight * rl * (lb - pos) + self.globalWeight * rg * (gb - pos)

                p.velocity[index] = newVel

            p.updatePosition()
