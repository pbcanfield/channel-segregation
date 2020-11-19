from neuron import h
import os
import matplotlib.pyplot as plt
import numpy as np
import time as t
from scipy.signal import find_peaks
import PSO

#First compile our modfiles.
if os.system('nrnivmodl modfiles/') == 0:
    print("Compilation Succeeded.")
else:
    print("Compilation Failed.")


h.load_file('stdrun.hoc')

#Define simumulation parameters.
SIMRUNTIME = 1500
DELAY      = 400
INJTIME    = 500
I_INJ      = 0.1

h.tstop    = SIMRUNTIME
h.v_init   = -75

#Now set up vectors to record the membrane potential and the time.
potential = h.Vector()
time      = h.Vector()


tonic_soma = h.Section(name = 'tonic_soma')

tonic_iclamp = h.IClamp(tonic_soma(0.5))
tonic_iclamp.dur   = INJTIME
tonic_iclamp.amp   = I_INJ
tonic_iclamp.delay = DELAY

#We will make a super simple one compartement tonic spiking cell.
tonic_soma.nseg = 1
tonic_soma.L    = 25
tonic_soma.diam = 24.75

#Now insert the two segregated modules into the cell.
#First insert the passive module.
tonic_soma.insert('leakCA3')
tonic_soma.insert('hdCA3')

#Now insert the spiking module.
tonic_soma.insert('natCA3')
tonic_soma.insert('kdrCA3')

#record time and membrane potential for tonic cell.
#Record the membrane variables.
time.record(h._ref_t) 
potential.record(tonic_soma(0.5)._ref_v)


def run_tonic_model(amp, gna, gkdr, gh, mseg, nseg, lseg):

    tonic_iclamp.amp = amp

    tonic_soma.glbar_leakCA3 = 1.53*3e-5
    tonic_soma.gbar_natCA3   = gna
    tonic_soma.mseg_natCA3   = mseg
    tonic_soma.gbar_kdrCA3   = gkdr
    tonic_soma.nseg_kdrCA3   = nseg
    tonic_soma.ghdbar_hdCA3  = gh
    tonic_soma.lseg_hdCA3    = lseg
    
    
    h.run()

#Set the default parameters for the simulations.

default_tonic = [I_INJ, 0.065, 0.015, 1.0e-5, -52.5, -55, -62.4]

observed_trace = potential.as_numpy()
observed_time = time.as_numpy()

run_tonic_model(default_tonic[0], 
                default_tonic[1],
                default_tonic[2],
                default_tonic[3],
                default_tonic[4],
                default_tonic[5],
                default_tonic[6] )

#Plot membrane potential of tonic cell.
def plot_membrane():
    plt.close()
    plt.figure(figsize = (20,5))
    plt.plot(time, potential)
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential')
    plt.show()


#Given the votage trace provided in the potential vector we can calculate the summary statistics of the spiker.
#The statistics for this model will be spike number, resting potential, average spike height, and average spike width.
def calculate_summary_statistics():
    trace = potential.as_numpy()

    zero_crossings = np.where(np.diff(np.sign(trace)))[0] #Detect when the voltage trace crosses 0.
    
    spike_width = 0
    for i in range(0,len(zero_crossings), 2):
        spike_width += zero_crossings[i + 1] - zero_crossings[i]
    
    spike_num = (len(zero_crossings) / 2)
    
    if spike_num != 0:
        spike_width /= spike_num
    
    resting = trace[-1]
    
    peaks, _ =  find_peaks(trace,0)
    
    #Calculate average height.
    average_spike_height = np.mean(_['peak_heights'])
    
    return [spike_num, resting, average_spike_height, spike_width]

observed = calculate_summary_statistics()
plot_membrane()

#Now define min and max parameter values.
ranges = [[0.01,0.3],[0.01,0.3]]


#for now lets just use gna and gkdr.
def cost_function(params):
    #given the parameter space evaluate the cost of the particle.
    run_tonic_model(default_tonic[0], 
                    params[0]       ,
                    params[1]       ,
                    default_tonic[3],
                    default_tonic[4],
                    default_tonic[5],
                    default_tonic[6]) 

    statistics = calculate_summary_statistics()
    #generate the summary statistics for the model and calculate the LMS based on observed statistics.
    LMS = 0
    for i in range(4):
        LMS = (statistics[i] - observed[i]) ** 2
    
    return LMS

#Create an optimizer object with size 100, a list of para
optimizer = PSO.PSOOptimizer(10,ranges,0.1,0.1,0.1)

optimizer.evaluateCost(cost_function)
optimizer.printParameters()

print("\n")

start = t.time()
for i in range(10):
    
    optimizer.evaluateCost(cost_function)
    optimizer.optimizationStep()
        
    print('iteration: ' + str(i) + ' took: ' + str(t.time()-start) + ' seconds')
    start = t.time()

optimizer.evaluateCost(cost_function)
optimizer.printParameters()


run_tonic_model(default_tonic[0], 
                    optimizer.bestParams[0]       ,
                    optimizer.bestParams[1]       ,
                    default_tonic[3],
                    default_tonic[4],
                    default_tonic[5],
                    default_tonic[6]) 

fig = plt.figure(figsize=(7,5))

# plot observation
plt.plot(observed_time, observed_trace, lw=2, label='observation')

# simulate and plot samples
plt.plot(time.as_numpy(), potential.as_numpy(), '--', lw=2, label='posterior sample')

plt.xlabel('time (ms)')
plt.ylabel('voltage (mV)')

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.3, 1), 
          loc='upper right')

