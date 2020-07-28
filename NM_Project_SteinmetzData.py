# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:58:28 2020

@author: TheBeast
"""

# =============================================================================
#                         Clear Variables
# =============================================================================

# Clear variables before running
from IPython import get_ipython
get_ipython().magic('reset -sf')

# =============================================================================
#                        Import Libraries
# =============================================================================


import numpy as np
import os
from scipy import signal
import os, requests
import plotly.express as px
from plotly.offline import plot

from matplotlib import rcParams 
from matplotlib import pyplot as plt
rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] =15
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True


# =============================================================================
#                       Import the Data
# =============================================================================


fname = []
for j in range(4):
  # fname.append('steinmetz_part%d.npz'%j)
  if j==0:
    fname.append('steinmetz_lfp.npz')
  else:
    fname.append('steinmetz_part%d.npz'%j)

url = ["https://osf.io/kx3v9/download"]#["https://osf.io/agvxh/download"]
url.append("https://osf.io/agvxh/download")
url.append("https://osf.io/uv3mw/download")
url.append("https://osf.io/ehmw2/download")
# url.append("https://osf.io/kx3v9/download")

for j in range(len(url)):
  if not os.path.isfile(fname[j]):
    try:
      r = requests.get(url[j])
    except requests.ConnectionError:
      print("!!! Failed to download data !!!")
    else:
      if r.status_code != requests.codes.ok:
        print("!!! Failed to download data !!!")
      else:
        with open(fname[j], "wb") as fid:
          fid.write(r.content)



# =============================================================================
#                        Data Parameters
# =============================================================================

fs=100



# =============================================================================
#                         Sorting Parts of Data
# =============================================================================


#Here we pull out the data and store it in 'alldata'
alldat_lfp = np.array([])
alldat = np.array([])

alldat_lfp = np.hstack((alldat, np.load('steinmetz_lfp.npz', allow_pickle=True)['dat']))

for j in range(1,len(fname)):
  if j !=0: #CHANGE BACK TO 0
    alldat = np.hstack((alldat, np.load('steinmetz_part%d.npz'%j, allow_pickle=True)['dat']))




# Get all the mouse names
all_mouse_names =[]  
for ii in range(0,np.size(alldat)):
    all_mouse_names.append(alldat[ii]['mouse_name'])

#Gets all the unique mouse names
all_mouse_names_unique= list(set(all_mouse_names))



## _____________________Define Brain Regions _________________________________

# groupings of brain regions
regions = ["Vis Ctx", "Thal", "Hippo","Motor" "other ctx", "midbrain", "basal ganglia", "cortical subplate", "other"]
brain_groups = [["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"], # visual cortex
                ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"], # thalamus
                ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"], # hippocampal
                ["MD","MG","MOp","MOs","MRN"], #Motor areas
                ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP"," TT"], # non-visual cortex
                ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"], # midbrain
                ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"], # basal ganglia 
                ["BLA", "BMA", "EP", "EPd", "MEA"] # cortical subplate
                ]


# Making a trial time variable from -50 to 200
# AA=np.linspace(0, 250,  1)
# BB=[50]*len(AA)

Trial_t=np.linspace(-50, 200,  num=250)#np.subtract(AA,BB)


# =============================================================================
#                       Select Mouse and Pull Out Info
# =============================================================================

# select just one of the recordings here. 11 is nice because it has some neurons in vis ctx. 


file_num=11 #This is one mouse session that we will look at
dat = alldat[file_num]
dat.update(alldat_lfp[file_num])

print(dat.keys())

# _____________________________________________________________________________






dt = dat['bin_size'] # binning at 10 ms
NT = dat['spks'].shape[-1]


response = dat['response'] # right - nogo - left (-1, 0, 1)
vis_right = dat['contrast_right'] # 0 - low - high
vis_left = dat['contrast_left'] # 0 - low - high

#___________________________Brain Area Spikes only_______________________________
nareas = 4 # only the top 4 regions are in this particular mouse
NN = len(dat['brain_area']) # number of neurons
barea = nareas * np.ones(NN, ) # last one is "other"

#Loop over 4 brain areas
for j in range(nareas):
  barea[np.isin(dat['brain_area'], brain_groups[j])] = j # assign a number to each region



#___________________________Brain Area LFP only_______________________________
nareas = 4 # only the top 4 regions are in this particular mouse
NN = len(dat['brain_area_lfp']) # number of neurons
barea_lfp = nareas * np.ones(NN, ) # last one is "other"

#Loop over 4 brain areas
for j in range(nareas):
  barea_lfp[np.isin(dat['brain_area_lfp'], brain_groups[j])] = j # assign a number to each region






##________________ Now we pull out features __________________________________





# plt.plot(1/dt *  dat['spks'][1][:,:].mean(axis=1))
plt.show()

#Right more than left
Look_area=3.
y_RL=dat['lfp'][barea_lfp==Look_area,:,vis_right>vis_left].mean(axis=(0,1))

#Left more than right
y_LR=dat['lfp'][barea_lfp==Look_area][vis_left>vis_right,:].mean(axis=(0,1))


fig = px.line(x=Trial_t, y=[y_RL,y_LR])
# fig = px.line(x=Trial_t, y=np.ndarray.tolist(y_RL))
fig.show()
plot(fig, auto_open=True)

Pxx_den=[]
for ii in range(0,len(alldat[10]['lfp'][0])):
    y=alldat[10]['pupil'][0][ii]
    f, Pxx = signal.welch(y[50:], fs, nperseg=1024)
    Pxx_den.append(Pxx)
fig = px.line(x=f, y=Pxx_den)
fig.show()
plot(fig, auto_open=True)












from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show


p1 = figure( title="Stock Closing Prices")

p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Price'

p1.line(Trial_t,y_LR, color='#A6CEE3', legend_label='AAPL')
p1.legend.location = "top_left"

output_file("stocks.html", title="stocks.py example")
# show(gridplot([[p1]], plot_width=400, plot_height=400))  # open a browser
show(p1)  # open a browser



























### Junk 


for j in range(nareas):
  ax = plt.subplot(1,nareas,j+1)
  
  plt.plot(1/dt *  dat['spks'][barea==j][:,np.logical_and(vis_left==0, vis_right>0)].mean(axis=(0,1)))
  plt.plot(1/dt *  dat['spks'][barea==j][:,np.logical_and(vis_left>0 , vis_right==0)].mean(axis=(0,1)))
  plt.plot(1/dt *  dat['spks'][barea==j][:,np.logical_and(vis_left==0 , vis_right==0)].mean(axis=(0,1)))
  plt.plot(1/dt *  dat['spks'][barea==j][:,np.logical_and(vis_left>0, vis_right>0)].mean(axis=(0,1)))  
  plt.text(.25, .92, 'n=%d'%np.sum(barea==j), transform=ax.transAxes)
 
  if j==0:
    plt.legend(['right only', 'left only', 'neither', 'both'], fontsize=12)
  ax.set(xlabel = 'binned time', ylabel = 'mean firing rate (Hz)', title = regions[j])


