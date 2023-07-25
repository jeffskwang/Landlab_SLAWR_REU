###################################################
#####NO NEED TO MODIFY THE CODE IN THIS BLOCK######
###################################################

###IMPORT LIBRARIES###
#system library for importing
from functions import *
import sys
import importlib
if len(sys.argv) == 1:
    sys.exit('Error: Please specify a parameters file!!!')
parameter = importlib.import_module(sys.argv[1])
globals().update(parameter.__dict__)

#Numpy, array library
import numpy as np
np.random.seed(random_seed)

#import os, a folder management library
import os #os - allows interaction between files and folders on your computer
import shutil #library for deleting folders, copying, etc.

#time library
import time #keeps time

#makes numerical grid for landscapes
from landlab import RasterModelGrid #sets up grid

###Import the components that we need from landlab
from landlab.components import FlowAccumulator #flow accumulation
from landlab.components import FastscapeEroder #fluvial erosion
from landlab.components import LinearDiffuser #hillslope diffusion
from landlab.components import Lithology #Lithology for changing bedrock

start_time = time.time()

#x and y
x = np.linspace(0.,dx*(cols-1),cols)
y = np.linspace(0.,dx*(rows-1),rows)

#make the folder if it doesn't exist
main_folder = os.getcwd()
save_folder = main_folder + '/' + save_folder
if os.path.exists(save_folder) == True:
    shutil.rmtree(save_folder)
    time.sleep(5)
os.mkdir(save_folder)

#make topographic elevation field with all zeros
grid.add_zeros('topographic__elevation', at = 'node')

#This sets the boundary conditions to open (False) - Close (True)
#(right edge, top edge, left edge, bottom edge)
grid.set_closed_boundaries_at_grid_edges(closed_bc_right,closed_bc_top,closed_bc_left,closed_bc_bottom)

###NOTE: No need to modify anything below this###
#reset the landscape at the beginning of the run
reset_landscape_random(grid,initial_topography)

#set up lithlogy
total_number_of_layers=len(ids)
for i in range(0,total_number_of_layers):
    if i == 0:
        thicknesses[i] = thicknesses[i] + grid.at_node['topographic__elevation']
    else:
        thicknesses[i] = thicknesses[i] + np.zeros(grid.number_of_nodes)

#set up lithology
lith = Lithology(grid, thicknesses, ids, attrs, dz_advection = U * dt)
#set up flow accumulator, finds the drainage area
flow = FlowAccumulator(grid,flow_director="D8",depression_finder = 'DepressionFinderAndRouter')
#set up fuvial erosion
fluvial = FastscapeEroder(grid, K_sp = grid.at_node['K_sp'],m_sp=0.5,n_sp=1.0)
#set up the hillslope diffusion
diffuse = LinearDiffuser(grid, linear_diffusivity=D, deposit = False)

K_values = list(attrs['K_sp'].values())
diss_values = list(attrs['diss'].values())

#set initial figure number
fignum = 1
total_plots = int(T/dt_plot + 1)
#relief time series
R = np.zeros(int(T/dt + 1))
for t in range(0,int(T/dt)+1):
    #calculate total relief
    R[t] = np.max(grid.at_node['topographic__elevation']) - np.min(grid.at_node['topographic__elevation'])
    
    #run flow accumulator
    flow.run_one_step()
    #run fluvial erosion, this requires flow accumulator to run first
    fluvial.run_one_step(dt=dt)
    #run hillslope diffusion
    diffuse.run_one_step(dt=dt)
    #dissolve landscape
    grid.at_node['topographic__elevation'][grid.core_nodes] -= grid.at_node['diss'][grid.core_nodes]  * dt
    #uplift landscape
    grid.at_node['topographic__elevation'][grid.core_nodes] += U * dt
    #update lithlogy
    lith.run_one_step()
    if t%(int(dt_plot/dt))==0:
        year = round(t * dt)
        print ('plotting at '+str(year)+' years...')
        if plot_topography == True:
            if max_elevation == -9999:
                max_elevation = np.max(grid.at_node['topographic__elevation'][grid.core_nodes])
            simple_plot(fignum,save_folder+'/topography_'+str(year)+'yrs.png',\
                        figsize,'T = ' + str(round(t*dt/1000)) +' kyrs',grid,'topographic__elevation','elevation [m]','terrain',False,[0,max_elevation])
        if plot_drainage_area == True:
            simple_plot(fignum+total_plots,save_folder+'/drainage_area_'+str(year)+'yrs.png',\
                        figsize,'T = ' + str(round(t*dt/1000)) +' kyrs',grid,'drainage_area',r'$log_{10}$(drainage area) [-]','viridis',True,[dx*dx,dx*dx*rows*cols])
        if plot_rock_type == True:
            simple_plot(fignum+total_plots*2,save_folder+'/erodibility_'+str(year)+'yrs.png',\
                        figsize,'T = ' + str(round(t*dt/1000)) +' kyrs',grid,'K_sp','K [1/yr]','coolwarm',False,[np.min(K_values),np.max(K_values)])
        if plot_dissolution_rate == True:
            simple_plot(fignum+total_plots*3,save_folder+'/diss_'+str(year)+'yrs.png',\
                        figsize,'T = ' + str(round(t*dt/1000)) +' kyrs',grid,'diss','S [m/yr]','Blues',False,[np.min(diss_values),np.max(diss_values)])
        if plot_profile == True:
            profile_plot(fignum+total_plots*4,save_folder+'/profile_'+str(year)+'yrs.png',\
                        figsize_profile,'T = ' + str(round(t*dt/1000)) +' kyrs',grid,lith,ids,[min_lithology_elevation,max_elevation],rock_colors)
        if plot_x_cross_section == True:
            x_cross_section(fignum+total_plots*5,save_folder+'/x_cross_section_'+str(year)+'yrs.png',\
                        figsize,'T = ' + str(round(t*dt/1000)) +' kyrs',grid,lith,ids,y[int(rows/2)],[min_lithology_elevation,max_elevation],rock_colors)
        if plot_y_cross_section == True:
            y_cross_section(fignum+total_plots*6,save_folder+'/y_cross_section_'+str(year)+'yrs.png',\
                        figsize,'T = ' + str(round(t*dt/1000)) +' kyrs',grid,lith,ids,x[int(cols/2)],[min_lithology_elevation,max_elevation],rock_colors)
        fignum+=1
        
plot_relief(fignum+total_plots*6, save_folder+'/_relief.png',\
        save_folder+'/_relief.npy', save_folder+'/_relief.npy',\
        figsize,np.linspace(0,T,int(T/dt)+1),R,[0,max_elevation])  

if save_final_topography == True:
    np.save(save_folder+'/_final_topography.npy',grid.at_node['topographic__elevation'])
    
end_time = time.time()
print ('code took ' + str(round((end_time-start_time)/60.0,1)) + ' minutes to run')