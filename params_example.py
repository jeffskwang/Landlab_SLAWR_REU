#########################
###Physical Parameters###
#########################

#Erodibility
K = 0.0001 #1/yr

#Hillslope Diffusion Coefficient
D = 0.01 #m^2/yr

#Uplift Rate
U = 0.001 #m/yr

#Soluability Rate
S = 0.0005 #m/yr

##########################
###Numerical Parameters###
##########################

#Random Seed
random_seed = 1

#Simulation Time
T = 200000. #yr

#Model Timestep
dt = 100. #yr

#Plot Timestep
dt_plot = 50000. #yr

#Grid Size
rows = 100 #y_cells
cols = 100 #x cells

#Cell Size
dx = 50. #meters

#Boundary Conditions, True - Closed, False - Open
closed_bc_top = True
closed_bc_bottom = False
closed_bc_right = True
closed_bc_left = True

####################
###I/O Parameters###
####################

#Initial Topography File
#   -Leave blank ("") if you want a randomized landscape
#   -Give an input if to use it as an input; make sure to give the full pathname
initial_topography = ""

#Output Folder
save_folder = "run_example" #name of the folder that the images will save to
save_final_topography = True #Boolean to decide if final topography is saved or not

#Figure Sizes
figsize = (5,4) #Tuple in inches for most figures
figsize_profile = (9,4) #Tuple in inches for the profile figure

#Figure extent
max_elevation = 150. #max elevation to scale plots, use -9999 if you want to be automated
min_lithology_elevation = -100. #min elevation to scale plots, use -9999 if you want to be automated

#Boolean variables to decide if figures are made or not
plot_topography = True #plot showing topography
plot_drainage_area = True #plot showing drainage area
plot_rock_type = True #plot showing erodibility 
plot_dissolution_rate = True #plot showing dissolution rate
plot_x_cross_section = True #x transect plot
plot_y_cross_section = True #y transect plot
plot_profile = True #profile plot

##################################################
###################DO NOT MODIFY##################
from landlab import RasterModelGrid #sets up grid
grid = RasterModelGrid((rows,cols),dx) #makes grid
from functions import *
###################DO NOT MODIFY##################
##################################################

#IMPORTANT!!! PLEASE COMMENT OUT CLAY'S SECTION (if you are marco) OR
#MARCO'S SECTION (if you are clay) BEFORE RUNNING the model

###############
###Lithology### 
###############

#Rock Colors in RGB values (tuples). Need to specify RGB values for each rock type. This case has two.
rock_colors = [(0.74, 0.62, 0.51),\
               (0.80, 0.80, 0.80)] #2 layer-example

###############
#####MARCO##### Dissolution
###############
'''
#Layer Thicknesses in meters
thicknesses = [50.,50.,50.,50.,50.,50.,50.,50.,5000000.] #meters

#Rock IDs of the Layers
ids = [1, 2, 1, 2, 1, 2, 1, 2, 1]

#Attributes of Rock IDS - Ksp is erodibility, diss is dissolution
attrs = {'K_sp': {1: K, 2: K},\
         'diss': {1: 0.0, 2: S}}
'''
##############
#####CLAY##### Folds
##############
#Fold Geometry
fold_amplitude = 50. #meters, fold amplitude
fold_wavelength = dx*rows/2.0 #meters, fold wavelength
fold_thickness = 300. #meters, fold thickness
fold_orientation = 'vertical' #or 'horizontal', can also be a slope number

#Make new field for the fold geometry
fold = grid.add_zeros('fold', at = 'node')
if fold_orientation == 'vertical':
    add_folds(grid,'fold',0.0,fold_wavelength,fold_amplitude,False,True) 
elif fold_orientation == 'horizontal':
    add_folds(grid,'fold',0.0,fold_wavelength,fold_amplitude,True,False) 
else: 
    add_folds(grid,'fold',fold_orientation,fold_wavelength,fold_amplitude,False,False) 

#Layer Thicknesses in meters
thicknesses = [fold,\
                np.ones(grid.number_of_nodes) * fold_thickness,\
                np.ones(grid.number_of_nodes) * 100000.] 

#Rock IDs of the Layers
ids = [1, 2, 1]

#Attributes of Rock IDS - Ksp is erodibility, diss is dissolution
attrs = {'K_sp': {1: K, 2: K*0.25},'diss': {1: 0.0, 2: 0.0}}