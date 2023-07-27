import numpy as np
import matplotlib.pyplot as plt
from landlab.components import ChannelProfiler #library for plotting profile

def simple_plot(fignum, filename, figsize, title, grid, field, color_label,cmap,log_scale,v):
    plt.figure(fignum, figsize = figsize)
    data = np.flipud(grid.at_node[field].reshape(grid.shape)).copy()
    if log_scale == True:
        data[data==0]=np.nan
        im =plt.imshow(np.log10(data),cmap = cmap,extent=[0,grid.dx*grid.shape[1],0,grid.dx*grid.shape[0]],vmin=np.log10(v[0]),vmax=np.log10(v[1]))
    else:    
        im =plt.imshow(data,cmap = cmap,extent=[0,grid.dx*grid.shape[1],0,grid.dx*grid.shape[0]],vmin=v[0],vmax=v[1])
    plt.title(title)
    plt.xlabel('x [meters]')
    plt.ylabel('y [meters]')
    cbar = plt.colorbar(im,label=color_label)
    plt.tight_layout()
    if filename != "":
        plt.savefig(filename,dpi=250)
        plt.close()
        
def rock_plot(fignum, filename, figsize, title, grid, field, color_label,cmap,log_scale,v):
    plt.figure(fignum, figsize = figsize)
    data = np.flipud(grid.at_node[field].reshape(grid.shape)).copy()
    if log_scale == True:
        data[data==0]=np.nan
        im =plt.imshow(np.log10(data),cmap = cmap,extent=[0,grid.dx*grid.shape[1],0,grid.dx*grid.shape[0]],vmin=np.log10(v[0]),vmax=np.log10(v[1]))
    else:    
        im =plt.imshow(data,cmap = cmap,extent=[0,grid.dx*grid.shape[1],0,grid.dx*grid.shape[0]],vmin=v[0],vmax=v[1])
    plt.title(title)
    plt.xlabel('x [meters]')
    plt.ylabel('y [meters]')
    cbar = plt.colorbar(im,label=color_label)
    plt.tight_layout()
    if filename != "":
        plt.savefig(filename,dpi=250)
        plt.close()
        
def reset_landscape_random(grid,initial_topography):
    if initial_topography == "":
        grid.at_node['topographic__elevation'] = np.random.rand(len(grid.at_node['topographic__elevation']))
    else:
        grid.at_node['topographic__elevation'] = np.load(initial_topography)
    
def profile_plot(fignum, filename, figsize, title, grid, lith,ids,v,rock_colors):
    profiler = ChannelProfiler(grid,number_of_watersheds=1,main_channel_only=True)
    profiler.run_one_step()
    
    fig1 = plt.figure(fignum, figsize = figsize)
    ax1 = fig1.add_subplot(121)
    
    outlets = list(profiler.data_structure.keys())
    distance = np.array([])
    eta = np.array([])
    ids_to_plot = []
    for outlet in outlets:
        segments = list(profiler.data_structure[outlet].keys())
        for i, segment in enumerate(segments):
            ids_ = profiler.data_structure[outlet][segment]['ids']
            ids_to_plot += [ids_]
            distance = np.concatenate((distance, profiler.data_structure[outlet][segment]['distances']), axis=None)
            eta = np.concatenate((eta, grid.at_node['topographic__elevation'][ids_]), axis=None)
    
    num_layers = lith.z_top.shape[0]
    layers = np.zeros((num_layers+1,len(eta)))
    layers[0,:] = 10000.
    index = 0
    for outlet in outlets:
        segments = list(profiler.data_structure[outlet].keys())
        for i, segment in enumerate(segments):
            ids_ = profiler.data_structure[outlet][segment]['ids']
            for j in ids_:
                for k in range(0,num_layers-1):
                    layers[k+1,index] = lith.z_top[k,j]
                index+=1
        
    for k in range(0,num_layers):
        rock_id = ids[len(ids)-k-1]
        ax1.fill_between(distance,eta-layers[k,:],eta-layers[k+1,:],color=rock_colors[rock_id-1])
        
        
    ax1.plot(distance,eta,color='k')
    ax1.set_title(title)
    ax1.set_xlabel('s [meters]')
    ax1.set_ylabel(r'$\eta$ [meters]')
    if v[0] == -9999. and v[1] != -9999.:
        ax1.set_ylim(0,v[1])
    elif v[0] != -9999. and v[1] == -9999.:
        ax1.set_ylim(v[0],)
    elif v[0] != -9999. and v[1] != -9999.:
        ax1.set_ylim(v[0],v[1])
    np.save(filename[:-4]+'_s.npy',distance)
    np.save(filename[:-4]+'_eta.npy',eta)
        
    ax2 = fig1.add_subplot(122)
    data = np.flipud(grid.at_node['topographic__elevation'].reshape(grid.shape)).copy() 
    im = ax2.imshow(data,cmap = 'terrain',extent=[0,grid.dx*grid.shape[1],0,grid.dx*grid.shape[0]],vmin=0,vmax=v[1])
    profile_2d = np.zeros_like(grid.at_node['topographic__elevation'])
    profile_2d[ids_to_plot] = 1.0
    profile_2d = np.flipud(profile_2d.reshape(grid.shape))
    ax2.imshow(profile_2d,cmap = 'bwr',extent=[0,grid.dx*grid.shape[1],0,grid.dx*grid.shape[0]],alpha=profile_2d)
    ax2.set_title(title)
    ax2.set_xlabel('x [meters]')
    ax2.set_ylabel('y [meters]')
    cbar = plt.colorbar(im,label=r'$\eta$')     
        
    plt.tight_layout()
    if filename != "":
        plt.savefig(filename,dpi=250)
        plt.close()
               
def x_cross_section(fignum, filename, figsize, title, grid, lith,ids, y_plot,v,rock_colors):
    plt.figure(fignum, figsize = figsize)
    x_plot = grid.x_of_node[grid.y_of_node == y_plot]
    eta = grid.at_node['topographic__elevation'][grid.y_of_node == y_plot]
    
    num_layers = lith.z_top.shape[0]
    layers = np.zeros((num_layers+1,len(eta)))
    layers[0,:] = 10000.
    for k in range(0,num_layers):
        layers[k+1,:] = lith.z_top[k,:][grid.y_of_node == y_plot]
    for k in range(0,num_layers):
        rock_id = ids[len(ids)-k-1]
        plt.fill_between(x_plot,eta-layers[k,:],eta-layers[k+1,:],color=rock_colors[rock_id-1])
        
    plt.plot(x_plot,eta,color='k')
    plt.title(title)
    plt.xlabel('x [meters]')
    plt.ylabel(r'$\eta$ [meters]')
    if v[0] == -9999. and v[1] != -9999.:
        plt.ylim(0,v[1])
    elif v[0] != -9999. and v[1] == -9999.:
        plt.ylim(v[0],)
    elif v[0] != -9999. and v[1] != -9999.:
        plt.ylim(v[0],v[1])
    plt.tight_layout()
    if filename != "":
        plt.savefig(filename,dpi=250)
        plt.close()
        
def y_cross_section(fignum, filename, figsize, title, grid, lith,ids, x_plot,v,rock_colors):
    plt.figure(fignum, figsize = figsize)
    y_plot = grid.y_of_node[grid.x_of_node == x_plot]
    eta = grid.at_node['topographic__elevation'][grid.x_of_node == x_plot]
    
    num_layers = lith.z_top.shape[0]
    layers = np.zeros((num_layers+1,len(eta)))
    layers[0,:] = 10000.
    for k in range(0,num_layers):
        layers[k+1,:] = lith.z_top[k,:][grid.x_of_node == x_plot]
    for k in range(0,num_layers):
        rock_id = ids[len(ids)-k-1]
        plt.fill_between(y_plot,eta-layers[k,:],eta-layers[k+1,:],color=rock_colors[rock_id-1])
        
    plt.plot(y_plot,eta,color='k')
    plt.title(title)
    plt.xlabel('y [meters]')
    plt.ylabel(r'$\eta$ [meters]')
    if v[0] == -9999. and v[1] != -9999.:
        plt.ylim(0,v[1])
    elif v[0] != -9999. and v[1] == -9999.:
        plt.ylim(v[0],)
    elif v[0] != -9999. and v[1] != -9999.:
        plt.ylim(v[0],v[1])
    plt.tight_layout()
    if filename != "":
        plt.savefig(filename,dpi=250)
        plt.close()
         
def plot_relief(fignum, filename,time_save,relief_save, figsize,t_R,R,v):
    plt.figure(fignum, figsize = figsize)
    plt.plot(t_R,R,color='k')
    plt.title('Relief')
    plt.xlabel('t [yrs]')
    plt.ylabel('R [meters]')
    plt.xlim(t_R[0],t_R[-1])
    plt.ylim(v[0],v[1])
    plt.tight_layout()
    if filename != "":
        plt.savefig(filename,dpi=250)
        plt.close()
    np.save(time_save,t_R)
    np.save(relief_save,R)
        
def add_folds(grid,field,slope,wavelength,amplitude,horizontal_line_boolean,vertical_line_boolean):
    if horizontal_line_boolean == True and vertical_line_boolean == True:
        sys.exit("Cannot have horizontal_line_boolean == True and vertical_line_boolean == True.")   
    elif horizontal_line_boolean == True:
        for i in range(0,grid.shape[0]):
            for j in range(0,grid.shape[1]):
                y = float(i) * grid.dy
                grid.at_node[field].reshape(grid.shape)[i,j] = amplitude * (1. - np.cos(2.0 * np.pi * y / wavelength))
    elif vertical_line_boolean == True:  
        for i in range(0,grid.shape[0]):
            for j in range(0,grid.shape[1]):
                x = float(j) * grid.dx
                grid.at_node[field].reshape(grid.shape)[i,j] = amplitude * (1. - np.cos(2.0 * np.pi * x / wavelength))
    else:
        if slope == 0.0:
            sys.exit("Slope cannot be zero. Try using horizontal_line_boolean == True or vertical_line_boolean == True instead.") 
        else:
            #origin at the center
            for i in range(0,grid.shape[0]):
                for j in range(0,grid.shape[1]):
                    x = float(j) * grid.dx
                    y = float(i) * grid.dy
                    x_o = (slope ** 2.0 * grid.extent[1] / 2.0 - slope * grid.extent[0] / 2.0 + slope * y + x) / (1. + slope ** 2.)
                    y_o = slope * (x_o - grid.extent[1] / 2.0) + grid.extent[0] / 2.0
                    dist = np.sqrt((x-x_o)**2.0 + (y-y_o)**2.0)
                    grid.at_node[field].reshape(grid.shape)[i,j] = amplitude * (1. - np.cos(2.0 * np.pi * dist / wavelength))
                    