import os
import shutil
import sys
import random
import time
import operator
import math 
import copy
import fnmatch
import shutil
import collections
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import multiprocessing
import itertools
import plotly
import plotly.plotly as py
import pandas
import argparse
import pandas as pd
import seaborn as sns
import scipy.ndimage as ndimage
#plotly.offline.init_notebook_mode()
from plotly.graph_objs import *
from pylab import rcParams
from copy import deepcopy 
from pylab import rcParams
from scipy import special
from PIL import Image
from io import StringIO
from cycler import cycler
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from pyBadlands.model import Model as badlandsModel

def problem_setup(problem = 1):
    random.seed(time.time()) 


    inittopo_estimated = np.array([])

    if problem == 1: #this will have region and time rainfall of Problem 1
        problemfolder = 'Examples/etopo_extended/'
        datapath = problemfolder + 'data/final_elev.txt'
        groundtruth_elev = np.loadtxt(datapath)
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')
 
        res_summaryfile = '/results_temporalrain.txt'
        inittopo_expertknow = [] # no expert knowledge as simulated init topo
        len_grid = 1  # ignore - this is in case if init topo is inferenced
        wid_grid = 1   # ignore
        simtime = 1000000
        resolu_factor = 1
        #true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True
        real_rain = 1.5 #m/a
        real_erod = 5.e-6 
        m = 0.5  #Stream flow parameters
        n = 1 #
        real_cmarine = 5.e-1 # Marine diffusion coefficient [m2/a] -->
        real_caerial = 8.e-1 #aerial diffusion
        rain_min = 0.0
        rain_max = 3.0 
        # assume 4 regions and 4 time scales
        rain_regiongrid = 1  # how many regions in grid format 
        rain_timescale = 4  # to show climate change 

        if rain_timescale ==4:
            xmlinput = problemfolder + 'etopo.xml'
        elif rain_timescale ==8:
            xmlinput = problemfolder + 'etopo_t8.xml' 
        elif rain_timescale ==16:
            xmlinput = problemfolder + 'etopo_t16.xml'

        rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale)
        rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale)
        minlimits_others = [4.e-7, 0, 0, 0 ,  0, 0, 0, 0, 15000, 0, 0]  # make some extra space for future param (last 5)
        maxlimits_others = [6.e-7, 1, 2, 0.1, 0.1, 1, 1, 10, 30000, 10, 1]
        minlimits_vec = np.append(rain_minlimits,minlimits_others)
        maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)
        print(maxlimits_vec, ' maxlimits ')

        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
        true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
        stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now
        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
        print(vec_parameters) 
        erodep_coords = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])  # need to hand pick given your problem

    elif problem == 2: #this will have region and time rainfall of Problem 1
        problemfolder = 'Examples/aus_short/'
        xmlinput = problemfolder + 'aus_short.xml'
        simtime = -5.E+06 #-1.E+05
        resolu_factor = 1

        datapath = problemfolder + 'data/final_elev.txt'
        # datapath = problemfolder + 'data/initial_elev.txt'
        
        groundtruth_elev = np.loadtxt(datapath)
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')
        inittopo_expertknow = []
 

        res_summaryfile = '/results_temporalrain.txt'
        inittopo_expertknow =  np.array([]) # no expert knowledge as simulated init topo

        #true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True

        len_grid = 1  # ignore - this is in case if init topo is inferenced
        wid_grid = 1   # ignore

        real_rain = 3.0 #m/a
        real_erod = 5.e-7 
        m = 0.5  #Stream flow parameters
        n = 1 #
        real_cmarine = 0.005 # Marine diffusion coefficient [m2/a] -->
        real_caerial = 0.001 #aerial diffusion
        rain_min = 1.0
        rain_max = 3.0 

        # assume 4 regions and 4 time scales
        rain_regiongrid = 1  # how many regions in grid format 
        rain_timescale = 4  # to show climate change 
        rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale) 
        rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale) 

        #--------------------------------------------------------
        minlimits_others = [4.e-7, 0, 0, 0 ,  0, 0, 0, 0, 15000, 0, 0]  # make some extra space for future param (last 5)
        maxlimits_others = [6.e-7, 1, 2, 0.1, 0.1, 1, 1, 10, 30000, 10, 1]

        # minlimits_others = [real_erod, m, n, real_cmarine, real_caerial, 0, 0, 0, 0, 0]  # 
        # maxlimits_others = [real_erod, m, n, real_cmarine, real_caerial, 1, 1, 1, 1, 1] # fix erod rain etc

        # need to read file matrix of n x m that defines the course grid of initial topo. This is generated by final
        # topo ground-truth assuming that the shape of the initial top is similar to final one. 

        minlimits_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)

        print(maxlimits_vec, ' maxlimits ')
        print(minlimits_vec, ' maxlimits ')

        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
        true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
        stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
        print(vec_parameters) 

        erodep_coords = np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
    
    elif problem == 3: # 150 MILLION YEARS
        problemfolder = 'Examples/aus/'
        xmlinput = problemfolder + 'aus.xml'
        simtime = -1.49e08
        resolu_factor = 1

        datapath = problemfolder + 'data/final_elev.txt'
        # datapath = problemfolder + 'data/initial_elev.txt'
        
        groundtruth_elev = np.loadtxt(datapath)
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')
        inittopo_expertknow = []

        res_summaryfile = '/results_temporalrain.txt'
        inittopo_expertknow = np.array([]) # no expert knowledge as simulated init topo

        #true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True

        len_grid = 1  # ignore - this is in case if init topo is inferenced
        wid_grid = 1   # ignore

        real_rain = 1.5 #m/a
        real_erod = 5.e-7 
        m = 0.5  #Stream flow parameters
        n = 1 #
        real_cmarine = 0.005 # Marine diffusion coefficient [m2/a] -->
        real_caerial = 0.001 #aerial diffusion

        rain_min = 1.0
        rain_max = 3.0 

        # assume 4 regions and 4 time scales
        rain_regiongrid = 1  # how many regions in grid format 
        rain_timescale = 4  # to show climate change 
        rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale) 
        rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale) 

        #--------------------------------------------------------
        minlimits_others = [4.e-7, 0, 0, 0 ,  0, 0, 0, 0, 15000, 0, 0]  # make some extra space for future param (last 5)
        maxlimits_others = [6.e-7, 1, 2, 0.1, 0.1, 1, 1, 10, 30000, 10, 1]

        # minlimits_others = [real_erod, m, n, real_cmarine, real_caerial, 0, 0, 0, 0, 0]  # 
        # maxlimits_others = [real_erod, m, n, real_cmarine, real_caerial, 1, 1, 1, 1, 1] # fix erod rain etc

        # need to read file matrix of n x m that defines the course grid of initial topo. This is generated by final
        # topo ground-truth assuming that the shape of the initial top is similar to final one. 

        minlimits_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)
        print(maxlimits_vec, ' maxlimits ')
        print(minlimits_vec, ' maxlimits ')

        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
        true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
        stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
        print(vec_parameters) 

        erodep_coords = np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])


    elif problem == 4: # 5 MILLION YEARS with INIT TOPO


        problemfolder = 'Examples/aus_short_inittopo/'
        xmlinput = problemfolder + 'aus_short.xml'

        inittopo_expertknow = np.array([0]) # no expert knowledge as simulated init topo
         
        simtime = -5.E+06 #-1.E+05
        resolu_factor = 1 
 

        datapath = problemfolder + 'data/final_elev.txt'
        # datapath = problemfolder + 'data/initial_elev.txt'
        
        groundtruth_elev = np.loadtxt(datapath)
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')



        inittopo_expertknow =[] # estimated expert in grid version not needed anymore with new init topo formulation
        inittopo_estimated = np.loadtxt(problemfolder + 'init_expertknowlegeprocess/init_estimated.txt') 

        res_summaryfile = '/results_temporalrain.txt'


        inittopo_expertknow = inittopo_expertknow.T # no expert knowledge as simulated init topo

        #true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True

        len_grid = 1  # ignore - this is in case if init topo is inferenced
        wid_grid = 1   # ignore

        real_rain = 1.5 #m/a
        real_erod = 5.e-7 
        m = 0.5  #Stream flow parameters
        n = 1 #
        real_cmarine = 0.005 # Marine diffusion coefficient [m2/a] -->
        real_caerial = 0.001 #aerial diffusion

        rain_min = 1.0
        rain_max = 3.0 

        # assume 4 regions and 4 time scales
        rain_regiongrid = 1  # how many regions in grid format 
        rain_timescale = 4  # to show climate change 
        rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale) 
        rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale) 

        #--------------------------------------------------------
        minlimits_others = [4.e-7, 0, 0, 0 ,  0, 0, 0, 0, 15000, 0, 0]  # make some extra space for future param (last 5)
        maxlimits_others = [6.e-7, 1, 2, 0.1, 0.1, 1, 1, 10, 30000, 10, 1]
 
 
        #----------------------------------------InitTOPO

        #inittopo_gridlen = 20  # should be of same format as @   inittopo_expertknow
        #inittopo_gridwidth = 20

        epsilon = 0.5 

        inittopo_gridlen = 20  # should be of same format as @   inittopo_expertknow
        inittopo_gridwidth = 20


        len_grid = int(groundtruth_elev.shape[0]/inittopo_gridlen)  # take care of left over
        wid_grid = int(groundtruth_elev.shape[1]/inittopo_gridwidth)   # take care of left over

        print(len_grid, wid_grid, groundtruth_elev.shape[0], groundtruth_elev.shape[1] ,'  sub_gridlen, sub_gridwidth   ------------ ********')

         
        inittopo_minlimits = np.repeat( -200  , inittopo_gridlen*inittopo_gridwidth)
        inittopo_maxlimits = np.repeat(200 , inittopo_gridlen*inittopo_gridwidth)
 

        #--------------------------------------------------------


        minlimits_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)



        temp_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        minlimits_vec = np.append(temp_vec, inittopo_minlimits)

        temp_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)
        maxlimits_vec = np.append(temp_vec, inittopo_maxlimits)


  

        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
        true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
        stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
        print(vec_parameters, 'vec_parameters') 

        erodep_coords = np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])


    elif problem == 5: # 5 MILLION YEARS with INIT TOPO


        problemfolder = 'Examples/aus_inittopo/'
        xmlinput = problemfolder + 'aus.xml'

        inittopo_expertknow = np.array([0]) # no expert knowledge as simulated init topo
         
        simtime = -1.49e08
        resolu_factor = 1 
 

        datapath = problemfolder + 'data/final_elev.txt'
        # datapath = problemfolder + 'data/initial_elev.txt'
        
        groundtruth_elev = np.loadtxt(datapath)
        groundtruth_erodep = np.loadtxt(problemfolder + 'data/final_erdp.txt')
        groundtruth_erodep_pts = np.loadtxt(problemfolder + 'data/final_erdp_pts.txt')



        inittopo_expertknow =[] # estimated expert in grid version not needed anymore with new init topo formulation
        inittopo_estimated = np.loadtxt(problemfolder + 'init_expertknowlegeprocess/init_estimated.txt') 

        res_summaryfile = '/results_temporalrain.txt'


        inittopo_expertknow = inittopo_expertknow.T # no expert knowledge as simulated init topo

        #true_parameter_vec = np.loadtxt(problemfolder + 'data/true_values.txt')
        likelihood_sediment = True

        len_grid = 1  # ignore - this is in case if init topo is inferenced
        wid_grid = 1   # ignore

        real_rain = 1.5 #m/a
        real_erod = 5.e-7 
        m = 0.5  #Stream flow parameters
        n = 1 #
        real_cmarine = 0.005 # Marine diffusion coefficient [m2/a] -->
        real_caerial = 0.001 #aerial diffusion

        rain_min = 1.0
        rain_max = 3.0 

        # assume 4 regions and 4 time scales
        rain_regiongrid = 1  # how many regions in grid format 
        rain_timescale = 4  # to show climate change 
        rain_minlimits = np.repeat(rain_min, rain_regiongrid*rain_timescale) 
        rain_maxlimits = np.repeat(rain_max, rain_regiongrid*rain_timescale) 

        #--------------------------------------------------------
        minlimits_others = [4.e-7, 0, 0, 0 ,  0, 0, 0, 0, 15000, 0, 0]  # make some extra space for future param (last 5)
        maxlimits_others = [6.e-7, 1, 2, 0.1, 0.1, 1, 1, 10, 30000, 10, 1]
 
 
        #----------------------------------------InitTOPO

        #inittopo_gridlen = 20  # should be of same format as @   inittopo_expertknow
        #inittopo_gridwidth = 20

        epsilon = 0.5 

        inittopo_gridlen = 20  # should be of same format as @   inittopo_expertknow
        inittopo_gridwidth = 20


        len_grid = int(groundtruth_elev.shape[0]/inittopo_gridlen)  # take care of left over
        wid_grid = int(groundtruth_elev.shape[1]/inittopo_gridwidth)   # take care of left over

        print(len_grid, wid_grid, groundtruth_elev.shape[0], groundtruth_elev.shape[1] ,'  sub_gridlen, sub_gridwidth   ------------ ********')

         
        inittopo_minlimits = np.repeat( -200  , inittopo_gridlen*inittopo_gridwidth)
        inittopo_maxlimits = np.repeat(200 , inittopo_gridlen*inittopo_gridwidth)
 

        #--------------------------------------------------------


        minlimits_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        maxlimits_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)



        temp_vec = np.append(rain_minlimits,minlimits_others)#,inittopo_minlimits)
        minlimits_vec = np.append(temp_vec, inittopo_minlimits)

        temp_vec = np.append(rain_maxlimits,maxlimits_others)#,inittopo_maxlimits)
        maxlimits_vec = np.append(temp_vec, inittopo_maxlimits)


  

        vec_parameters = np.random.uniform(minlimits_vec, maxlimits_vec) #  draw intial values for each of the free parameters
        true_parameter_vec = vec_parameters # just as place value for now, true parameters is not used for plotting 
        stepsize_ratio  = 0.1 #   you can have different ratio values for different parameters depending on the problem. Its safe to use one value for now

        stepratio_vec =  np.repeat(stepsize_ratio, vec_parameters.size) 
        num_param = vec_parameters.size
        print(vec_parameters, 'vec_parameters') 

        erodep_coords = np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])

    return (problemfolder, xmlinput, simtime, resolu_factor, datapath, groundtruth_elev, groundtruth_erodep,
    groundtruth_erodep_pts, res_summaryfile, inittopo_expertknow, len_grid, wid_grid, simtime, 
    resolu_factor, likelihood_sediment, rain_min, rain_max, rain_regiongrid, minlimits_others,
    maxlimits_others, stepsize_ratio, erodep_coords, inittopo_estimated, vec_parameters,minlimits_vec, maxlimits_vec)