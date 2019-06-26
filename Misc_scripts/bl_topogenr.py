##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesLands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##

#Main Contributer: Danial Azam  Email: dazam92@gmail.com

"""
This script is intended to generate the input and final-time topography used by the mcmc file.
"""
import os
import numpy as np
import random
import time
import math
import copy
import fnmatch
import shutil
import plotly
import collections
import plotly.plotly as py
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import cmocean as cmo
import plotly.graph_objs as go
from copy import deepcopy
from pylab import rcParams
from PIL import Image
from io import StringIO
from cycler import cycler
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from plotly.graph_objs import *
from plotly.offline.offline import _plot_html
plotly.offline.init_notebook_mode()


def interpolateArray(coords=None, z=None, dz=None):
	"""
	Interpolate the irregular spaced dataset from badlands on a regular grid.

	Parameters
	----------
	variable : coords
		R.
	variable: z
		F.
	variable: dz
		F.
	
	"""
	x, y = np.hsplit(coords, 2)
	dx = (x[1]-x[0])[0]

	nx = int((x.max() - x.min())/dx+1)
	ny = int((y.max() - y.min())/dx+1)
	xi = np.linspace(x.min(), x.max(), nx)
	yi = np.linspace(y.min(), y.max(), ny)

	xi, yi = np.meshgrid(xi, yi)
	xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
	XY = np.column_stack((x,y))

	tree = cKDTree(XY)
	distances, indices = tree.query(xyi, k=3)
	if len(z[indices].shape) == 3:
		z_vals = z[indices][:,:,0]
		dz_vals = dz[indices][:,:,0]
	else:
		z_vals = z[indices]
		dz_vals = dz[indices]

	zi = np.average(z_vals,weights=(1./distances), axis=1)
	dzi = np.average(dz_vals,weights=(1./distances), axis=1)
	onIDs = np.where(distances[:,0] == 0)[0]
	if len(onIDs) > 0:
		zi[onIDs] = z[indices[onIDs,0]]
		dzi[onIDs] = dz[indices[onIDs,0]]
	zreg = np.reshape(zi,(ny,nx))
	dzreg = np.reshape(dzi,(ny,nx))
	return zreg,dzreg

def topoGenerator(directory, inputname, rain, erodibility, m, n, simtime, erdp_coords, elev_coords ,final_noise):
	"""
	
	Parameters
	----------
	variable : directory
		
	variable: inputname
		
	variable: height
		
	variable: zmin

	variable: zmax

	variable: height

	variable: zData

	variable: title

	"""
	num_successive_topo = 4
	sim_interval = np.arange(0,  simtime+(simtime/num_successive_topo), simtime/num_successive_topo) # for generating successive topography
	if simtime < 0:
		sim_interval = sim_interval[::-1]
	print("Simulation time interval", sim_interval)

	model = badlandsModel()
	model.load_xml(str(simtime), inputname, verbose = False, muted = False)
	model.input.SPLero = erodibility
	model.flow.erodibility.fill(erodibility)
	# model.force.rainVal[:] = rain
	model.input.SPLm = m
	model.input.SPLn = n

	elev_vec = collections.OrderedDict()
	erdp_vec = collections.OrderedDict()
	erdp_pts_vec = collections.OrderedDict()
	elev_pts_vec = collections.OrderedDict()
	
	for x in range(len(sim_interval)):
		
		simtime = sim_interval[x]

		model.run_to_time(simtime, muted = True)
		
		elev, erdp = interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)
		
		erdp_pts = np.zeros((erdp_coords.shape[0]))
		elev_pts = np.zeros((elev_coords.shape[0]))

		for count, val in enumerate(erdp_coords):
			erdp_pts[count] = erdp[val[0], val[1]]

		for count, val in enumerate(elev_coords):
			elev_pts[count] = elev[val[0], val[1]]

		# Adding Noise
		tausq_elev = elev.max()* (0.01) + 0.5
		tausq_erdp = erdp.max()* (0.01) + 0.5
		tausq_erdp_pts = erdp_pts.max()*(0.01) + 0.5 
		tausq_elev_pts = elev_pts.max()*(0.01) + 0.5 
		
		elev_noise = np.random.normal(0, np.sqrt(abs(tausq_elev)), elev.size)
		elev_noise = np.reshape(elev_noise,(elev.shape[0],elev.shape[1]))	
		erdp_noise = np.random.normal(0, np.sqrt(abs(tausq_erdp)), erdp.size)
		erdp_noise = np.reshape(erdp_noise,(erdp.shape[0],erdp.shape[1]))	
		erdp_pts_noise = np.random.normal(0, np.sqrt(abs(tausq_erdp_pts)), erdp_pts.size)
		erdp_pts_noise = np.reshape(erdp_pts_noise,(erdp_pts.shape))

		elev_pts_noise = np.random.normal(0, np.sqrt(abs(tausq_elev_pts)), elev_pts.size)
		elev_pts_noise = np.reshape(elev_pts_noise,(elev_pts.shape))
		
		elev_=np.matrix(elev)
		erdp_=np.matrix(erdp)
		erdp_pts_ = np.matrix(erdp_pts)
		elev_pts_ = np.matrix(elev_pts)

		if final_noise and simtime==sim_interval[-1]:
			elev_mat=np.add(elev_, elev_noise)
			erdp_mat=np.add(erdp_, erdp_noise)
			erdp_pts_mat = np.add(erdp_pts_, erdp_pts_noise)
			elev_pts_mat = np.add(elev_pts_, elev_pts_noise)
		else:
			elev_mat = elev_
			erdp_mat = erdp_
			erdp_pts_mat = erdp_pts_
			elev_pts_mat = elev_pts_
	
		elev_vec[simtime] = elev_mat
		erdp_vec[simtime] = erdp_mat
		erdp_pts_vec[simtime] = erdp_pts_mat
		elev_pts_vec[simtime] = elev_pts_mat
		
	for k, v in elev_vec.items():
		if k == sim_interval[0]:
			np.savetxt('%s/data/initial_elev.txt' %directory,  elev_vec[k],fmt='%.5f')
			viewGrid(directory,'initial_elev', 'N/A', rain, erodibility, zData=elev_vec[k], title='Export Slope Grid')
		
		elif k == sim_interval[-1]:
			np.savetxt('%s/data/final_elev.txt' %directory, elev_vec[k],fmt='%.5f')
			viewGrid(directory,'final_elev', 'N/A', rain, erodibility, zData=elev_vec[k], title='Export Slope Grid')

		# np.savetxt('%s/data/elev_%s.txt' %(directory,k),  elev_vec[k],fmt='%.5f')
		# viewGrid(directory,'elev%s' %(k), 'N/A', rain, erodibility, zData=elev_vec[k], title='Export Slope Grid')

	for k, v in erdp_vec.items():
		if k == sim_interval[0]:
			# np.savetxt('%s/data/initial_erdp.txt' %directory,  erdp_vec[k],fmt='%.5f')
			viewMap(directory,'initial_erdp', 'N/A', rain, erodibility, zData=erdp_vec[k], title='Export Slope Grid')
		
		if k == sim_interval[-1]:
			np.savetxt('%s/data/final_erdp.txt' %directory, erdp_vec[k],fmt='%.5f')
			viewMap(directory,'final_erdp', 'N/A', rain, erodibility, zData=erdp_vec[k], title='Export Slope Grid')
			
		# np.savetxt('%s/data/erdp_%s.txt' %(directory,k), erdp_vec[k],fmt='%.5f')
		# viewMap(directory,'erdp_%s' %(k), 'N/A', rain, erodibility, zData=erdp_vec[k], title='Export Slope Grid')


	erdp_pts_arr = np.zeros((sim_interval.size, erdp_pts_mat.size))
	count = 0
	for k, v in erdp_pts_vec.items():
		erdp_pts_arr[count] = v
		count +=1
		if k == sim_interval[0]:
			# np.savetxt('%s/data/initial_erdp_pts.txt' %directory,  erdp_pts_vec[k],fmt='%.5f')
			viewBar(directory,'initial_erdp_pts', 'N/A', rain, erodibility, xData = erdp_coords, yData=erdp_pts_mat, title='Export Slope Grid')
		
		if k == sim_interval[-1]:
			np.savetxt('%s/data/final_erdp_pts.txt' %directory,erdp_pts_arr,fmt='%.5f')
			viewBar(directory,'final_erdp_pts', 'N/A', rain, erodibility, xData = erdp_coords ,yData=erdp_pts_arr[-1], title='Export Slope Grid')

	# 	# np.savetxt('%s/data/erdp_pts_%s.txt' %(directory,k),erdp_pts_arr,fmt='%.5f')
	# 	# viewBar(directory,'erdp_pts_%s' %(k), 'N/A', rain, erodibility, xData = erdp_coords ,yData=erdp_pts_arr[count-1], title='Export Slope Grid')


	elev_pts_arr = np.zeros((sim_interval.size, elev_pts_mat.size))
	count = 0
	for k, v in elev_pts_vec.items():
		elev_pts_arr[count] = v
		count +=1
		if k == sim_interval[0]:
			# np.savetxt('%s/data/initial_erdp_pts.txt' %directory,  elev_pts_vec[k],fmt='%.5f')
			viewBar(directory,'initial_elev_pts', 'N/A', rain, erodibility, xData = elev_coords , yData=elev_pts_mat, title='Export Slope Grid')
		
		if k == sim_interval[-1]:
			np.savetxt('%s/data/final_elev_pts.txt' %directory,elev_pts_arr,fmt='%.5f')
			viewBar(directory,'final_elev_pts', 'N/A', rain, erodibility, xData = elev_coords ,yData=elev_pts_arr[-1], title='Export Slope Grid')

		# np.savetxt('%s/data/erdp_pts_%s.txt' %(directory,k),elev_pts_arr,fmt='%.5f')
		# viewBar(directory,'erdp_pts_%s' %(k), 'N/A', rain, erodibility, xData = erdp_coords ,yData=elev_pts_arr[count-1], title='Export Slope Grid')



	return

def viewGrid(directory,sample_num, likl, rain, erod, width = 1000, height = 1000, zmin = None, zmax = None, zData = None, title='Export Grid'):
	"""
	Use Plotly library to visualise the grid in 3D.
	
	Parameters
	----------
	variable : resolution
		Required resolution for the model grid (in metres).
	variable: width
		Figure width.
	variable: height
		Figure height.
	variable: zmin
		Minimal elevation.
	variable: zmax
		Maximal elevation.
	variable: height
		Figure height.
	variable: zData
		Elevation data to plot.
	variable: title
		Title of the graph.
	"""
	if zmin == None:
		zmin = zData.min()

	if zmax == None:
		zmax = zData.max()

	data = Data([ Surface( x=zData.shape[0], y=zData.shape[1], z=zData, colorscale='YlGnBu' ) ])

	layout = Layout(
		title='Crater Elevation     rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
		autosize=True,
		width=width,
		height=height,
		scene=Scene(
			zaxis=ZAxis(range=[zmin, zmax],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			xaxis=XAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=5,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			yaxis=YAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=5,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			bgcolor="rgb(244, 244, 248)"
		)
	)

	fig = Figure(data=data, layout=layout)
	graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/images/elev_grid_%s.html' %(directory,sample_num), validate=False)
	return    

def viewMap(directory,sample_num, likl, rain, erod, width = 600, height = 600, zmin = None, zmax = None, zData = None, title='Export Grid'):
	"""
	Use Plotly library to visualise the Erosion Deposition Heatmap.

	Parameters
	----------
	variable : resolution
		Required resolution for the model grid (in metres).

	variable: width
		Figure width.

	variable: height
		Figure height.

	variable: zmin
		Minimal elevation.

	variable: zmax
		Maximal elevation.

	variable: height
		Figure height.

	variable: zData
		Elevation data to plot.

	variable: title
		Title of the graph.
	"""
	if zmin == None:
		zmin = zData.min()

	if zmax == None:
		zmax = zData.max()

	trace = go.Heatmap(z=zData)

	data=[trace]
	layout = Layout(
		title='Crater Erosiondeposition     rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
		autosize=True,
		width=width,
		height=height,
		scene=Scene(
			zaxis=ZAxis(range=[zmin, zmax],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			xaxis=XAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			yaxis=YAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			bgcolor="rgb(244, 244, 248)"
		)
	)

	fig = Figure(data=data, layout=layout)
	graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/images/erdp_heatmap_%s.html' %(directory,sample_num), validate=False)
	return    

def viewBar(directory,sample_num, likl, rain, erod, width = 500, height = 500, xData = None, yData = None, title='Export Grid'):
		"""
		Use Plotly library to visualise the BarPlot of Erosion Deposition at certain coordinates.

		Parameters
		----------
		variable : resolution
			Required resolution for the model grid (in metres).
		variable: width
			Figure width.
		variable: height
			Figure height.
		variable: zmin
			Minimal elevation.
		variable: zmax
			Maximal elevation.
		variable: height
			Figure height.
		variable: zData
			Elevation data to plot.
		variable: title
			Title of the graph.
		"""
		xData = np.array_str(xData)
		trace = go.Bar(x=xData, y = yData)
		data=[trace]
		layout = Layout(
			title='Crater Erosion deposition pts    rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
			autosize=True,
			width=width,
			height=height,
			scene=Scene(
				xaxis=XAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				yaxis=YAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				bgcolor="rgb(244, 244, 248)"
			)
		)
		fig = Figure(data=data, layout=layout)
		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/images/erdppts_barcht_%s.html' %(directory, sample_num), validate=False)
		
		return

def checkUplift(directory, u_filename, t_filename):
	upl = np.loadtxt('%s%s.csv' %(directory,u_filename))
	top = np.loadtxt('%s%s.csv' %(directory,t_filename))
	upl = upl.reshape(upl.shape[0],1)
	# print(upl.shape)
	# print(top.shape)
	comb = np.hstack((top, upl))
	
	min_bound_x = comb[:,0].min()
	max_bound_x = comb[:,0].max()
	min_bound_y = comb[:,1].min()
	max_bound_y = comb[:,1].max()

	# print ('comb.shape', comb.shape)
	# print ('comb[:,0:2]', comb[:,0:2])
	# print ('max_b x', max_bound_x, 'min_b x', min_bound_x)
	# print ('max_b y', max_bound_y, 'min_b y', min_bound_y)
	for x in range(comb.shape[0]):
		# print (comb[x,:])
		row = comb[x,:]
		ind_min_x = (row == min_bound_x)
		ind_max_x = (row == max_bound_x)
		ind_min_y = (row == min_bound_y)
		ind_max_y = (row == max_bound_y)
		# print (ind_min)
		if (ind_min_x[0] == True or ind_max_x[0] == True) or (ind_min_y[1] == True or ind_max_y[1]==True):
			# print('Im in the IF')
			comb[x,3] = 0.0
			# print (comb[x,:])

	try:
		if os.path.exists('%s%s.csv'% (directory, u_filename)):
			os.remove('%s%s.csv'% (directory, u_filename))
	except OSError:
		pass

	np.savetxt('%s%s.csv'% (directory, u_filename), comb[:,3])

	return True

def main():
	"""
	
	"""
	uplift_verified = False
	choice = input("Please choose a Badlands example to create an Initial and Final Topography for:\n 1) aus_short\n 2) aus\n")
	directory = ""

	erdp_coords_australia = np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
	elev_coords_australia = np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])

	final_noise = True

	tstart = time.time()

	if choice == 1:
		directory = 'Examples/aus_short'
		print '%s/aus_short.xml' %(directory)
		topoGenerator(directory,'%s/aus_short.xml' %(directory), 1.0 , 5.e-7, 0.5, 1, -5.E+06, erdp_coords_australia, elev_coords_australia, final_noise)

	elif choice == 2:
		directory = 'Examples/aus'
		print '%s/aus.xml' %(directory)
		topoGenerator(directory,'%s/aus.xml' %(directory), 1.0 , 5.e-7, 0.5, 1, -1.49E08, erdp_coords_australia, elev_coords_australia, final_noise)

	elif choice == 3:
		directory = 'Examples/aus'
		print '%s/aus_present.xml' %(directory)
		topoGenerator(directory,'%s/aus.xml' %(directory), 1.0 , 5.e-7, 0.5, 1, -100, erdp_coords_australia, elev_coords_australia, final_noise)

	tend = time.time()
	print 'TopoGen for aus completed in (s):',tend-tstart

if __name__ == "__main__": main()