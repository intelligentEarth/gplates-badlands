import numpy as np 

def convertToExpert(directory,fname, grid_size, surround_pts):
	
	arr = np.loadtxt(fname)
	grid_len = int(np.sqrt(grid_size))
	m = (arr.shape[0])/(grid_len)
	n = (arr.shape[1])/(grid_len)
	with open('%s/data/inittopogrid.csv' %(directory), 'a') as the_file:
		
		for i in xrange(0, arr.shape[0], m):
			for j in xrange(0, arr.shape[1], n):
				
				print '(i , j)', i , j
				line = str(i) + ' ' + str(j) +  '\n'
				the_file.write(line)

	print('Original Array shape : ', arr.shape)
	conv = np.loadtxt('%s/data/inittopogrid.csv' %(directory))
	print('Converted Array shape: ', conv.shape)

def main():
	directory = 'Examples/carmen_australia/'
	convertToExpert(directory, '%s/data/initial_elev.txt'%(directory), grid_size = 100, surround_pts = 1)

if __name__ == "__main__": main()
