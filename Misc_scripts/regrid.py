import numpy as np 

def convertInitialTXT_CSV(directory,fname, res_fact, reduce_factor):
	
	arr = np.loadtxt(fname)
	with open('%s/data/convertedInitial_low.csv' %(directory), 'a') as the_file:
		for i in xrange(0, arr.shape[0]-2, reduce_factor):
			for j in xrange(0, arr.shape[1]-2, reduce_factor):
				x_c = i*res_fact
				y_c = j*res_fact

				line = str(float(y_c)) + ' ' + str(float(x_c))+ ' ' + str(float("{0:.2f}".format(arr[i,j]))) +  '\n'
				the_file.write(line)

	print('Original Array shape : ', arr.shape)
	conv = np.loadtxt('%s/data/convertedInitial_low.csv' %(directory))
	print('Converted Array shape: ', conv.shape)

def main():
	directory = 'Examples/aus_1m/'
	convertInitialTXT_CSV(directory, '%s/data/initial_elev.txt'%(directory), res_fact = 50, reduce_factor = 2)

if __name__ == "__main__": main()