# Import badlands grid generation toolbox
import pybadlands_companion.toolGrid as tools
import sys
import basemap
def main():
	grid = tools.toolGrid(llcrnrlon = 111.50, llcrnrlat = -39.00, urcrnrlon = 159.50, urcrnrlat = -9.50)
	grid.plotEPSG( epsg=3857, llspace=0.25, fsize=(8,8), title = 'Map EPSG::3857 Australia and PNG' )
	grid.getSubset(tfile = 'etopo1', offset = 0.1, smooth = False)
	grid.mapUTM(contour=50, fsize=(8,8), saveFig=False, nameFig='map')
	grid.buildGrid(resolution=50000., method='cubic', nameCSV='australia_xyz50k')
	grid.viewGrid(width=900, height=1000, zmin=0, zmax=1500, title='Export Grid')

if __name__ == "__main__": main()