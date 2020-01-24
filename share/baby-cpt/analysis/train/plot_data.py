import h5py
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import glob
import preprocess_image

potential_files = glob.glob('out/*.h5')
for pf in potential_files:
	try:
		#gamma_energy = int(pf[pf.rfind('-')+1 : pf.rfind('.')])
		pass
	except ValueError:
		continue
	f = h5py.File(pf, 'r')
	points = np.array(f['position'])
	energy = np.array(f['edep'])
	#plt.scatter(points[:, 0], points[:, 1], c=energy)

	#x_max = 300#= np.max(points[:, 0])+1
	#print(x_max)
	#x_bins = x_max
	#y_max = 156 #= np.max(points[:, 1])+1
	#print(y_max)
	#y_bins = y_max
	#mesh = np.zeros([x_bins, y_bins*2])

	#for i in range(len(points)):
	#	x = int(np.floor(points[i][0]/(x_max/x_bins)))
	#	y = int(np.floor(points[i][1]/(y_max/y_bins)))+y_bins
	#	mesh[x, y]+=energy[i]

	mesh = preprocess_image.get_image(points, energy)

	fig, ax = plt.subplots()
	im = ax.imshow(mesh, interpolation='bilinear', origin='lower')
	CS = ax.contour(mesh)
	#ax.clabel(CS, inline=1, fontsize=10)
	ax.set_title(pf)

	#fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')
	#ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=energy)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	#ax.set_zlabel('Z Label')

	fig.colorbar(im)
	name = pf[ : pf.rfind('.')]
	plt.savefig(name+'-pre.png')
	print('saved figure '+name+'-pre.png')
print('done')
