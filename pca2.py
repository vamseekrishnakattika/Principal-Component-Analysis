import numpy as np
import sys

if len(sys.argv) != 5 :
  print(sys.argv[0], "takes 4 arguments. Not ", len(sys.argv)-1)
  sys.exit()

dataFile=open(sys.argv[1],"r")
labelsFile=open(sys.argv[2],"r")
vectorsFile=sys.argv[3]
reducedDataFile=sys.argv[4]

data=np.loadtxt(dataFile,delimiter=',')
labels=np.loadtxt(labelsFile)

mu1 = np.mean(data, axis=0)

meanSubData=data-mu1

C = np.dot(meanSubData.T,meanSubData)

evals,evecs = np.linalg.eig(C) 
# eigenvalues in increasing order, not decreasing order. Sort them.

idx = np.argsort(evals)[::-1] # sort in reverse order
evals = evals[idx]
evecs = evecs[:,idx] # evectors are the cols of evecs

# extract the 2 dominant eigenvectors
r = 2
V_r = evecs[:,:r]

V_rt = V_r.T

reducedData = np.dot(meanSubData,V_r)

np.savetxt(vectorsFile, V_rt, delimiter=',')

np.savetxt(reducedDataFile, reducedData, delimiter=',')