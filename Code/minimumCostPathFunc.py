import copy
import math
import numpy as np

def minimumCostMask(Ref, B1, B2, overlap_type, overlap_size):
    """
    Ref : numpy array => block to be concatenated to B1 and B2
    B1  : numpy array => block present to the left of Ref (if present)
    B1  : numpy array => block present to the above of Ref (if present)
    
    B1 and B2 are already present, we're trying to add Ref
    Regions of overlap will have best of Ref and other block
    To highlight the parts of Ref lost, the numpy.ones() array
    ref_mask will denote those pixels as 0.
    Placement is as follows:
        __ B2
        B1 Ref
         (or)
        B2
        Ref
         (or)
        B1 Ref
    overlap_type: Type of overlap i.e only B1 is present or B2 is present or both B1, B2 are present
    overlap_size: Number of layers to overlap
    Return value: ref_mask is a numpy array 
    """
    ref_mask = np.ones(Ref.shape)
    #vertical
    if overlap_type=='v':
        arr = np.power(B1[:,-overlap_size:]-Ref[:,0:overlap_size], 2).tolist()
        ref_mask[:,0:overlap_size] = minimumCostPathOnArray(arr)

    #horizontal
    elif overlap_type=='h':
        arr = np.power(B2[-overlap_size:, :]-Ref[0:overlap_size, :], 2)
        arr = arr.transpose()
        arr = arr.tolist()
        ref_mask[0:overlap_size,:] = minimumCostPathOnArray(arr).transpose()
    #both
    elif overlap_type=='b':
        # Vertical overlap
        arrv = np.power(B1[:,-overlap_size:]-Ref[:,0:overlap_size], 2).tolist()
        ref_mask[:,0:overlap_size] = minimumCostPathOnArray(arrv)
        # Horizontal overlap
        arrh = np.power(B2[-overlap_size:, :]-Ref[0:overlap_size, :], 2)
        arrh = arrh.transpose()
        arrh = arrh.tolist()
        ref_mask[0:overlap_size,:] = ref_mask[0:overlap_size,:]*(minimumCostPathOnArray(arrh).transpose())
        # To ensure that 0's from previous assignment to ref_mask remain 0's
    else:
        print("Error in min path")

    return ref_mask

def minimumCostPathOnArray(block):
    """
    Standard array 'arr' is traversed top to bottom in minimum cost path
    Return value: arr_mask
    """
    arr_mask = np.ones(np.array(block).shape)

    rows = len(block)
    cols = len(block[0])

    for i in range(1,rows):
        block[i][0] = block[i][0] + min(block[i-1][0], block[i-1][1])
        for j in range(1, cols-1):
            block[i][j] = block[i][j] + min(block[i-1][j-1], block[i-1][j], block[i-1][j+1])
        block[i][cols-1] = block[i][cols-1] + min(block[i-1][cols-2], block[i-1][cols-1])

    min_index = [0]*rows
    min_cost = min(block[-1])
    for k in range(1,cols-1):
        if block[-1][k] == min_cost:
            min_index[-1] = k

    for i in range(rows-2, -1, -1):
        j = min_index[i+1]
        lower = 0
        upper = 1 # Bounds for the case j=1
        
        if j==cols-1:
            lower = cols-2
            upper = cols-1
        elif j>0:
            lower = j-1
            upper = j+1
        
        min_cost = min(block[i][lower:upper+1])
        for k in range(lower, upper+1):
            if block[i][k] == min_cost:
                min_index[i] = k


    path = []
    for i in range(0, rows):
        arr_mask[i,0:min_index[i]] = np.zeros(min_index[i])
        path.append((i+1, min_index[i]+1))
    # print("Minimum cost path is: ")
    # print(path)
    return arr_mask
# Uncomment below lines to run this as stand-alone file

#arr = np.random.rand(15,15)
#print(minimumCostMask(arr, arr, arr, 'b', 7))