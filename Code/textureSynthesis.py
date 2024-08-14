from PIL import Image
import numpy as np
from minimumCostPathFunc import minimumCostMask 

def Construct(img, Bsize, overlap, X_Out, Y_Out, tolerance):
    img = np.array(img)
    [m,n,c] = img.shape
    blocks = []
    for i in range(0,m-Bsize[0]):
        for j in range(0,n-Bsize[1]):
            #blocks are added to a list
            blocks.append(img[i:i+Bsize[0],j:j+Bsize[1],:])                              
    blocks = np.array(blocks)
    #final image is initialised with elemnts as -1.
    finalImage = np.ones([X_Out, Y_Out, c])*-1
    # finalImage[0:Bsize[0],0:Bsize[1],:] = img[0:Bsize[0],0:Bsize[1],:]
    finalImage[0:Bsize[0],0:Bsize[1],:] = blocks[np.random.randint(len(blocks))]
    BlocksInARow = 1 + np.ceil((X_Out - Bsize[1])*1.0/(Bsize[1] - overlap))
    BlocksInACol = 1 + np.ceil((Y_Out - Bsize[0])*1.0/(Bsize[0] - overlap))
    for i in range(int(BlocksInARow)):
        for j in range(int(BlocksInACol)):
            if i == 0 and j == 0:
                continue
            #start and end location of block to be filled is initialised
            block_row_start = int(i*(Bsize[0] - overlap))
            block_col_start = int(j*(Bsize[1] - overlap))
            block_row_ends = int(min(block_row_start+Bsize[0],X_Out))
            block_col_ends = int(min(block_col_start+Bsize[1],Y_Out))

            toFill = finalImage[block_row_start:block_row_ends,block_col_start:block_col_ends,:]
            #MatchBlock returns the best suited block
            
            matchBlock = MatchBlock(blocks, toFill, Bsize, tolerance)

            B1block_col_ends = block_col_start+overlap-1
            B1block_col_start = B1block_col_ends-(matchBlock.shape[1])+1
            B1block_row_ends = block_row_start+overlap-1
            B1block_row_start = B1block_row_ends-(matchBlock.shape[0])+1
            #print(B1block_col_start,B1block_col_ends,block_row_start,block_col_start)
            if i == 0:      
                overlapType = 'v'
                B1 = finalImage[block_row_start:block_row_ends,B1block_col_start:B1block_col_ends+1,:]
                #print(B1.shape,matchBlock.shape,'v',B1block_col_start,B1block_col_ends,block_row_start,block_col_start)
                mask = minimumCostMask(matchBlock[:,:,0],B1[:,:,0],0,overlapType,overlap)
            elif j == 0:          
                overlapType = 'h'
                B2 = finalImage[B1block_row_start:B1block_row_ends+1, block_col_start:block_col_ends, :]
                #print(B2.shape,matchBlock.shape,B1block_row_start,B1block_col_ends)
                mask = minimumCostMask(matchBlock[:,:,0],0,B2[:,:,0],overlapType,overlap)
            else:
                overlapType = 'b'
                B1 = finalImage[block_row_start:block_row_ends,B1block_col_start:B1block_col_ends+1,:]
                B2 = finalImage[B1block_row_start:B1block_row_ends+1, block_col_start:block_col_ends, :]
                #print(B1.shape,B2.shape,matchBlock.shape)
                mask = minimumCostMask(matchBlock[:,:,0],B1[:,:,0],B2[:,:,0],overlapType,overlap)
            #print(mask)
            mask = np.repeat(np.expand_dims(mask,axis=2),3,axis=2)
            maskNegate = mask==0
            finalImage[block_row_start:block_row_ends,block_col_start:block_col_ends,:] = maskNegate*toFill
            finalImage[block_row_start:block_row_ends,block_col_start:block_col_ends,:] = matchBlock*mask+toFill
            completion = 100.0/BlocksInARow*(i + j*1.0/BlocksInACol)

            print("{0:.2f}% complete...".format(completion), end="\r", flush=True)
            if block_col_ends == Y_Out:
                break

        if block_row_ends == X_Out:
            print("100% complete!\n")
            break
    return finalImage

def SSDError(Bi, toFill): 
    error = np.sum(((toFill+0.99)>0.1)*(Bi - toFill)*(Bi - toFill))
    return [error]

def MatchBlock(blocks, toFill, Bsize, tolerance):   
    error = []
    [m,n,p] = toFill.shape
    bestBlocks = []
    count = 0
    for i in range(blocks.shape[0]):
        #blocks to be searched are cropped to the size of empty location
        Bi = blocks[i,:,:,:]
        Bi = Bi[0:m,0:n,0:p]
        error.append(SSDError(Bi, toFill))
    minVal = np.min(error)
    #bestBlocks = [block[:m, :n, :p] for i, block in enumerate(blocks) if error[i] <= (1.0+tolerance)*minVal]
    for i in range(blocks.shape[0]):
             if error[i] <= (1.0+tolerance)*minVal:
                  block = blocks[i,:,:,:]
                  bestBlocks.append(block[0:m,0:n,0:p])
                  count = count+1
    c = np.random.randint(count)
    return bestBlocks[c]

