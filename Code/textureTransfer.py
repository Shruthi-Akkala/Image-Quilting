import numpy as np
from PIL import Image
from minimumCostPathFunc import minimumCostMask 


def texture_transfer(textureImg, targetImg, blockSize, overlapSize, alpha, tolerance):
    textureImg = np.array(textureImg)
    targetImg = np.array(targetImg)
    # print(textureImgArray.shape, targetImgArray.shape)
    X_out = targetImg.shape[0]
    Y_out = targetImg.shape[1]
    [m,n,c] = textureImg.shape
    blocks = []
    for i in range(m-blockSize[0]):
        for j in range(n-blockSize[1]):
            #blocks are added to a list
            blocks.append(textureImg[i:i+blockSize[0],j:j+blockSize[1],:])                              
    blocks = np.array(blocks)
    #final image is initialised with elemnts as -1.
    finalImg = np.ones([X_out, Y_out, c])*-1
    finalImg[0:blockSize[0],0:blockSize[1],:] = textureImg[0:blockSize[0],0:blockSize[1],:]
    noOfBlocksInRow = 1+np.ceil((X_out - blockSize[1])*1.0/(blockSize[1] - overlapSize))
    noOfBlocksInCol = 1+np.ceil((Y_out - blockSize[0])*1.0/(blockSize[0] - overlapSize))
    for i in range(int(noOfBlocksInRow)):
        for j in range(int(noOfBlocksInCol)):
            if i == 0 and j == 0:
                continue
            #start and end location of block to be filled is initialised
            startX = int(i*(blockSize[0] - overlapSize))
            startY = int(j*(blockSize[1] - overlapSize))
            endX = int(min(startX+blockSize[0],X_out))
            endY = int(min(startY+blockSize[1],Y_out))
            # print(startX, endX, startY, endY)

            toFill = finalImg[startX:endX,startY:endY,:]
            targetBlock = targetImg[startX:endX,startY:endY,:]
            #print(targetBlock.shape==blocks.shape[1:])
            
            if targetBlock.shape != blocks.shape[1:]:
                blocks1 = []
                for x in range(m - targetBlock.shape[0]):
                    for y in range(n - targetBlock.shape[1]):
                        blocks1.append(textureImg[x:x+targetBlock.shape[0],y:y+targetBlock.shape[1],:]) 
                blocks1 = np.array(blocks1)
                matchBlock = MatchBlock(blocks1, toFill, targetBlock, blockSize, alpha, tolerance)
            # print(toFill.shape, targetBlock.shape)
            #MatchBlock returns the best suited block
            else:
                matchBlock = MatchBlock(blocks, toFill, targetBlock, blockSize, alpha, tolerance)

            B1EndY = startY+overlapSize-1
            B1StartY = B1EndY-(matchBlock.shape[1])+1
            B1EndX = startX+overlapSize-1
            B1StartX = B1EndX-(matchBlock.shape[0])+1

            if i == 0:      
                overlapType = 'v'
                B1 = finalImg[startX:endX,B1StartY:B1EndY+1,:]
                #print(B1.shape,matchBlock.shape,'v',B1StartY,B1EndY,startX,startY)
                mask = minimumCostMask(matchBlock[:,:,0],B1[:,:,0],0,overlapType,overlapSize)

            elif j == 0:          
                overlapType = 'h'
                B2 = finalImg[B1StartX:B1EndX+1, startY:endY, :]
                #print(B2.shape,matchBlock.shape,B1StartX,B1EndY)
                mask = minimumCostMask(matchBlock[:,:,0],0,B2[:,:,0],overlapType,overlapSize)
            else:
                overlapType = 'b'
                B1 = finalImg[startX:endX,B1StartY:B1EndY+1,:]
                B2 = finalImg[B1StartX:B1EndX+1, startY:endY, :]
                #print(B1.shape,B2.shape,matchBlock.shape)

                mask = minimumCostMask(matchBlock[:,:,0],B1[:,:,0],B2[:,:,0],overlapType,overlapSize)
                
            mask = np.repeat(np.expand_dims(mask,axis=2),3,axis=2)
            maskNegate = mask==0
            finalImg[startX:endX,startY:endY,:] = maskNegate*finalImg[startX:endX,startY:endY,:]
            finalImg[startX:endX,startY:endY,:] = matchBlock*mask+finalImg[startX:endX,startY:endY,:]

            completion = 100.0/noOfBlocksInRow*(i + j*1.0/noOfBlocksInCol);
            print("{0:.2f}% complete...".format(completion), end="\r", flush=True)

            if endY == Y_out:
                break
        if endX == X_out:
            print("100% complete!", end="\r", flush = True)
            break
    return finalImg

def SSDError(Bi, toFill, targetBlock, alpha): 
    [m,n,p] = toFill.shape
    #blocks to be searched are cropped to the size of empty location
    Bi = Bi[0:m,0:n,0:p]
    #Locations where toFill+1 gives 0 are those where any data is not stored yet. Only those which give greater than 1 are compared for best fit.

    lum_Bi = np.sum(Bi, axis = 2)*1.0/3
    lum_target = np.sum(targetBlock, axis = 2)*1.0/3
    lum_toFill = np.sum(toFill, axis = 2)*1.0/3
    error = alpha*np.sqrt(np.sum(((toFill+0.99)>0.1)*(Bi - toFill)*(Bi - toFill))) + (1-alpha)*np.sqrt(np.sum(((lum_toFill+0.99)>0.1)*(lum_Bi - lum_target)*(lum_Bi - lum_target)))
    return [error]

def MatchBlock(blocks, toFill, targetBlock, blockSize, alpha, tolerance):
    error = []
    [m,n,p] = toFill.shape
    bestBlocks = []
    count = 0
    for i in range(blocks.shape[0]):
        #blocks to be searched are cropped to the size of empty location
        Bi = blocks[i,:,:,:]
        Bi = Bi[0:m,0:n,0:p]
        error.append(SSDError(Bi, toFill, targetBlock, alpha))
    minVal = np.min(error)
    for i in range(blocks.shape[0]):
        if error[i] <= (1.0+tolerance)*minVal:
            block = blocks[i,:,:,:]
            bestBlocks.append(block[0:m,0:n,0:p])
            count = count+1
    c = np.random.randint(count)
    return bestBlocks[c]