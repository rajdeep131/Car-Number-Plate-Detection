#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageProcessingUtils as iput
import morphologicalOperatorUtils as mut
import connectedComponentsUtils as ccut
import carNumPlateDetectionUtils as cnput
import cv2
import pytesseract
import os
import shutil
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image as im 

#%%

Loading = mpimg.imread('FrontCapturedImage/img10.jpeg')

#%%
greyscaleImage=iput.convertToGreyscale(Loading,gamma=1.9)
#plt.imshow(greyscaleImage,cmap='gray')


#%%
invertedOtsu=iput.otsuThresholdedImage(greyscaleImage,invert=True)
plt.imshow(invertedOtsu,cmap='grey') 

# %%
labels, num_components = ccut.find_connected_components(invertedOtsu)
#print("Number of connected components:", num_components)

# %%
szcc=cnput.sortedSizeConnectedComponent(labels,num_components)
#print(szcc)
xdiff=cnput.sortedXDiff(labels,num_components)
#print(xdiff)
ydiff=cnput.sortedYDiff(labels,num_components)
#print(ydiff)

# %%
filteredCoordinates=cnput.filter(szcc,xdiff,ydiff,szccFilter=50,xdiffFilter=10,ydiffFilter=5)
print(len(filteredCoordinates))

#%%

plt.imshow(cnput.showLabel(labels,filteredCoordinates.keys()),cmap='grey')

#%%
def calcCentroids(labels,filteredCoordinate):
    tempdict1={}
    m=len(labels[:,0])
    n=len(labels[0,:])
    for i in range(m):
        for j in range(n):
            if labels[i][j]==0 or labels[i][j] not in filteredCoordinate.keys():
                continue
            else:
                tempdict1.setdefault(labels[i][j],[0,0])
                tempdict1[labels[i][j]][0]+=i
                tempdict1[labels[i][j]][1]+=j
    
    for keys,values in tempdict1.items():
        tempdict1[keys]=[int(tempdict1[keys][0]/filteredCoordinate[keys][0]),int(tempdict1[keys][1]/filteredCoordinate[keys][0])]
    
    for key,val in tempdict1.items():
        tempdict1[key]=val+[15*filteredCoordinate[key][1],15*filteredCoordinate[key][2]]
    return tempdict1

#%%
centroids=calcCentroids(labels,filteredCoordinates)
#print(centroids)

#%%
def calcBottomKDistScoreBasedOnCentroid(centroid,k):
    tempArr2=[]
    for key1,val1 in centroid.items():
        tempArr1=[]
        for key2,val2 in centroid.items():
            if key1==key2:
                continue
            else:
                dist=cnput.distance(val1,val2)
                tempArr1.append(dist)
        tempArr1.sort()
        sm=sum(tempArr1[:10])
        tempArr2.append([sm,key1])
    tempArr2.sort()
    return tempArr2[:k]

#%%
sortedCentroidScore=calcBottomKDistScoreBasedOnCentroid(centroids,k=9)
#print(sortedCentroidScore)

#%%
sortedCentroidScoreCoordinate=[el[1] for el in sortedCentroidScore]

# %%

plt.imshow(cnput.showLabel(labels,sortedCentroidScoreCoordinate),cmap='grey') 

# %%
def saveNumberplateCharacters(labels,arr):
    tempArr1={el:[100000000,-1] for el in arr}
    m=len(labels[:,0])
    n=len(labels[0,:])
    for i in range(m):
        for j in range(n):
            if labels[i][j]==0 or labels[i][j] not in arr:
                continue
            else:
                if tempArr1[labels[i][j]][0] > i :
                    tempArr1[labels[i][j]][0]=i
                if tempArr1[labels[i][j]][1] < i :
                    tempArr1[labels[i][j]][1]=i
    
    tempArr2={el:[100000000,-1] for el in arr}
    m=len(labels[:,0])
    n=len(labels[0,:])
    for i in range(m):
        for j in range(n):
            if labels[i][j]==0 or labels[i][j] not in arr:
                continue
            else:
                if tempArr2[labels[i][j]][0] > j :
                    tempArr2[labels[i][j]][0]=j
                if tempArr2[labels[i][j]][1] < j :
                    tempArr2[labels[i][j]][1]=j
    tempdict={}
    for el in arr:
        height=tempArr1[el][1]-tempArr1[el][0]+1+6
        width=tempArr2[el][1]-tempArr2[el][0]+1+6
        tempImg=np.zeros((height,width),dtype=np.uint8)
        for i in range(tempArr1[el][0],tempArr1[el][1]+1):
            for j in range(tempArr2[el][0],tempArr2[el][1]+1):
                if labels[i][j]!=el:
                    continue
                else:
                    tempImg[i-tempArr1[el][0]+3][j-tempArr2[el][0]+3]=255
        tempdict[el]=tempImg
    
    temparr=[[tempArr2[el][0],el] for el in tempArr2.keys()]
    temparr.sort()
    charOrder=[el[1] for el in temparr]

    return (tempdict,charOrder)

#%%
CroppedCharacters,characterOrder=saveNumberplateCharacters(labels,sortedCentroidScoreCoordinate)
#print(characterOrder)
# %%
#plt.imshow(cnput.showLabel(labels,[223]),cmap='grey')

#%%
def clear_folder_contents(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
        
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) 
    else:
        print(f"The folder {folder_path} does not exist.")
#%%

def saveCharacterImage(croppedChar,charOrder):
    clear_folder_contents("CharImg")
    for i in range(len(charOrder)):
        el=charOrder[i]
        data = im.fromarray(iput.invertBinaryImage(croppedChar[el]))
        data.save('CharImg/'+str(i)+'.png') 

#%%
saveCharacterImage(CroppedCharacters,characterOrder)
# %%
#plt.imshow(CroppedCharacters[194],cmap='grey')



#%%
def showNumberPlate(path):
    arr=[int(el[0]) for el in os.listdir(path)]
    arr.sort()
    temparr1=[]
    for el in arr:
        image_path = path+str(el)+'.png'
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        letter = pytesseract.image_to_string(thresh, config='--psm 10')
        temparr1.append(letter.strip())
    return '-'.join(temparr1)

#%%
print(showNumberPlate('CharImg/'))


# %%
