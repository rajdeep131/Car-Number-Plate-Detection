import numpy as np

def sortedSizeConnectedComponent(labels,num_components):
    tempArr1=[[0,i+1] for i in range(num_components)]
    m=len(labels[:,0])
    n=len(labels[0,:])
    for i in range(m):
        for j in range(n):
            if labels[i][j]==0:
                continue
            tempArr1[labels[i][j]-1][0]+=1
    tempArr1.sort()
    return tempArr1


def sortedXDiff(labels,num_components):
    tempArr1=[[100000000,-1,i+1] for i in range(num_components)]
    m=len(labels[:,0])
    n=len(labels[0,:])
    for i in range(m):
        for j in range(n):
            if labels[i][j]==0:
                continue
            else:
                if tempArr1[labels[i][j]-1][0] > i :
                    tempArr1[labels[i][j]-1][0]=i
                if tempArr1[labels[i][j]-1][1] < i :
                    tempArr1[labels[i][j]-1][1]=i
    
    tempArr2=[[tempArr1[i][1]-tempArr1[i][0],i+1] for i in range(num_components)]
    tempArr2.sort()
    return tempArr2

def sortedYDiff(labels,num_components):
    tempArr1=[[100000000,-1,i+1] for i in range(num_components)]
    m=len(labels[:,0])
    n=len(labels[0,:])
    for i in range(m):
        for j in range(n):
            if labels[i][j]==0:
                continue
            else:
                if tempArr1[labels[i][j]-1][0] > j :
                    tempArr1[labels[i][j]-1][0]=j
                if tempArr1[labels[i][j]-1][1] < j :
                    tempArr1[labels[i][j]-1][1]=j
    
    tempArr2=[[tempArr1[i][1]-tempArr1[i][0],i+1] for i in range(num_components)]
    tempArr2.sort()
    return tempArr2



def outlierRemoval(arr):
    tempArr1=[]
    Q1=arr[int(len(arr)/4)][0]
    Q3=arr[int(3*len(arr)/4)][0]
    IQR=Q3-Q1
    thresh=Q3+(1.5*IQR)
    for el in arr:
        if el[0]<thresh:
            tempArr1.append(el)
    return tempArr1

def filter(szcc,xdiff,ydiff,szccFilter=50,xdiffFilter=10,ydiffFilter=8):
    t_arr_szcc=[]
    for el in szcc:
        if el[0]>szccFilter:
            t_arr_szcc.append(el)
    t_arr_xdiff=[]
    for el in xdiff:
        if el[0]>xdiffFilter:
            t_arr_xdiff.append(el)
    t_arr_ydiff=[]
    for el in ydiff:
        if el[0]>ydiffFilter:
            t_arr_ydiff.append(el)
    
    n_arr_szcc=t_arr_szcc
    n_arr_xdiff=t_arr_xdiff
    n_arr_ydiff=t_arr_ydiff

    n_arr_szcc=outlierRemoval(t_arr_szcc)
    #n_arr_xdiff=outlierRemoval(t_arr_xdiff)
    #n_arr_ydiff=outlierRemoval(t_arr_ydiff)

    t_dict={}

    for el in n_arr_szcc:
        t_dict.setdefault(el[1],[-1,-1,-1])
        t_dict[el[1]][0]=el[0]
    
    for el in n_arr_xdiff:
        t_dict.setdefault(el[1],[-1,-1,-1])
        t_dict[el[1]][1]=el[0]
    
    for el in n_arr_ydiff:
        t_dict.setdefault(el[1],[-1,-1,-1])
        t_dict[el[1]][2]=el[0]
    
    filteredDictionary=filterDictionary(t_dict)

    return filteredDictionary

def filterDictionary(dictionaty):
    t_dict={}
    for key,value in dictionaty.items():
        if -1 in value:
            continue
        t_dict[key]=value
    return t_dict

def normalize(coordiantes):
    mx=[-1,-1,-1]
    mn=[1000000,100000,100000]
    for values in coordiantes.values():
        if values[0]<mn[0]:
            mn[0]=values[0]
        if values[1]<mn[1]:
            mn[1]=values[1]
        if values[0]<mn[0]:
            mn[2]=values[2]
        
        if values[0]>mx[0]:
            mx[0]=values[0]
        if values[1]>mx[1]:
            mx[1]=values[1]
        if values[0]>mx[0]:
            mx[2]=values[2]
    for key,value in coordiantes.items():
        coordiantes[key][0]=(coordiantes[key][0]-mn[0])/(mx[0]-mn[0])
        coordiantes[key][1]=(coordiantes[key][1]-mn[1])/(mx[1]-mn[1])
        coordiantes[key][2]=(coordiantes[key][2]-mn[2])/(mx[2]-mn[2])
    return coordiantes

def showLabel(labeled_img,label_array):
    m=len(labeled_img[:,0])
    n=len(labeled_img[0,:])
    T_IMG=np.zeros((m,n),dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            if labeled_img[i][j] in label_array:
                T_IMG[i][j]=255
    return T_IMG

def distance(arr1,arr2):
    distance=0
    for i in range(len(arr1)):
        distance+=(arr1[i] - arr2[i])**2
    return distance**(0.5)

def sortedScore(coordinates):
    tempArr1=[]
    for key1,values1 in coordinates.items():
        score=0
        for key2,values2 in coordinates.items():
            if key2==key1:
                continue
            else:
                score+=2**(-distance(values1,values2))
        tempArr1.append([score,key1])
    tempArr1.sort(reverse=True)
    return tempArr1

def topkXdiff(coordinates,k):
    arr=[]
    for keys,values in coordinates.items():
        arr.append([values[1],keys])
    arr.sort(reverse=True)
    return arr[:k]

def returnXmin(labels,label_num):
    min=100000000
    coor=[0,0]
    m=len(labels[:,0])
    n=len(labels[0,:])
    for i in range(m):
        for j in range(n):
            if labels[i][j]!=label_num:
                continue
            else:
                if min > i :
                    min=i
                    coor[0]=min
                    coor[1]=j
    return coor

def angle(coor1,coor2):
    return np.arctan2((coor1[1]-coor2[1]),(coor1[0]-coor2[0]))

def bottomkAngulerScore(arr,labels,k):
    coor_arr=[]
    for el in arr:
        coor=returnXmin(labels,el)
        coor_arr.append([coor,el])
    
    temp_arr=[]
    for el1 in coor_arr:
        t_arr=[]
        for el2 in coor_arr:
            if el1[1]==el2[1]:
                continue
            else:
                t_arr.append(angle(el1[0],el2[0]))
        t_arr.sort()
        t_arr2=[]
        for i in range(1,len(t_arr)):
            t_arr2.append(t_arr[i]-t_arr[i-1])
        t_arr2.sort()
        sm=sum(t_arr2[:8])
        temp_arr.append([sm,el1[1]])
    temp_arr.sort()
    return temp_arr[:k]