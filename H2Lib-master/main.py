import ctypes
from math import *

# 加载共享库
lib = ctypes.CDLL("./hm.so")



pi=3.1415926


def read_mshV2(fname):  #read msh file
    f=open(fname,'r')
    start_node=0
    start_ele=0
    K1=0
    node=[]
    ele=[]
    for line in f:
        line=line[:-1]
        
        if(start_node==1):
            K1=K1+1
            sliptdata=line.split()
            #print(sliptdata)
            if(K1>1):
                try:
                         
                    nodeline=[]
                    tem=float(line.split()[1])
                    nodeline.append(tem)
                    tem=float(line.split()[2])
                    nodeline.append(tem)
                    tem=float(line.split()[3])
                    nodeline.append(tem)
                    node.append(nodeline)
                    #print(nodeline)
                except:
                    start_node=0
                    K1=0
        if(start_ele==1):
            K1=K1+1
            sliptdata=line.split()
            if(K1>1):
                try:
                    if(len(line.split())>=8): 
                        eleline=[]
                        tem=int(sliptdata[-3])
                        eleline.append(tem)
                        tem=int(sliptdata[-2])
                        eleline.append(tem)
                        tem=int(sliptdata[-1])
                        eleline.append(tem)
                        ele.append(eleline)
                        #print(eleline)
                except:
                    start_ele=0

        if(line=='$Nodes'):
            start_node=1
            
        if(line=='$Elements'):
            start_ele=1
            #print(start_ele)
        
        #boundary_edges,boundary_nodes=find_boundary_edges_and_nodes(ele)

    return np.array(node)*1e3,np.array(ele)


# fname='1.msh'
# nodelst,elelst=read_mshV2(fname)
# print(nodelst.shape,elelst.shape)

def get_eleVec(nodelst,elelst,jud_ele_order):
    eleVec=[]
    xg=[]
    for i in range(len(elelst)):
        xa=nodelst[elelst[i,0]-1]
        xb=nodelst[elelst[i,1]-1]
        xc=nodelst[elelst[i,2]-1]

        # if(i==12):
        #     plt.scatter(xa[0],xa[1],color='r')  #查看节点逆时针还是顺时针
        #     plt.scatter(xb[0],xb[1],color='b')
        #     plt.scatter(xc[0],xc[1],color='y')
        #     plt.show()

        xg.append([np.mean([xa[0],xb[0],xc[0]]),np.mean([xa[1],xb[1],xc[1]]),np.mean([xa[2],xb[2],xc[2]])])

        vba=xb-xa
        vca=xc-xa

       

        
        if(jud_ele_order==True): 
            #节点顺时针ac*ab
            ev31 = vca[1]*vba[2]-vca[2]*vba[1]
            ev32 = vca[2]*vba[0]-vca[0]*vba[2]
            ev33 = vca[0]*vba[1]-vca[1]*vba[0]
        else:
            #节点逆时针WMF
            ev31 = vba[1]*vca[2]-vba[2]*vca[1]
            ev32 = vba[2]*vca[0]-vba[0]*vca[2]
            ev33 = vba[0]*vca[1]-vba[1]*vca[0]
        rr = sqrt(ev31*ev31+ev32*ev32+ev33*ev33)
        # unit vectors for local coordinates of elements
        ev31 /=rr
        ev32 /=rr
        ev33 /= rr

        if( abs(ev33) < 1 ):
            ev11 = ev32
            ev12 = -ev31
            ev13 = 0 
            rr = sqrt(ev11*ev11 + ev12*ev12) 
            ev11 /=rr
            ev12 /=rr
        
        else:
            ev11= 1
            ev12= 0
            ev13= 0
        

        ev21 = ev32*ev13-ev33*ev12
        ev22 = ev33*ev11-ev31*ev13
        ev23 = ev31*ev12-ev32*ev11
        eleVec.append([ev11,ev12,ev13,ev21,ev22,ev23,ev31,ev32,ev33])
    eleVec=np.array(eleVec)
    xg=np.array(xg)
    #print(eleVec)
    return eleVec,xg





def transgr(nodelst,elelst,eleVec,xg):
    gr={}
    edges = set()
    for index, element in enumerate(elelst):
        # 为每个三角形单元定义三条边
        edges.add((element[0], element[1]))
        edges.add((element[1], element[2]))
        edges.add((element[2], element[0]))

    # 打印唯一的边
    edges_lst=list(edges)
    edges_npy=np.array(edges_lst)

    #indices = np.where(np.all(edges_npy == [610,832], axis=1))[0]
    gr['vertices'],gr['edges'],gr['triangles']=len(nodelst),len(edges_lst),len(elelst)
    gr['x']=nodelst
    gr['e']=edges_npy
    gr['t']=elelst
    s=[]
    for i in range(len(elelst)):

        indices0 = np.where(np.all(edges_npy == elelst[i,[1,2]], axis=1))[0]
        indices1 = np.where(np.all(edges_npy == elelst[i,[2,0]], axis=1))[0]
        indices2 = np.where(np.all(edges_npy == elelst[i,[0,1]], axis=1))[0]

        if(len(indices0)!=1 or len(indices1)!=1 or len(indices2)!=1):
            print('error!!!!!!!!:',i,indices0,indices1,indices2)
            break
        s.append([indices0[0],indices1[0],indices2[0]])
    gr['s']=np.array(s)

    #print(gr)
    return gr


import numpy as np
import hmatrix
#fnamegeo='1/cascadia35km_ele4.msh'
fnamegeo='bp5t.msh'
nodelst,elelst=read_mshV2(fnamegeo)
eleVec,xg=get_eleVec(nodelst,elelst,jud_ele_order=False)
gr=transgr(nodelst,elelst,eleVec,xg)

hmatrix.create_hmatrix_structure(gr)

A1s=np.load('../bp5t_core/A1s.npy')
hmatrix.create_Hmvalue(A1s)

A1d=np.load('../bp5t_core/A1d.npy')
hmatrix.create_Hmvalue(A1d)

A2d=np.load('../bp5t_core/A2d.npy')
hmatrix.create_Hmvalue(A2d)

x=np.ones(A1d.shape[0])*0.001

import time
start_time = time.time()
y1s=hmatrix.Hmatrix_dot_X(0,x)
# y1d=hmatrix.Hmatrix_dot_X(1,x)
# y2d=hmatrix.Hmatrix_dot_X(2,x)
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print(f"代码执行时间为: {execution_time} 秒")

# ytem=np.dot(A1s,x)
# print(np.max(ytem),np.min(ytem))
# print(np.max(y1s),np.min(y1s))

# start_time = time.time()
# y1s=hmatrix.Hmatrix_dot_X(0,x)
# y1d=hmatrix.Hmatrix_dot_X(1,x)
# y2d=hmatrix.Hmatrix_dot_X(2,x)
# end_time = time.time()

# # 计算执行时间
# execution_time = end_time - start_time
# print(f"代码执行时间为: {execution_time} 秒")

# ytem=np.dot(A1s,x)
# print(np.max(ytem),np.min(ytem))
# print(np.max(y1s),np.min(y1s))

# start_time = time.time()
# y1s=hmatrix.Hmatrix_dot_X(0,x)
# y1d=hmatrix.Hmatrix_dot_X(1,x)
# y2d=hmatrix.Hmatrix_dot_X(2,x)
# end_time = time.time()

# # 计算执行时间
# execution_time = end_time - start_time
# print(f"代码执行时间为: {execution_time} 秒")


start_time = time.time()
ytem=np.dot(A1s,x)
# print(np.max(ytem),np.min(ytem))
# print(np.max(y1s),np.min(y1s))
end_time = time.time()
execution_time = end_time - start_time
print(f"代码执行时间为: {execution_time} 秒")

print(y1s[:10])
print(ytem[:10])
'''
class prStruct(ctypes.Structure):
    _fields_ = [
        ("vertices", ctypes.c_int),
        ("edges", ctypes.c_int),
        ("triangles", ctypes.c_int),
        ("x", ctypes.POINTER(ctypes.c_float)),
        ("e", ctypes.POINTER(ctypes.c_int)),
        ("t", ctypes.POINTER(ctypes.c_int)),
        ("s", ctypes.POINTER(ctypes.c_int)),
    ]

# 帮助函数：将numpy数组转换为指针
# def numpy_to_pointer(arr):
#     arr = np.ascontiguousarray(arr, dtype=np.float32)
#     pointer_array = (ctypes.POINTER(ctypes.c_float) * len(arr))()
#     for i in range(len(arr)):
#         pointer_array[i] = arr[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#     return pointer_array

# 定义 numpy_to_pointer 函数
# def numpy_to_pointer(arr):
#     # 确保数组是连续的，并且是 float32 类型
#     arr = np.ascontiguousarray(arr, dtype=np.float32)
    
#     # 检查数组是否是二维的
#     if arr.ndim != 2:
#         raise ValueError("Input array must be 2-dimensional")
    
#     # 创建指针数组，长度为数组的行数
#     pointer_array = (ctypes.POINTER(ctypes.c_float) * len(arr))()
    
#     # 为每一行生成指针并存储到 pointer_array
#     for i in range(len(arr)):
#         pointer_array[i] = arr[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
#     return pointer_array


# 定义C函数参数和返回类型
lib.createHmatrixstructure.argtypes = [ctypes.POINTER(prStruct)]
lib.createHmatrixstructure.restype = None

pr_python = prStruct()
pr_python.vertices = gr['vertices']
pr_python.edges = gr['edges']
pr_python.triangles = gr['triangles']


import gc
# 禁用垃圾回收
gc.disable()

# 将numpy数组转为指针并赋值给结构体

x_data = gr['x'].astype(np.float32).flatten()
e_data = gr['e'].astype(np.int32).flatten()
t_data = gr['t'].astype(np.int32).flatten()
s_data = gr['s'].astype(np.int32).flatten()

#arrtem=gr['x'].flatten().astype(np.float32)
pr_python.x=x_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
# arrtem=gr['e'].flatten().astype(np.float32)
pr_python.e=e_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
pr_python.t=t_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
pr_python.s=s_data.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


#lib.print_array(gr['x'].astype(np.float32).flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
# lib.print_array(gr['e'].astype(np.float32).flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
# 重新启用垃圾回收
gc.enable()

lib.createHmatrixstructure(ctypes.byref(pr_python))
#print(gr['s'][:10])


#print(pr_python.s.shape)
# lib.getHmatrix.restype = ctypes.POINTER(ctypes.c_void_p)
# lib.getHmatrix.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float))]

A1d=np.load('core/A1d.npy')
#print(A1d[0,:20],A1d.shape)
A1d_data = A1d.astype(np.float32).flatten()
A1d_data=A1d_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# 定义C函数参数和返回类型
lib.create_Hmvalue.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.create_Hmvalue.restype = None

k=0
lib.create_Hmvalue(A1d_data)

x=np.ones(A1d.shape[0])*0.001
yh=np.dot(A1d,x)
#print(yh[:20])

A2d=np.load('core/A2d.npy')
A2d_data = A2d.astype(np.float32).flatten()
A2d_data=A2d_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

lib.create_Hmvalue(A2d_data)
yh=np.dot(A2d,x)
print(yh[:20])



lib.Hmatrix_dot_X.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
lib.Hmatrix_dot_X.restype = ctypes.POINTER(ctypes.c_float)


X_vector=x.astype(np.float32).flatten()
X_vector=X_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


hk=1
yh=lib.Hmatrix_dot_X(hk,X_vector)

array = [yh[i] for i in range(A2d.shape[0])]
array=np.array(array)
print(array[:20])

'''