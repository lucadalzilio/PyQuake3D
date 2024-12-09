import ctypes
from math import *
import numpy as np
import gc

def readPara0(fname):
    Para0={}
    f=open(fname,'r')
    for line in f:
        #print(line)
        if ':' in line:
            tem=line.split(':')
            #if(tem[0]==)
            Para0[tem[0].strip()]=tem[1].strip()
    return Para0

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


# 加载共享库
lib = ctypes.CDLL("src/hm.so")

# 定义C函数参数和返回类型
lib.createHmatrixstructure.argtypes = [ctypes.POINTER(prStruct)]
lib.createHmatrixstructure.restype = None

lib.create_Hmvalue.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.create_Hmvalue.restype = None

lib.Hmatrix_dot_X.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
lib.Hmatrix_dot_X.restype = ctypes.POINTER(ctypes.c_float)


def create_hmatrix_structure(gr):
    pr_python = prStruct()
    pr_python.vertices = gr['vertices']
    pr_python.edges = gr['edges']
    pr_python.triangles = gr['triangles']

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


def create_Hmvalue(Adata):
    A1d_data = Adata.astype(np.float32).flatten()
    A1d_data=A1d_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.create_Hmvalue(A1d_data)


def Hmatrix_dot_X(indexk,x):
    X_vector=x.astype(np.float32).flatten()
    X_vector=X_vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    yh=lib.Hmatrix_dot_X(indexk,X_vector)
    array = [yh[i] for i in range(x.shape[0])]
    return np.array(array)
    

