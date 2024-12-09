import ctypes
from math import *
import numpy as np
import gc


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
lib = ctypes.CDLL("./hm.so")

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
    

