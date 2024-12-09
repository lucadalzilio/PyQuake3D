import numpy as np
import struct
import matplotlib.pyplot as plt
from math import *
import SH_greenfunction
import DH_greenfunction
import os
import sys
#import json
from concurrent.futures import ProcessPoolExecutor
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import griddata
import readmsh





def get_sumS(X,Y,Z,nodelst,elelst):
    Ts,Ss,Ds=0,0,1
    mu=0.33e11
    lambda_=0.33e11
    Strs=[]
    Stra=[]
    Dis=[]
    for i in range(len(elelst)):
        P1=np.copy(nodelst[elelst[i,0]-1])
        P2=np.copy(nodelst[elelst[i,1]-1])
        P3=np.copy(nodelst[elelst[i,2]-1])
        Stress,Strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,mu,lambda_)

        P1=np.copy(nodelst[elelst[i,0]-1])
        P2=np.copy(nodelst[elelst[i,1]-1])
        P3=np.copy(nodelst[elelst[i,2]-1])
        ue,un,uv=DH_greenfunction.TDdispHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,0.25)
        
        Dis_tems=np.array([ue,un,uv])
        #print(ue.shape,un.shape)
        if(len(Strs)==0):
            Strs=Stress
            Stra=Strain
            Dis=Dis_tems
        else:
            Strs=Strs+Stress
            Stra=Stra+Strain
            Dis=Dis+Dis_tems
    return Dis,Strs,Stra

def find_boundary_edges_and_nodes(triangles):
    from collections import defaultdict
    edge_count = defaultdict(int)
    boundary_nodes = set()

    # 遍历每个三角形，统计边的出现次数
    for tri in triangles:
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ]
        for edge in edges:
            edge_count[edge] += 1

    # 找到只出现一次的边，并记录边上的节点
    boundary_edges = []
    for edge, count in edge_count.items():
        if count == 1:
            boundary_edges.append(edge)
            boundary_nodes.update(edge)

    return boundary_edges, np.array(list(boundary_nodes))

from scipy.spatial.distance import cdist

def find_min_euclidean_distance(coords1, coords2):
    # 使用 scipy.spatial.distance.cdist 计算成对距离
    distances = cdist(coords1, coords2, 'euclidean')
    # 找到最小距离及其对应的索引
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    min_distance = distances[min_idx]
    return min_distance

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


class QDsim:
    def __init__(self,elelst,nodelst,fnamePara):
        #for i in range(len(xg)):


        last_backslash_index = fnamePara.rfind('/')

        # 获取最后一个反斜杠之前的所有内容
        if last_backslash_index != -1:
            self.dirname = fnamePara[:last_backslash_index]
        else:
            self.dirname = fnamePara
        #print(self.dirname)
        self.Para0=readPara0(fnamePara)
        #jud_ele_order=self.Para0['Node_order']=='True'
        eleVec,xg=readmsh.get_eleVec(nodelst,elelst)
        self.eleVec=eleVec
        self.elelst=elelst
        self.nodelst=nodelst
        self.xg=xg

        

        self.mu=float(self.Para0['Shear modulus'])
        self.lambda_=float(self.Para0['Lame constants'])
        self.density=float(self.Para0['Rock density'])
        self.htry=1e-3
        self.Cs=sqrt(self.mu/self.density)
        self.time=0
        self.halfspace_jud=self.Para0['Half space']=='True'
        self.InputHetoparamter=self.Para0['InputHetoparamter']=='True'
        self.num_process=int(self.Para0['Processors'])
        self.Batch_size=int(self.Para0['Batch_size'])
        YoungsM=self.mu*(3.0*self.lambda_+2.0*self.mu)/(self.lambda_+self.mu)
        possonratio=self.lambda_/2.0/(self.lambda_+self.mu)

        print('Cs',self.Cs)
        print('First Lamé constants',self.lambda_)
        print('Shear Modulus',self.mu)
        print('Poisson ratio',self.mu)
        print('Youngs Modulus',YoungsM)
        print('Poissons ratio',possonratio)
        
        
        jud_coredir=True
        #directory = 'surface_core'
        directory=self.Para0['Corefunc directory']
        if os.path.exists(directory):
            #directory = 'bp5t_core'
            file_path = os.path.join(directory, 'A1s.npy')
            if not os.path.exists(file_path):
                jud_coredir=False
            file_path = os.path.join(directory, 'A2s.npy')
            if not os.path.exists(file_path):
                jud_coredir=False
            file_path = os.path.join(directory, 'Bs.npy')
            if not os.path.exists(file_path):
                jud_coredir=False

            file_path = os.path.join(directory, 'A1d.npy')
            if not os.path.exists(file_path):
                jud_coredir=False

            file_path = os.path.join(directory, 'A2d.npy')
            if not os.path.exists(file_path):
                jud_coredir=False
            file_path = os.path.join(directory, 'Bd.npy')
            if not os.path.exists(file_path):
                jud_coredir=False
        else:
            os.mkdir(directory)
            jud_coredir=False


        # 检查目录是否存在
        if(jud_coredir==True):
            #print('exit!!!')
            print('Start to load core functions...')
            self.A1s=np.load(directory+'/A1s.npy',allow_pickle=True)
            self.A2s=np.load(directory+'/A2s.npy',allow_pickle=True)
            self.Bs=np.load(directory+'/Bs.npy',allow_pickle=True)
            self.A1d=np.load(directory+'/A1d.npy',allow_pickle=True)
            self.A2d=np.load(directory+'/A2d.npy',allow_pickle=True)
            self.Bd=np.load(directory+'/Bd.npy',allow_pickle=True)
            print('Core functions load completed.')
        else:
            print('Start to calculate core functions...')
            print('Start to calculate core functions...')
            #self.A1s,self.A2s,self.Bs,self.A1d,self.A2d,self.Bd=self.get_coreAB_mulproess()
            #self.A1s,self.A2s,self.Bs,self.A1d,self.A2d,self.Bd=self.get_coreAB()
            self.A1s,self.A2s,self.Bs=self.get_coreAB_mulproess1()
            np.save(directory+'/A1s',self.A1s) #strike slip contribute to traction along direction 1
            np.save(directory+'/A2s',self.A2s)  #strike slip contribute to traction along direction 2
            np.save(directory+'/Bs',self.Bs)  #strike slip contribute to normal traction
            self.A1d,self.A2d,self.Bd=self.get_coreAB_mulproess2()
            np.save(directory+'/A1d',self.A1d) #dip slip contribute to traction along direction 1
            np.save(directory+'/A2d',self.A2d) #dip slip contribute to traction along direction 2
            np.save(directory+'/Bd',self.Bd)   #dip slip contribute to normal traction
            print('Core functions computation completed')
        
        #print(np.any(np.isnan([self.A1s])))
        #self.ouputVTK(param1=self.Tno, param2=self.Tt1o, param3=self.Tt2o)
        
        self.Init_condition()

        if(self.Para0['H-matrix']=='True'):
            #import hmatrix as hm
            global hm
            import hmatrix as hm                
            print('Start to create Hierarchical Matrix structure...')
            gr=hm.transgr(nodelst,elelst,eleVec,xg)
            hm.create_hmatrix_structure(gr)
            hm.create_Hmvalue(self.A1s)
            hm.create_Hmvalue(self.A2s)
            hm.create_Hmvalue(self.Bs)
            hm.create_Hmvalue(self.A1d)
            hm.create_Hmvalue(self.A2d)
            hm.create_Hmvalue(self.Bd)

            print('Hierarchical Matrix structure Constructing completed')

        

        # f=open('Tvalue.txt','w')
        # f.write('xg1,xg2,xg3,se1,se2,se3\n')
        # for i in range(len(xg)):
        #     #f.write('%f %f %f %f %f %f\n' %(xg[i,0],xg[i,1],xg[i,2],self.Tt1o[i],self.Tt2o[i],self.Tno[i]))
        #     f.write('%f,%f,%f,%f,%f,%f\n' %(xg[i,0],xg[i,1],xg[i,2],self.T_globalarr[i,0],self.T_globalarr[i,1],self.T_globalarr[i,2]))
        # f.close()

    
    


    def get_rotation1(self,x):
        if(x<70):
            theta=-10
        elif(x>=60.0 and x<80.0):
            temx=((x-60.0)/10.0-1.0)*np.pi/2
            theta=-10.0-(sin(temx)+1.0)*10.0
        else:
            theta=-30.0
        return theta


    def Init_condition(self):
        N=len(self.eleVec)
        self.Tt1o=np.zeros(N)
        self.Tt2o=np.zeros(N)
        self.Tno=np.zeros(N)
        ssv_scale=float(self.Para0['Vertical principal stress'])
        ssh1_scale=float(self.Para0['Maximum horizontal principal stress'])
        ssv0_scale=float(self.Para0['Minimum horizontal principal stress'])
        trac_nor=float(self.Para0['Vertical principal stress value'])
        for i in range(N):
            if(self.Para0['Vertical principal stress value varies with depth']=='True'):
                turning_dep=float(self.Para0['Turnning depth'])
                ssv= -self.xg[i,2]/turning_dep
                if(ssv>1.0):
                    ssv=ssv*ssv_scale
            #ssv=ssv*1e6
            ssv=trac_nor*ssv
            ssh1=-ssv*ssh1_scale
            ssh2=-ssv*ssv0_scale
            ssv=-ssv
            #ssv= -xg3[i]*maxside/5.;
            #Ph1ang=self.get_rotation1(xg[i,0])-10.0
            #Ph1ang=np.pi/180.*Ph1ang
            Ph1ang=float(self.Para0['Angle between ssh1 and X-axis'])
            Ph1ang=np.pi/180.*Ph1ang
            v11=cos(Ph1ang)
            v12=-sin(Ph1ang)
            v21=sin(Ph1ang)
            v22=cos(Ph1ang)
            Rmatrix=np.array([[v11,v12],[v21,v22]])
            Pstress=np.array([[ssh1,0],[0,ssh2]])
            stress=np.dot(np.dot(Rmatrix,Pstress),Rmatrix.transpose())
            stress3D=np.array([[stress[0][0],stress[0][1],0],[stress[1][0],stress[1][1],0],[0,0,ssv]])
            #Me=self.eleVec[i].reshape([3,3])
            #T_global=np.dot(Me.transpose(),T_local)
            tra=np.dot(stress3D,self.eleVec[i,-3:])
            ev11,ev12,ev13=self.eleVec[i,0],self.eleVec[i,1],self.eleVec[i,2]
            ev21,ev22,ev23=self.eleVec[i,3],self.eleVec[i,4],self.eleVec[i,5]
            ev31,ev32,ev33=self.eleVec[i,6],self.eleVec[i,7],self.eleVec[i,8]
            self.Tt1o[i]=tra[0]*ev11+tra[1]*ev12+tra[2]*ev13
            self.Tt2o[i]=tra[0]*ev21+tra[1]*ev22+tra[2]*ev23
            self.Tno[i]=tra[0]*ev31+tra[1]*ev32+tra[2]*ev33

        
        self.Tno=np.abs(self.Tno)
        self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)
        tem=self.Tt/self.Tno
        x=self.Tt1o/self.Tt
        y=self.Tt2o/self.Tt
        solve_shear=self.Para0['Shear traction solved from stress tensor']=='True'
        if(self.Para0['Rake solved from stress tensor']=='True'):
            self.rake=np.arctan2(y,x)
        else:
            self.rake=np.ones(len(self.Tt))*float(self.Para0['Fix_rake'])
        #self.rake=np.ones(len(x))*35.0/180.0*np.pi
        self.vec_Tra=np.array([x,y]).transpose()
        
        #print(self.Tt1o)
        #print(np.max(tem),np.min(tem))


        T_globalarr=[]
        N=self.Tt1o.shape[0]
        self.Vpl_con=1e-6
        self.Vpl_con=float(self.Para0['Plate loading rate'])
        self.Vpls=np.zeros(N)
        self.Vpld=np.zeros(N)
        
        self.shear_loadingS=np.zeros(N)
        self.shear_loadingD=np.zeros(N)
        self.shear_loading=np.zeros(N)
        self.normal_loading=np.zeros(N)

        self.V0=float(self.Para0['Reference slip rate'])
        # self.dc=np.ones(N)*0.01
        # self.f0=np.ones(N)*0.4
        self.dc=np.ones(N)*0.02
        self.f0=np.ones(N)*float(self.Para0['Reference friction coefficient'])
        self.a=np.zeros(N)
        self.b=np.ones(N)*0.03

        self.slipv1=np.zeros(N)
        self.slipv2=np.zeros(N)
        self.slipv=np.zeros(N)
        self.slip1=np.zeros(N)
        self.slip2=np.zeros(N)
        self.slip=np.zeros(N)

        self.arriT=np.ones(N)*1e9

        
        

        boundary_edges,boundary_nodes=find_boundary_edges_and_nodes(self.elelst)
        boundary_coord=self.nodelst[boundary_nodes-1]
        #print(boundary_coord.shape,boundary_nodes.shape)

        xmin,xmax=np.min(self.xg[:,0]),np.max(self.xg[:,0])
        ymin,ymax=np.min(self.xg[:,1]),np.max(self.xg[:,1])
        zmin,zmax=np.min(self.xg[:,2]),np.max(self.xg[:,2])

        nux=float(self.Para0['Nuclea_posx'])
        nuy=float(self.Para0['Nuclea_posy'])
        nuz=float(self.Para0['Nuclea_posz'])
        nuclearloc=np.array([nux,nuy,nuz])
        #nuclearloc=np.array([-20000,0,-20000])
        Wedge=float(self.Para0['Widths of VS region'])
        self.localTra=np.zeros([N,2])
        fric_VS=float(self.Para0['Shear traction in VS region'])
        fric_VW=float(self.Para0['Shear traction in VW region'])
        fric_nu=float(self.Para0['Shear traction in nucleation region'])
        transregion=float(self.Para0['Transition region ratio from VS to VW region'])
        aVs=float(self.Para0['Rate-and-state parameters a in VS region'])
        bVs=float(self.Para0['Rate-and-state parameters b in VS region'])
        dcVs=float(self.Para0['Characteristic slip distance in VS region'])
        aVw=float(self.Para0['Rate-and-state parameters a in VW region'])
        bVw=float(self.Para0['Rate-and-state parameters b in VW region'])
        dcVw=float(self.Para0['Characteristic slip distance in VW region'])

        aNu=float(self.Para0['Rate-and-state parameters a in nucleation region'])
        bNu=float(self.Para0['Rate-and-state parameters b in nucleation region'])
        dcNu=float(self.Para0['Characteristic slip distance in nucleation region'])
        slivpNu=float(self.Para0['Initial slip rate in nucleation region'])
        Set_nuclear=self.Para0['Set_nucleation']=='True'
        Radiu_nuclear=float(self.Para0['Radius of nucleation'])

        for i in range(self.Tt1o.shape[0]):
            #tem=min(self.xg[i,0]-xmin,xmax-self.xg[i,0],self.xg[i,2]-zmin,zmax-self.xg[i,2],Wedge)/Wedge
            coords1=np.array([self.xg[i]])
            #print(coords1.shape, boundary_coord.shape)
            distem=find_min_euclidean_distance(coords1, boundary_coord)
            distem=distem/Wedge
            
            nuclearregion=1.0-transregion
            if(distem<nuclearregion):
                self.a[i]=aVs
                self.b[i]=bVs
                if(solve_shear==False):
                    self.Tt[i]=self.Tno[i]*fric_VS
                self.slipv[i]=self.Vpl_con
                self.dc[i]=bVs

            elif(distem<=1.0):
                self.a[i]=aVs-(aVs-aVw)*(distem-nuclearregion)/transregion
                if(solve_shear==False):
                    self.Tt[i]=self.Tno[i]*(fric_VS+(fric_VW-fric_VS)*(distem-nuclearregion)/transregion)
                self.slipv[i]=self.Vpl_con
                self.dc[i]=dcVs
            else:
                self.a[i]=aVw
                self.b[i]=bVw
                if(solve_shear==False):
                    self.Tt[i]=self.Tno[i]*fric_VW
                self.dc[i]=dcVw
                self.slipv[i]=self.Vpl_con
                # self.Tt1o[i]=self.Tt1o[i]*2
                # self.Tt2o[i]=self.Tt2o[i]*2
            
            distem=np.linalg.norm(self.xg[i]-nuclearloc)

            if(distem<Radiu_nuclear and Set_nuclear==True):
                self.slipv[i]=slivpNu
                #self.slipv[i]=self.Vpl_con
                self.dc[i]=dcNu
                self.a[i]=aNu
                self.b[i]=bNu
                if(solve_shear==False):
                    self.Tt[i]=self.Tno[i]*fric_nu

    
            T_local=np.zeros(3)
            T_local[0]=cos(self.rake[i])
            T_local[1]=sin(self.rake[i])
            Me=self.eleVec[i].reshape([3,3])
            T_global=np.dot(Me.transpose(),T_local)
            #print(self.Tt1o[i],self.Tt2o[i],T_global)
            T_globalarr.append(T_global)  
        self.T_globalarr=np.array(T_globalarr)

        self.Tt1o=self.Tt*np.cos(self.rake)
        self.Tt2o=self.Tt*np.sin(self.rake)
        self.slipv1=self.slipv*np.cos(self.rake)
        self.slipv2=self.slipv*np.sin(self.rake)
        self.vec_Tra=np.array([np.cos(self.rake),np.sin(self.rake)]).transpose()
        
        # x=self.Tt1o/self.Tt
        # y=self.Tt2o/self.Tt
        # self.vec_Tra=np.array([x,y]).transpose()
        #print(self.vec_Tra.shape)

        self.fric=self.Tt/self.Tno
        self.state=np.log(np.sinh(self.Tt/self.Tno/self.a)*2.0*self.V0/self.slipv)*self.a

        if(self.InputHetoparamter==True):
            self.read_parameter(self.Para0['Inputparamter file'])
        #print(min(self.xg[i,0]-xmin,xmax-self.xg[i,0],self.xg[i,2]-zmin,zmax-self.xg[i,2],Wedge)/Wedge)


    
    def read_parameter(self,fname):
        f=open(self.dirname+'/'+fname,'r')
        values=[]
        for line in f:
            tem=line.split()
            tem=np.array(tem).astype(float)
            values.append(tem)
        f.close()

        values=np.array(values)
        Ncell=self.eleVec.shape[0]
        self.rake=values[:Ncell,0]
        self.a=values[:Ncell,1]
        self.b=values[:Ncell,2]
        self.dc=values[:Ncell,3]
        self.f0=values[:Ncell,4]
        #self.Tt1o=values[:Ncell,5]*1e6
        #self.Tt2o=values[:Ncell,5]*0
        self.Tt=values[:Ncell,5]*1e6
        self.Tno=values[:Ncell,6]*1e6
        #self.slipv1=values[:Ncell,7]
        #self.slipv2=-values[:Ncell,7]*0.0
        self.slipv=values[:Ncell,7]
        
        self.shear_loading=values[:Ncell,8]
        self.normal_loading=values[:Ncell,9]

        #self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)

        # x=self.Tt1o/self.Tt
        # y=self.Tt2o/self.Tt
        # self.rake=np.arctan2(y,x)
        #print(self.rake)
        #self.vec_Tra=np.array([x,y]).transpose()
        #print(self.vec_Tra.shape)
        #self.slipv=np.sqrt(self.slipv1*self.slipv1+self.slipv2*self.slipv2)

        
        x=np.cos(self.rake)
        y=np.sin(self.rake)
        self.Tt1o=self.Tt*x
        self.Tt2o=self.Tt*y
        self.slipv1=self.slipv*x
        self.slipv2=self.slipv*y
        self.vec_Tra=np.array([x,y]).transpose()



        self.fric=self.Tt/self.Tno
        self.state=np.log(np.sinh(self.Tt/self.Tno/self.a)*2.0*self.V0/self.slipv)*self.a

        T_globalarr=[]
        for i in range(len(self.rake)):
            T_local=np.zeros(3)
            T_local[0]=cos(self.rake[i])
            T_local[1]=sin(self.rake[i])
            Me=self.eleVec[i].reshape([3,3])
            T_global=np.dot(Me.transpose(),T_local)
            #print(self.Tt1o[i],self.Tt2o[i],T_global)
            T_globalarr.append(T_global)  
        self.T_globalarr=np.array(T_globalarr)


    
    def derivative(self,Tno,Tt,state1):
        #try:
        #dVdtau1=2*self.V0/(self.a*Tno)*np.exp(-state1/self.a)*np.cosh(Tt1o/(self.a*Tno))
        dVdtau=2*self.V0/(self.a*Tno)*np.exp(-state1/self.a)*np.cosh(Tt/(self.a*Tno))
        #print('dVdtao',np.mean(dVdtao))
        dVdsigma=-2*self.V0*Tt/(self.a*Tno*Tno)*np.exp(-state1/self.a)*np.cosh(Tt/(self.a*Tno))
        #dVdsigma2=-2*self.V0*Tt2o/(self.a*Tno*Tno)*np.exp(-state1/self.a)*np.cosh(Tt2o/(self.a*Tno))

        dVdstate=-2*self.V0/self.a*np.exp(-state1/self.a)*np.sinh(Tt/(self.a*Tno))
        #dVdstate=-self.slipv/self.a
        dstatedt=self.b/self.dc*(self.V0*np.exp((self.f0-state1)/self.b)-np.abs(self.slipv))


        
        #fss=self.f0+(self.a-self.b)*np.log(np.abs(self.slipv)/self.V0)
        #dstatedt=-np.abs(self.slipv)/self.dc*(np.abs(self.Tt)/self.Tno-fss)
        # except FloatingPointError as e:
        #     print("overflow encountered:", e)
        #     return False

        # slipv1=(self.slipv)*self.localTra[:,0]
        # slipv2=(self.slipv)*self.localTra[:,1]
        
 
        # slipv1=-self.slipv1+self.Vpl_con*np.cos(self.rake)
        # slipv2=-self.slipv2+self.Vpl_con*np.sin(self.rake)

        slipv=self.slipv-self.Vpl_con

        slipv1=slipv*np.cos(self.rake)
        slipv2=slipv*np.sin(self.rake)
        #slipv1=-slipv1
        #slipv2=-slipv2
        #print(np.mean(self.localTra[:,0]),np.mean(self.localTra[:,1]))
        if(self.Para0['H-matrix']=='False'):
            dsigmadt=np.dot(self.Bs,slipv1)+np.dot(self.Bd,slipv2)+self.normal_loading
            
            AdotV1=np.dot(self.A1s,slipv1)+np.dot(self.A1d,slipv2)
            AdotV2=np.dot(self.A2s,slipv1)+np.dot(self.A2d,slipv2)
            
        elif(self.Para0['H-matrix']=='True'):
            
            dsigmadt=hm.Hmatrix_dot_X(2,slipv1)+hm.Hmatrix_dot_X(5,slipv2)+self.normal_loading
            
            AdotV1=hm.Hmatrix_dot_X(0,slipv1)+hm.Hmatrix_dot_X(3,slipv2)
            
            AdotV2=hm.Hmatrix_dot_X(1,slipv1)+hm.Hmatrix_dot_X(4,slipv2)
            #print(np.min(AdotV1),np.min(AdotV2))
            

        AdotV=np.array([AdotV1,AdotV2]).transpose()
        AdotV=-np.sum(AdotV * self.vec_Tra, axis=1)
        #print('slipv:',np.max(slipv1),np.max(slipv2))
        #print('AdotV:',np.max(AdotV1),np.max(AdotV2),np.max(slipv1),np.max(slipv2))
        # AdotV=np.sqrt(AdotV1*AdotV1+AdotV2*AdotV2)


        #AdotV=np.dot(self.A1s,self.slipv)
        #print(np.min(AdotV),np.max(AdotV),np.max(np.abs(AdotV1)),np.max(np.abs(AdotV2)))


        dtau1dt=(AdotV+self.shear_loading-self.mu/(2.0*self.Cs)*(dVdsigma*dsigmadt+dVdstate*dstatedt))/(1.0+self.mu/(2.0*self.Cs)*dVdtau)
        #dtau2dt=(AdotV2+self.shear_loading-self.mu/(2.0*self.Cs)*np.sin(self.rake)*(dVdsigma*dsigmadt+dVdstate*dstatedt))/(1.0+self.mu/(2.0*self.Cs)*dVdtau)
        
        #dVdt=dVdtao*dtaodt+dVdsigma*dsigmadt+dVdstate*dstatedt
        #print(np.max(dVdt),np.min(dVdt))
        #print('dtaodt',np.mean(AdotV),np.mean(dtaodt))
        #print('dstatedt',dstatedt[2747],dtaodt[2747])
        
        
        return dstatedt,dsigmadt,dtau1dt
    
    def simu_forward(self,dttry):
        nrjct=0
        kp=0.08
        
        self.AbsTol1=1e-6
        self.AbsTol2=800
        self.RelTol1=1e-4
        self.RelTol2=1e-4

        h=dttry
        running=True
        while running:
            Tno_yhk,Tt_yhk,state_yhk,condition1,condition2,hnew1,hnew2=self.RungeKutte_solve_Dormand_Prince(h)
            #print('condition1,condition2,hnew1,hnew2:',condition1,condition2,hnew1,hnew2)

            if(max(condition1,condition2)<1.0 and not (np.isnan(condition1) or np.isnan(condition2))):
                
                dtnext=np.min([hnew1,hnew2])
                #print('received condition1,condition2:',condition1,condition2,'  dt:',h)
                break
            else:
                nrjct=nrjct+1
                h=h*0.5
                print('nrjct:',nrjct,'  condition1,',condition1,' condition2:',condition2,'  dt:',h)

                if(h<1.e-15 or nrjct>20):
                    print('error: dt is too small')
                    sys.exit()
        


        #     if(errormax<1.0):
        #         print('received errormax:',errormax,'  dt:',h)
        #         break
            
        #     nrjct=nrjct+1
        #     if(np.isnan(errormax)):
        #         print('overflow encountered')
        #         h=0.1*h
        #     else:

        #         h=max(0.5*h,SAFETY*h*(errormax**PSHRNK))

        #     print('nrjct:',nrjct,'  errormax:',errormax,'  dt:',h)
        #     if(h<1.e-15 or nrjct>20):
        #         print('error: dt is too small')
        #         sys.exit()
        
        
        # hnext=min(2*h,SAFETY*h*(errormax**PGROW))
        self.time=self.time+h
        self.Tno=Tno_yhk
        self.Tt=Tt_yhk
        #print('self.Tt2o',np.max(self.Tt2o),np.min(self.Tt2o))
        #self.Tt=np.sqrt(self.Tt1o*self.Tt1o+self.Tt2o*self.Tt2o)

        self.state=state_yhk

        self.Tt1o=self.Tt*np.cos(self.rake)
        self.Tt2o=self.Tt*np.sin(self.rake)
        #self.rake=np.arctan2(y,x)


        self.slipv=(2.0*self.V0)*np.exp(-self.state/self.a)*np.sinh(self.Tt/self.Tno/self.a)
        self.slipv1=self.slipv*np.cos(self.rake)
        self.slipv2=self.slipv*np.sin(self.rake)
        #print('self.slipv:',np.min(self.slipv1),np.max(self.slipv2))

        self.slip1=self.slip1+self.slipv1*h
        self.slip2=self.slip2+self.slipv2*h
        self.slip=np.sqrt(self.slip1*self.slip1+self.slip2*self.slip2)
        
        self.fric=self.Tt/self.Tno

        return h,dtnext

    def RungeKutte_solve_Dormand_Prince(self,h):

        B21=.2
        B31=3./40
        B32=9./40.

        B41=44./45.
        B42=-56./15
        B43=32./9

        B51=19372./6561.
        B52=-25360/2187.
        B53=64448./6561.
        B54=-212./729.

        B61=9170./3168.
        B62=-355./33.
        B63=-46732./5247.
        B64=49./176.
        B65=-5103./18656.

        B71=35./384.
        B73=500./1113.
        B74=125./192.
        B75=-2187./6784.
        B76=11./84.

        B81=5179./57600.
        B83=7571./16695.
        B84=393./640.
        B85=-92097./339200.
        B86=187./2100.
        B87=1./40.



        #dstatedt1,dsigmadt1,dtau1dt1,dtau2dt1=self.derivative(self.Tno,np.abs(self.Tt1o),np.abs(self.Tt2o),self.state)
        dstatedt1,dsigmadt1,dtau1dt1=self.derivative(self.Tno,self.Tt,self.state)
        Tno=self.Tno
        Tt1o=self.Tt
        Tno_yhk=Tno+h*B21*dsigmadt1
        Tt1o_yhk=Tt1o+h*B21*dtau1dt1
        #Tt2o_yhk=Tt2o+h*B21*dtau2dt1
        state_yhk=self.state+h*B21*dstatedt1
        #print('Tt_yhk',np.mean(Tt_yhk))
        

        dstatedt2,dsigmadt2,dtau1dt2=self.derivative(Tno_yhk,Tt1o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B31*dsigmadt1+B32*dsigmadt2)
        Tt1o_yhk=Tt1o+h*(B31*dtau1dt1+B32*dtau1dt2)
        #Tt2o_yhk=Tt2o+h*(B31*dtau2dt1+B32*dtau2dt2)
        state_yhk=self.state+h*(B31*dstatedt1+B32*dstatedt2)
        #print('Tt_yhk',np.mean(Tt_yhk))

        dstatedt3,dsigmadt3,dtau1dt3=self.derivative(Tno_yhk,Tt1o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B41*dsigmadt1+B42*dsigmadt2+B43*dsigmadt3)
        Tt1o_yhk=Tt1o+h*(B41*dtau1dt1+B42*dtau1dt2+B43*dtau1dt3)
        #Tt2o_yhk=Tt2o+h*(B41*dtau2dt1+B42*dtau2dt2+B43*dtau2dt3)
        state_yhk=self.state+h*(B41*dstatedt1+B42*dstatedt2+B43*dstatedt3)
        #print('Tt_yhk',np.mean(Tt_yhk))

        dstatedt4,dsigmadt4,dtau1dt4=self.derivative(Tno_yhk,Tt1o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B51*dsigmadt1+B52*dsigmadt2+B53*dsigmadt3+B54*dsigmadt4)
        Tt1o_yhk=Tt1o+h*(B51*dtau1dt1+B52*dtau1dt2+B53*dtau1dt3+B54*dtau1dt4)
        #Tt2o_yhk=Tt2o+h*(B51*dtau2dt1+B52*dtau2dt2+B53*dtau2dt3+B54*dtau2dt4)
        state_yhk=self.state+h*(B51*dstatedt1+B52*dstatedt2+B53*dstatedt3+B54*dstatedt4)

        dstatedt5,dsigmadt5,dtau1dt5=self.derivative(Tno_yhk,Tt1o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B61*dsigmadt1+B62*dsigmadt2+B63*dsigmadt3+B64*dsigmadt4+B65*dsigmadt5)
        Tt1o_yhk=Tt1o+h*(B61*dtau1dt1+B62*dtau1dt2+B63*dtau1dt3+B64*dtau1dt4+B65*dtau1dt5)
        #Tt2o_yhk=Tt2o+h*(B61*dtau2dt1+B62*dtau2dt2+B63*dtau2dt3+B64*dtau2dt4+B65*dtau2dt5)
        state_yhk=self.state+h*(B61*dstatedt1+B62*dstatedt2+B63*dstatedt3+B64*dstatedt4+B65*dstatedt5)
        

        dstatedt6,dsigmadt6,dtau1dt6=self.derivative(Tno_yhk,Tt1o_yhk,state_yhk)
        Tno_yhk=Tno+h*(B71*dsigmadt1+B73*dsigmadt3+B74*dsigmadt4+B75*dsigmadt5+B76*dsigmadt6)
        Tt1o_yhk=Tt1o+h*(B71*dtau1dt1+B73*dtau1dt3+B74*dtau1dt4+B75*dtau1dt5+B76*dtau1dt6)
        #Tt2o_yhk=Tt2o+h*(B71*dtau2dt1+B73*dtau2dt3+B74*dtau2dt4+B75*dtau2dt5+B76*dtau2dt6)
        state_yhk=self.state+h*(B71*dstatedt1+B73*dstatedt3+B74*dstatedt4+B75*dstatedt5+B76*dstatedt6)

        dstatedt7,dsigmadt7,dtau1dt7=self.derivative(Tno_yhk,Tt1o_yhk,state_yhk)
        Tno_yhk8=Tno+h*(B81*dsigmadt1+B83*dsigmadt3+B84*dsigmadt4+B85*dsigmadt5+B86*dsigmadt6+B87*dsigmadt7)
        Tt1o_yhk8=Tt1o+h*(B81*dtau1dt1+B83*dtau1dt3+B84*dtau1dt4+B85*dtau1dt5+B86*dtau1dt6+B87*dtau1dt7)
        #Tt2o_yhk8=Tt2o+h*(B81*dtau2dt1+B83*dtau2dt3+B84*dtau2dt4+B85*dtau2dt5+B86*dtau2dt6+B87*dtau2dt7)
        state_yhk8=self.state+h*(B81*dstatedt1+B83*dstatedt3+B84*dstatedt4+B85*dstatedt5+B86*dstatedt6+B87*dstatedt7)

        state_yhk_err=np.abs(state_yhk8-state_yhk)
        Tno_yhk_err=np.abs(Tno_yhk8-Tno_yhk)
        Tt1o_yhk_err=np.abs(Tt1o_yhk8-Tt1o_yhk)
        #Tt2o_yhk_err=np.abs(Tt2o_yhk8-Tt2o_yhk)

        errormax1=np.max(state_yhk_err)+1e-20
        errormax2=np.max([np.max(Tno_yhk_err),np.max(Tt1o_yhk_err)])

        Relerrormax1=np.max(np.abs(state_yhk_err/state_yhk8))
        Relerrormax2=np.max([np.max(np.abs(Tno_yhk_err/Tno_yhk8)),np.max(np.abs(Tt1o_yhk_err/Tt1o_yhk8))])
        print('errormax1,errormax2,relaemax1,relaemax2:',errormax1,errormax2,Relerrormax1,Relerrormax2)

        maxiY1=np.max(np.abs(state_yhk))
        maxiYn1=np.max(np.abs(state_yhk8))

        maxiY2=np.max([np.max(Tno_yhk),np.max(Tt1o_yhk)])
        maxiYn2=np.max([np.max(Tno_yhk8),np.max(Tt1o_yhk8)])

        condition1=errormax1/max([self.AbsTol1,self.RelTol1*max(maxiY1,maxiYn1)])
        condition2=errormax2/max([self.AbsTol2,self.RelTol2*max(maxiY2,maxiYn2)])

        tol1=max(self.AbsTol1,self.RelTol1*maxiYn1)
        tol2=max(self.AbsTol2,self.RelTol2*maxiYn2)
        
        hnew1=h*min(0.9*(tol1/errormax1)**0.2,5)
        hnew2=h*min(0.9*(tol2/errormax2)**0.2,5)

        return Tno_yhk8,Tt1o_yhk8,state_yhk8,condition1,condition2,hnew1,hnew2



        
    def GetTtstress(self,Stress):
        Tra=[]
        #print(self.eleVec.shape)
        for i in range(self.eleVec.shape[0]):
            ev11,ev12,ev13=self.eleVec[i,0],self.eleVec[i,1],self.eleVec[i,2]
            ev21,ev22,ev23=self.eleVec[i,3],self.eleVec[i,4],self.eleVec[i,5]
            ev31,ev32,ev33=self.eleVec[i,6],self.eleVec[i,7],self.eleVec[i,8]

            Tr1=Stress[0,i]*ev31+Stress[3,i]*ev32+Stress[4,i]*ev33
            Tr2=Stress[3,i]*ev31+Stress[1,i]*ev32+Stress[5,i]*ev33
            Trn=Stress[4,i]*ev31+Stress[5,i]*ev32+Stress[2,i]*ev33

            Tt1=Tr1*ev11+Tr2*ev12+Trn*ev13
            Tt2=Tr1*ev21+Tr2*ev22+Trn*ev23
            Tn=Tr1*ev31+Tr2*ev32+Trn*ev33
            Tra.append([Tt1,Tt2,Tn])
        Tra=np.array(Tra)
        return Tra

    def worker1(self,batch):
        X=self.xg[:,0]
        Y=self.xg[:,1]
        Z=self.xg[:,2]
        result1 = []
        result2 = []
        result3 = []
        
        Ts,Ss,Ds=0,1,0
        for i in batch:
            P1=np.copy(self.nodelst[self.elelst[i,0]-1])
            P2=np.copy(self.nodelst[self.elelst[i,1]-1])
            P3=np.copy(self.nodelst[self.elelst[i,2]-1])

            Stress,Strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu,self.lambda_)
            Tra=self.GetTtstress(Stress)   #第i个源在所有单元上产生的牵引力
            result1.append(Tra[:,0])
            result2.append(Tra[:,1])
            result3.append(Tra[:,2])
        return result1,result2,result3


    def worker2(self,batch):
        X=self.xg[:,0]
        Y=self.xg[:,1]
        Z=self.xg[:,2]
        result4 = []
        result5 = []
        result6 = []
        Ts,Ss,Ds=0,0,1
        for i in batch:
            P1=np.copy(self.nodelst[self.elelst[i,0]-1])
            P2=np.copy(self.nodelst[self.elelst[i,1]-1])
            P3=np.copy(self.nodelst[self.elelst[i,2]-1])

            Stress,Strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu,self.lambda_)
            Tra=self.GetTtstress(Stress)   #第i个源在所有单元上产生的牵引力
            result4.append(Tra[:,0])
            result5.append(Tra[:,1])
            result6.append(Tra[:,2])
        return result4,result5,result6


    def get_coreAB_mulproess1(self):
        
        N=self.xg.shape[0]
        A1s=np.zeros([N,N])
        A2s=np.zeros([N,N])
        Bs=np.zeros([N,N])
        nums = list(range(len(self.elelst)))

        batch_size = self.Batch_size  # 每个批处理1000个任务
        batches = [nums[i:i + batch_size] for i in range(0, len(nums), batch_size)]
        print(len(batches),self.elelst.shape)

        executor = ProcessPoolExecutor(max_workers=self.num_process)
        try:
            results_generator = executor.map(self.worker1, batches)
            result1 = []
            result2 = []
            result3 = []
            K=0
            for result in results_generator:
                res1, res2, res3 = result
                result1.append(res1)
                result2.append(res2)
                result3.append(res3)
                print(f"Batch completed with worker1 results:{K} of {len(batches)}")
                K=K+1

        finally:
            executor.shutdown(wait=True)  # 确保资源在异常情况下也能正常关闭
        
        result1_=[]
        result2_=[]
        result3_=[]
        for i in range(len(result1)):
            result1_.append(np.array(result1[i]))
            result2_.append(np.array(result2[i]))
            result3_.append(np.array(result3[i]))
        result1_=np.concatenate(result1_,axis=0)
        result2_=np.concatenate(result2_,axis=0)
        result3_=np.concatenate(result3_,axis=0)
        A1s=result1_.transpose()
        A2s=result2_.transpose()
        Bs=result3_.transpose()

        return A1s,A2s,Bs
    
    

    def get_coreAB_mulproess2(self):
        N=self.xg.shape[0]
        A1d=np.zeros([N,N])
        A2d=np.zeros([N,N])
        Bd=np.zeros([N,N])
        nums = list(range(len(self.elelst)))

        batch_size = self.Batch_size  # 每个批处理1000个任务
        batches = [nums[i:i + batch_size] for i in range(0, len(nums), batch_size)]
        print(len(batches),self.elelst.shape)

        executor = ProcessPoolExecutor(max_workers=self.num_process)
        try:
            results_generator = executor.map(self.worker2, batches)
            result4 = []
            result5 = []
            result6 = []
            K=0
            for result in results_generator:
                res4,res5,res6 = result
                result4.append(res4)
                result5.append(res5)
                result6.append(res6)
                print(f"Batch completed with worker2 results:{K} of {len(batches)}")
                K=K+1

        finally:
            executor.shutdown(wait=True)  # 确保资源在异常情况下也能正常关闭
        
        result4_=[]
        result5_=[]
        result6_=[]
        for i in range(len(result4)):
            result4_.append(np.array(result4[i]))
            result5_.append(np.array(result5[i]))
            result6_.append(np.array(result6[i]))
        result4_=np.concatenate(result4_,axis=0)
        result5_=np.concatenate(result5_,axis=0)
        result6_=np.concatenate(result6_,axis=0)
        A1d=result4_.transpose()
        A2d=result5_.transpose()
        Bd=result6_.transpose()

        return A1d,A2d,Bd

    def worker(self,batch):
        X=self.xg[:,0]
        Y=self.xg[:,1]
        Z=self.xg[:,2]
        result1 = []
        result2 = []
        result3 = []
        result4 = []
        result5 = []
        result6 = []
        Ts,Ss,Ds=0,1,0
        for i in batch:
            P1=np.copy(self.nodelst[self.elelst[i,0]-1])
            P2=np.copy(self.nodelst[self.elelst[i,1]-1])
            P3=np.copy(self.nodelst[self.elelst[i,2]-1])
            if(self.halfspace_jud==True):
                Stress,Strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu,self.lambda_)
            else:
                Stress,Strain=DH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu,self.lambda_)
            Tra=self.GetTtstress(Stress)   #第i个源在所有单元上产生的牵引力
            result1.append(Tra[:,0])
            result2.append(Tra[:,1])
            result3.append(Tra[:,2])
        
        Ts,Ss,Ds=0,0,1
        for i in batch:
            P1=np.copy(self.nodelst[self.elelst[i,0]-1])
            P2=np.copy(self.nodelst[self.elelst[i,1]-1])
            P3=np.copy(self.nodelst[self.elelst[i,2]-1])
            if(self.halfspace_jud==True):
                Stress,Strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu,self.lambda_)
            else:
                Stress,Strain=DH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu,self.lambda_)
            Tra=self.GetTtstress(Stress)   #第i个源在所有单元上产生的牵引力
            result4.append(Tra[:,0])
            result5.append(Tra[:,1])
            result6.append(Tra[:,2])

        return result1,result2,result3,result4,result5,result6
    
    def get_coreAB_mulproess(self):
        
        N=self.xg.shape[0]
        A1s=np.zeros([N,N])
        A2s=np.zeros([N,N])
        Bs=np.zeros([N,N])
        A1d=np.zeros([N,N])
        A2d=np.zeros([N,N])
        Bd=np.zeros([N,N])
        nums = list(range(len(self.elelst)))

        batch_size = self.Batch_size  # 每个批处理1000个任务
        batches = [nums[i:i + batch_size] for i in range(0, len(nums), batch_size)]
        print(len(batches),self.elelst.shape)

        executor = ProcessPoolExecutor(max_workers=self.num_process)
        try:
            #results_generator = executor.map(self.worker, batches)
            #result1 = list(results_generator)
            results_generator = executor.map(self.worker, batches)
            result1 = []
            result2 = []
            result3 = []
            result4 = []
            result5 = []
            result6 = []
            for result in results_generator:
                res1, res2, res3, res4,res5,res6 = result
                result1.append(res1)
                result2.append(res2)
                result3.append(res3)
                result4.append(res4)
                result5.append(res5)
                result6.append(res6)
        finally:
            executor.shutdown(wait=True)  # 显式调用shutdown

        
        result1_=[]
        result2_=[]
        result3_=[]
        result4_=[]
        result5_=[]
        result6_=[]
        for i in range(len(result1)):
            result1_.append(np.array(result1[i]))
            result2_.append(np.array(result2[i]))
            result3_.append(np.array(result3[i]))
            result4_.append(np.array(result4[i]))
            result5_.append(np.array(result5[i]))
            result6_.append(np.array(result6[i]))
        result1_=np.concatenate(result1_,axis=0)
        result2_=np.concatenate(result2_,axis=0)
        result3_=np.concatenate(result3_,axis=0)
        result4_=np.concatenate(result4_,axis=0)
        result5_=np.concatenate(result5_,axis=0)
        result6_=np.concatenate(result6_,axis=0)
        A1s=result1_.transpose()
        A2s=result2_.transpose()
        Bs=result3_.transpose()
        A1d=result4_.transpose()
        A2d=result5_.transpose()
        Bd=result6_.transpose()

        

        # with ProcessPoolExecutor(max_workers=50) as executor:
        #     futures = [executor.submit(self.worker, batch) for batch in batches]
            
        #     results = []
        #     for future in as_completed(futures):
        #         results.extend(future.result())
        # result1 = [item for sublist in results for item in sublist]


        
        # dictCut={}
        # dictCut['data']=result1
        # with open('core.json', 'w') as f:
        #     json.dump(dictCut, f)
        # print(len(result1),np.array(result1[0]).shape)
        # result1_=[]
        # f=open('coretest.txt','w')
        # for i in range(len(result1)):
        #     result1_.append(np.array(result1[i]))
        #     for j in range(len(result1[i])):
        #         for k in range(len(result1[i][j])):
        #             f.write('%f\n' %result1[i][j][k])
        # f.close()
        # result1_=np.concatenate(result1_,axis=0)
        # np.save('result1_',result1_)
        return A1s,A2s,Bs,A1d,A2d,Bd
        

        


    def get_coreAB(self):

        X=self.xg[:,0]
        Y=self.xg[:,1]
        Z=self.xg[:,2]
        N=self.xg.shape[0]
        A1s=np.zeros([N,N])
        A2s=np.zeros([N,N])
        Bs=np.zeros([N,N])
        A1d=np.zeros([N,N])
        A2d=np.zeros([N,N])
        Bd=np.zeros([N,N])
        Ts,Ss,Ds=0,1,0
        for i in range(len(self.elelst)):
            P1=np.copy(self.nodelst[self.elelst[i,0]-1])
            P2=np.copy(self.nodelst[self.elelst[i,1]-1])
            P3=np.copy(self.nodelst[self.elelst[i,2]-1])

            Stress,Strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu,self.lambda_)
            Tra=self.GetTtstress(Stress)   #第i个源在所有单元上产生的牵引力
            A1s[:,i]=Tra[:,0]
            A2s[:,i]=Tra[:,1]
            Bs[:,i]=Tra[:,2]
        
        Ts,Ss,Ds=0,0,1
        for i in range(len(self.elelst)):
            #print(i,Ds,P1,P2,P3)
            P1=np.copy(self.nodelst[self.elelst[i,0]-1])
            P2=np.copy(self.nodelst[self.elelst[i,1]-1])
            P3=np.copy(self.nodelst[self.elelst[i,2]-1])
            Stress,Strain=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu,self.lambda_)
            Tra=self.GetTtstress(Stress)   #第i个源在所有单元上产生的牵引力
            A1d[:,i]=Tra[:,0]
            A2d[:,i]=Tra[:,1]
            Bd[:,i]=Tra[:,2]
        return A1s,A2s,Bs,A1d,A2d,Bd
      
 
    

    #def ouputVTK(self,**kwargs):
    def ouputVTK(self,fname):
        Nnode=self.nodelst.shape[0]
        Nele=self.elelst.shape[0]
        f=open(fname,'w')
        f.write('# vtk DataFile Version 3.0\n')
        f.write('test\n')
        f.write('ASCII\n')
        f.write('DATASET  UNSTRUCTURED_GRID\n')
        f.write('POINTS '+str(Nnode)+' float\n')
        for i in range(Nnode):
            f.write('%f %f %f\n'%(self.nodelst[i][0],self.nodelst[i][1],self.nodelst[i][2]))
        f.write('CELLS '+str(Nele)+' '+str(Nele*4)+'\n')
        for i in range(Nele):
            f.write('3 %d %d %d\n'%(self.elelst[i][0]-1,self.elelst[i][1]-1,self.elelst[i][2]-1))
        f.write('CELL_TYPES '+str(Nele)+'\n')
        for i in range(Nele):
            f.write('5 ')
        f.write('\n')
        

        f.write('CELL_DATA %d ' %(Nele))
        f.write('SCALARS Normal_[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tno)):
            f.write('%f '%(self.Tno[i]))
        f.write('\n')

        # f.write('SCALARS Shear1_[MPa] float\nLOOKUP_TABLE default\n')
        # for i in range(len(self.Tt1o)):
        #     f.write('%f '%(self.Tt1o[i]))
        # f.write('\n')

        # f.write('SCALARS Shear2_[MPa] float\nLOOKUP_TABLE default\n')
        # for i in range(len(self.Tt2o)):
        #     f.write('%f '%(self.Tt2o[i]))
        # f.write('\n')

        f.write('SCALARS Shear_[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tt)):
            f.write('%f '%(self.Tt[i]))
        f.write('\n')

        f.write('SCALARS Shear_1[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tt)):
            f.write('%f '%(self.Tt1o[i]))
        f.write('\n')

        f.write('SCALARS Shear_2[MPa] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.Tt)):
            f.write('%f '%(self.Tt2o[i]))
        f.write('\n')

        f.write('SCALARS rake[Degree] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.rake)):
            f.write('%f '%(self.rake[i]*180./np.pi))
        f.write('\n')


        # f.write('SCALARS Slipv1_[m/s] float\nLOOKUP_TABLE default\n')
        # for i in range(len(self.slipv1)):
        #     f.write('%f '%(self.slipv1[i]))
        # f.write('\n')

        # f.write('SCALARS Slipv2_[m/s] float\nLOOKUP_TABLE default\n')
        # for i in range(len(self.slipv2)):
        #     f.write('%f '%(self.slipv2[i]))
        # f.write('\n')


        f.write('SCALARS state float\nLOOKUP_TABLE default\n')
        for i in range(len(self.state)):
            f.write('%f '%(self.state[i]))
        f.write('\n')


        f.write('SCALARS Slipv[m/s] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipv)):
            f.write('%f '%(self.slipv[i]))
        f.write('\n')

        f.write('SCALARS Slipv1[m/s] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipv)):
            f.write('%f '%(self.slipv1[i]))
        f.write('\n')

        f.write('SCALARS Slipv2[m/s] float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slipv)):
            f.write('%f '%(self.slipv2[i]))
        f.write('\n')

        f.write('SCALARS a float\nLOOKUP_TABLE default\n')
        for i in range(len(self.a)):
            f.write('%f '%(self.a[i]))
        f.write('\n')

        f.write('SCALARS b float\nLOOKUP_TABLE default\n')
        for i in range(len(self.b)):
            f.write('%f '%(self.b[i]))
        f.write('\n')

        f.write('SCALARS dc float\nLOOKUP_TABLE default\n')
        for i in range(len(self.dc)):
            f.write('%f '%(self.dc[i]))
        f.write('\n')

        f.write('SCALARS fric float\nLOOKUP_TABLE default\n')
        for i in range(len(self.fric)):
            f.write('%f '%(self.fric[i]))
        f.write('\n')


        f.write('SCALARS slip float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slip)):
            f.write('%f '%(self.slip[i]))
        f.write('\n')

        f.write('SCALARS slip1 float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slip)):
            f.write('%f '%(self.slip1[i]))
        f.write('\n')

        f.write('SCALARS slip2 float\nLOOKUP_TABLE default\n')
        for i in range(len(self.slip)):
            f.write('%f '%(self.slip2[i]))
        f.write('\n')


        # Val_arr=[]
        # for key, value in kwargs.items():
        #     Val_arr.append(value)
        # f.write('CELL_DATA '+str(Nele)+'\n')
        # f.write('SCALARS stress double '+str(len(kwargs))+'\n')
        # f.write('LOOKUP_TABLE default\n')
        # for i in range(Nele):
        #     for j in range(len(Val_arr)):
        #         f.write('%f ' %(Val_arr[j][i]))
        #     f.write('\n')
        # f.write('\n')
        f.close()

    def calc_nucleaszie_cohesivezone(self):
        maxsize=0
        elesize=[]
        for i in range(len(self.eleVec)):
            P1=np.copy(self.nodelst[self.elelst[i,0]-1])
            P2=np.copy(self.nodelst[self.elelst[i,1]-1])
            P3=np.copy(self.nodelst[self.elelst[i,2]-1])
            sizeA=np.linalg.norm(P1-P2)
            sizeB=np.linalg.norm(P1-P3)
            sizeC=np.linalg.norm(P2-P3)
            size0=np.max([sizeA,sizeB,sizeC])
            if(size0>maxsize):
                maxsize=size0
            elesize.append(size0)
        elesize=np.array(elesize)
        self.maxsize=maxsize
        self.ave_elesize=np.mean(elesize)
        b=np.max(self.b)
        a=np.min(self.a)
        sigma=np.mean(self.Tno)
        L=np.max(self.dc)
        self.hRA=2.0/np.pi*self.mu*b*L/(b-a)/(b-a)/sigma
        self.hRA=self.hRA*np.pi*np.pi/4.0
        self.A0=9.0*np.pi/32*self.mu*L/(b*sigma)
        print('maximum element size',maxsize)
        print('average elesize',self.ave_elesize)
        print('Critical nucleation size',self.hRA)
        print('Cohesive zone:',self.A0)
        

    
    def outputtxt(self,fname):
        directory='out_txt'
        if not os.path.exists(directory):
            os.mkdir(directory)

        xmin,xmax=np.min(self.xg[:,0]),np.max(self.xg[:,0])
        zmin,zmax=np.min(self.xg[:,2]),np.max(self.xg[:,2])
        X1=np.linspace(xmin+self.maxsize,xmax-self.maxsize,500)
        Y1=np.linspace(zmin+self.maxsize,zmax-self.maxsize,300)
        #for i in range(self.xg):
        X_grid, Y_grid = np.meshgrid(X1, Y1)
        X=X_grid.flatten()
        Y=Y_grid.flatten()
        mesh1 = np.column_stack((X, Y))
        #print(self.xg[[0,2]].shape, self.slipv.shape)
        slipv_mesh=griddata(self.xg[:,[0,2]], self.slipv, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        #plt.pcolor(slipv_mesh)
        #plt.show()
        f=open(directory+'/X_grid.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%f '%X_grid[i][j])
            f.write('\n')
        f.close()

        f=open(directory+'/Y_grid.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%f '%Y_grid[i][j])
            f.write('\n')
        f.close()


        f=open(directory+'/'+fname+'slipv'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()


        slipv_mesh=griddata(self.xg[:,[0,2]], self.slipv1, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        f=open(directory+'/'+fname+'slipv1'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()

        slipv_mesh=griddata(self.xg[:,[0,2]], self.slipv2, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        f=open(directory+'/'+fname+'slipv2'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()

        slipv_mesh=griddata(self.xg[:,[0,2]], self.Tt, mesh1, method='cubic')
        slipv_mesh=slipv_mesh.reshape((Y1.shape[0],X1.shape[0]))
        f=open(directory+'/'+fname+'Traction'+'.txt','w')
        for i in range(slipv_mesh.shape[0]):
            for j in range(slipv_mesh.shape[1]):
                f.write('%.5f '%slipv_mesh[i][j])
            f.write('\n')
        f.close()

        # plt.pcolor(slipv_mesh)
        # plt.show()





            
        
        










    #def get_coreD()

        #self.readdata(fname)
        #a=self.external_header_length
        #self.data = data
        # 在这里可以进行一些初始化操作
        
    