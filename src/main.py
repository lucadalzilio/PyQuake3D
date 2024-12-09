import readmsh
import SH_greenfunction
import DH_greenfunction
import numpy as np
import sys
import matplotlib.pyplot as plt
import QDsim
from math import *
import time
import argparse
import os


file_name = sys.argv[0]
print(file_name)

if __name__ == "__main__":
    try:
        start_time = time.time()
        parser = argparse.ArgumentParser(description="Process some files and enter interactive mode.")
        parser.add_argument('-g', '--inputgeo', required=True, help='Input msh geometry file to execute')
        parser.add_argument('-p', '--inputpara', required=True, help='Input parameter file to process')

        args = parser.parse_args()

        fnamegeo = args.inputgeo
        fnamePara = args.inputpara
    
    except:
        fnamegeo='../examples/surface0/Surface0.msh'
        fnamePara='../examples/surface0/parameter.txt'
        #fnamegeo='examples/bp5t/bp5t.msh'
        #fnamePara='examples/bp5t/parameter.txt'
    print('Input msh geometry file:',fnamegeo)
    print('Input parameter file:',fnamePara)
    

    
    #fname='bp5t.msh'
    nodelst,elelst=readmsh.read_mshV2(fnamegeo)
    print('Number of Node',nodelst.shape)
    print('Number of Element',elelst.shape)
    #print(boundary_nodes)
    #print(np.max(nodelst[:,0]),np.min(nodelst[:,0]),np.max(nodelst[:,1]),np.min(nodelst[:,1]))
    
    
    #print(eleVec.shape,xg.shape)


    
    sim0=QDsim.QDsim(elelst,nodelst,fnamePara)
    print(sim0.__dict__)
    
    #print(f"Time taken: {end_time - start_time:.2f} seconds")


    #print(sim0.slipv.shape,sim0.Tt.shape)


    Sliplast=np.mean(sim0.slipv)
    sim0.calc_nucleaszie_cohesivezone()
    
    #print(sim0.mu*sim0.dc/(sim0.b[0]*1e6*10))
    SLIP=[]
    SLIPV=[]
    #print(size_nuclear)
    f=open('state.txt','a')
    totaloutputsteps=int(sim0.Para0['totaloutputsteps'])
    for i in range(totaloutputsteps):
        print('iteration:',i)

        # for k in range(len(sim0.arriT)):
        #     if(sim0.slipv[k]>=0.03 and sim0.arriT[k]==1e9):
        #         sim0.arriT[k]=sim0.time

        
        if(i==0):
            dttry=sim0.htry
        else:
            dttry=dtnext
        dttry,dtnext=sim0.simu_forward(dttry)
        print('dt:',dttry,' max_vel:',np.max(np.abs(sim0.slipv)),' Seconds:',sim0.time,'  Days:',sim0.time/3600/24)
        f.write('%f %f %f %f\n' %(dttry,np.max(np.abs(sim0.slipv)),sim0.time,sim0.time/3600.0/24.0))
        
        SLIP.append(sim0.slip)
        SLIPV.append(sim0.slipv)
        
        
        # if(sim0.time>60):
        #     break
        outsteps=int(sim0.Para0['outsteps'])
        directory='out'
        if not os.path.exists(directory):
            os.mkdir(directory)
        if(i%outsteps==0):
            SLIP=np.array(SLIP)
            SLIPV=np.array(SLIPV)
            # np.save('examples/bp5t/slipv/slipv_%d'%i,SLIPV)
            # np.save('examples/bp5t/slip/slip_%d'%i,SLIP)

            SLIP=[]
            SLIPV=[]

            if(sim0.Para0['outputstv']=='True'):
                fname=directory+'/step'+str(i)+'.vtk'
                sim0.ouputVTK(fname)
            if(sim0.Para0['outputmatrix']=='True'):
                fname='step'+str(i)
                sim0.outputtxt(fname)
    
    end_time = time.time()   
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    f.close()
    
    #sim0.test()
    #fname='test.vtk'
    #sim0.ouputVTK(fname)

    # f=open('arrivt.txt','w')
    # for i in range(len(sim0.arriT)):
    #     f.write('%f %f %f\n' %(sim0.xg[i,0],sim0.xg[i,2],sim0.arriT[i]))
    # f.close()


