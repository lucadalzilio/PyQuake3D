# PyQuake3D

![examples](https://github.com/lucadalzilio/PyQuake3D/guide/Logo-PyQuake3D.png)

PyQuake3D is a Python-based Boundary Element Method (BEM) code for simulating sequences of seismic and aseismic slip (SEAS) on a complex 3D fault geometry governed by rate- and state-dependent friction. This document provides an overview of how to use the script, as well as a detailed description of the input parameters.

# Contribution
Dr. Rongjiang Tang and Dr. Luca Dal Zilio developed the code framework and the Quasi-dynamic BIEM Seismic cycle model on a complex 3D fault geometry governed by regularized aging law. Dr.Rongjiang Tang and Dr.Luca Dal Zilio implemented the H-matrix Matrix-Vector Multiplication and developed the Cascadia model.

##requirements
numpy>=1.2
ctypes==1.1
python>=3.8

## Running the Script

To run the PyQuake3D script, use the following command:
```bash
python -g --inputgeo <input_geometry_file> -p --inputpara <input_parameter_file>
```
For example,To execute benchmarks like BP5-QD at Current Directory:
```bash
python src/main.py -g --examples/bp5t/bp5t.msh -p --examples/bp5t/parameter.txt
```
To run cascadia model, use:
```bash
python src/main.py -g --examples/cascadia/cascadia35km_ele4.msh -p --examples/cascadia/parameter.txt
```
Ensure you modify the input parameter (`parameter.txt`) as follows:
- `Corefunc directory`: `bp5t_core`
- `InputHetoparamter`: `True`
- `Inputparamter file`: `bp5tparam.dat`

## Input Parameter Description

### General Parameters

| Parameter                   | Default | Description |
|-----------------------------|---------|-------------|
| `Corefunc directory`        |         | The storage path for the kernel function matrix composed of stress Green's functions. |
| `InputHetoparamter`         | False   | If `True`, the heterogeneous stress and friction parameters are imported from external files. |
| `Inputparamter file`        |         | The file name of imported heterogeneous stress and friction parameters. |
| `Processors`                |         |             |
| `Processes`                 | 50      | The number of processes which can be scheduled by the operating system to run on different CPU cores. |
| `Batch_size`                | 1000    | The number of tasks for each batch of processes. If out of memory, Please reduce the size of Processes and Batch size|
| `H-matrix`                  | False   | If `True`, The kernel function will be approximated using HMatrix. Note that in this case, the code must be run under Linux system. The H-Matrix approximation is implemented using the open-source C library H2Lib, and the compilation of the dynamic library is done based on a Makefile. |
Note:
If you want to modify the matrix-vector multiplication computation in hmatrix, you can find the C source code in the H2Lib-master/mytest directory. The related Python source code is located in H2Lib-master/hmatrix.py. After making the modifications, recompile the code using make to generate a new hm.so dynamic library, and replace the dynamic library file in the src directory.

If InputHetoparamter is 'False', you may build the model by modifying the 'Stress Settings','Friction Settings' and 'Nucleation Settings':
### Stress Settings
| Parameter                                   | Default      | Description |
|---------------------------------------------|--------------|-------------|
| `Vertical principal stress`                 | 1.0          | The vertical principal stress scale: the real vertical principal stress is obtained by multiplying the scale and the value. |
| `Maximum horizontal principal stress`       | 1.6          | Maximum horizontal principal stress scale. |
| `Minimum horizontal principal stress`       | 0.6          | Minimum horizontal principal stress scale. |
| `Angle between ssh1 and X-axis`             | 30 degree    | Angle between maximum horizontal principal stress and X-axis. |
| `Vertical principal stress value`           |              | Vertical principal stress value. |
| `Vertical principal stress value varies with depth` | True       | Vertical principal stress value varies with depth. |
| `Turnning depth`                            | 15000 m      | After the vertical principal stress changes with depth, it maintains a constant value at the conversion depth, and the horizontal principal stress value also changes with depth simultaneously. |
| `Shear traction solved from stress tensor`  | False        | If `True`, the non-uniform shear stress is projected onto the curved fault surface by the stress tensor. |
| `Rake solved from stress tensor`            | True         | If `True`, the non-uniform rakes are solved from the stress tensor. |
| `Fix_rake`                                  | 30 degree    | Set fixed rakes if `Rake solved from stress tensor` is `False`. |
| `Shear traction in VS region`               | 0.53         | The ratio of shear traction to normal traction in the velocity strengthening region. |
| `Shear traction in VW region`               | 0.78         | The ratio of shear traction to normal traction in the velocity weakening region. |
| `Shear traction in nucleartion region`      | 1.0          | The ratio of shear traction to normal traction in the velocity nucleation region. |
| `Widths of VS region`                       | 10000 m      | The width of the velocity weakening region. |
| `Transition region ratio from VS to VW region` | 0.4      | The ratio of transition to VS region. |
| `Radius of nucleartion`                     | 8000 m       | The radius of the nucleation region. |

### Friction Settings

| Parameter                                   | Default      | Description |
|---------------------------------------------|--------------|-------------|
| `Reference slip rate`                       | 1e-6         | Reference slip rate. |
| `Reference friction coefficient`            | 0.6          | Reference friction coefficient. |
| `Rate-and-state parameters a in VS region`  | 0.04         | Rate-and-state parameters `a` in the velocity strengthening region. |
| `Rate-and-state parameters b in VS region`  | 0.03         | Rate-and-state parameters `b` in the velocity strengthening region. |
| `Characteristic slip distance in VS region` | 0.14         | Characteristic slip distance in the velocity strengthening region. |
| `Rate-and-state parameters a in VW region`  | 0.004        | Rate-and-state parameters `a` in the velocity weakening region. |
| `Rate-and-state parameters b in VW region`  | 0.03         | Rate-and-state parameters `b` in the velocity weakening region. |
| `Characteristic slip distance in VW region` | 0.14         | Characteristic slip distance in the velocity weakening region. |
| `Rate-and-state parameters a in nucleation region` | 0.004   | Rate-and-state parameters `a` in the nucleation region. |
| `Rate-and-state parameters b in nucleartion region` | 0.03   | Rate-and-state parameters `b` in the nucleation region. |
| `Characteristic slip distance in nucleartion region` | 0.13 | Characteristic slip distance in the nucleation region. |
| `Initial slip rate in nucleartion region`   | 0.03         | Initial slip rate in the nucleation region. |
| `Plate loading rate`                        | 1e-6 m/s     | Plate loading rate. |

### Nucleation Settings

| Parameter                                   | Default      | Description |
|---------------------------------------------|--------------|-------------|
| `Set_nucleation`                            | True         | If `True`, sets a patch whose shear stress and sliding rate are significantly greater than the surrounding area to meet the nucleation requirements. |
| `Nucleartion_posx`                          |              | Nucleation zone x coordinate. |
| `Nucleartion_posy`                          |              | Nucleation zone y coordinate. |
| `Nucleartion_posz`                          |              | Nucleation zone z coordinate. |

### Output Settings

Further output settings can be customized based on specific needs and configurations required for the simulation. 

## Example Usage

To run a benchmark test using the BP5-QD configuration:
```bash
python -g --bp5t.msh -p --parameter.txt
```
Ensure that the `parameter.txt` file is appropriately configured as described in the input parameter description.

## Contributing

We welcome contributions to PyQuake3D. Please ensure that you follow the contribution guidelines and maintain the consistency of the codebase.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

We would like to thank all the contributors and the community for their support and feedback.

We referred to the HBI code to develop original python-based BIEM algorithm:
Ozawa, S., Ida, A., Hoshino, T., & Ando, R. (2023). Large-scale earthquake sequence simulations on 3-D non-planar faults using the boundary element method accelerated by lattice H-matrices. Geophysical Journal International, 232(3), 1471-1481.

We referred to MATLAB code to develop the kernel function:
Nikkhoo, M., & Walter, T. R. (2015). Triangular dislocation: an analytical, artefact-free solution. Geophysical Journal International, 201(2), 1119-1141.

The implementation of the hierarchical matrix is mainly based on open-source code H2lib. http://www.h2lib.org/.

We would like to thank Associate Professor Ando Ryosuke and Dr.So Ozawa for their help in the code development, and Professor Steffen Börm for his assistance with HMatrix programming.


For further details, please refer to the official documentation or contact the development team.

Rongjiang Tang: rongjiang.igp@hotmail.com
Luca Dal Zilio: luca.dalzilio@ntu.edu.sg

