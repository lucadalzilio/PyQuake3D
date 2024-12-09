#include <stdio.h>

#include "basic.h"
#include "krylovsolvers.h"
#include "laplacebem3d.h"
#include "ddcluster.h"
#include <omp.h>

int readmatrixdata(const char *filename, real **matrix)
{
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
      perror("无法打开文件");
      return -1; // 返回错误码
  }

  int rows,cols;

  fscanf(file, "%d %d", &rows, &cols);
  
  

    // 读取矩阵数据
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) 
        {
            fscanf(file, "%lf", &matrix[i][j]);
        }
    }
  fclose(file);
  return 0; // 成功
}


int
mytest(int argc, char **argv)
{
  omp_set_num_threads(16);
  pstopwatch sw;
  pamatrix  G,Gp;			/* Original matrix */
  phmatrix  Gh;			/* H-matrix */
  ph2matrix G2;			/* H^2-matrix approximation */

  pmacrosurface3d mg;
  psurface3d gr;
  pbem3d    bem_slp, bem_dlp;
  uint      q_reg, q_sing;
  basisfunctionbem3d basis;
  pamatrix  V, KM;
  pavector  gd, b, x;
  real      eps_solve;
  uint      maxiter;
  real      t, size, norm;
  pavector  y, y1, yh, yh2;		/* Vectors for testing */
  pcluster  root;
  uint      clf,n,nn;
  pblock    broot;
  ptruncmode tm,tm2;		/* Truncation strategy */
  real      eta;		/* admissibility parameter */
  uint      m;			/* Approximation order */
  pclusterbasis rbf, cbf;	/* Adaptive cluster bases */
  real      eps;		/* Compression tolerance */
  real      error, normG;	/* Norm and error estimate */
  real **matrix;

  float tem,x1_,y1_,x2_,y2_,x3_,y3_,z1_,z2_,z3_,R;
  real (*gcoord)[3];
  
  eps = 5.0e-5;
  eta = 2.0;
  m=4;
  clf = 2 * m * m * m;
  basis = BASIS_CONSTANT_BEM3D;
  

  init_h2lib(&argc, &argv);
  /* Number of quadrature points for regular integrals. */
  q_reg = 2;
  /* Number of quadrature points for singular integrals. */
  q_sing = q_reg + 2;

  mg = new_sphere_macrosurface3d();
  /* Mesh the abstract geometry with 32 levels of refinement. */
  gr = build_from_macrosurface3d_surface3d(mg, 32);
  
  const char *filename = "1/trans_msh.txt";  
  //write_surface3d(gr, filename);
  gr=read_surface3d(filename);
  prepare_surface3d(gr);
  n=gr->triangles;

  gcoord=(real(*)[3]) allocmem((size_t) sizeof(real[3]) * n);
  

  for(int i=0;i<n;i++)
  {
    x1_=gr->x[gr->t[i][0]][0];
    y1_=gr->x[gr->t[i][0]][1];
    z1_=gr->x[gr->t[i][0]][2];

    x2_=gr->x[gr->t[i][1]][0];
    y2_=gr->x[gr->t[i][1]][1];
    z2_=gr->x[gr->t[i][1]][2];

    x3_=gr->x[gr->t[i][2]][0];
    y3_=gr->x[gr->t[i][2]][1];
    z3_=gr->x[gr->t[i][2]][2];

    gcoord[i][0]=(x1_+x2_+x3_)/3.0;
    gcoord[i][1]=(y1_+y2_+y3_)/3.0;
    gcoord[i][2]=(z1_+z2_+z3_)/3.0;

    
  }


  bem_slp = new_slp_laplace_bem3d(gr, q_reg, q_sing, basis, basis);

  root = build_bem3d_cluster(bem_slp, clf, basis);
  broot = build_nonstrict_block(root, root, &eta, admissible_2_min_cluster);
  //setup_hmatrix_aprx_inter_row_bem3d(bem_slp, root, root, broot, m);

  Gh = build_from_block_hmatrix(broot, m);
  //assemble_bem3d_hmatrix(bem_slp, broot, Gh);
  
  
  G = new_amatrix(n, n);
  //bem_slp->nearfield(NULL, NULL, bem_slp, false, G);
  //assemble_bem3d_amatrix(bem_slp, G);
  //random_amatrix(G);
  
  printf("Created geometry with %d vertices, %d edges and %d triangles %d rootsize\n",
	 gr->vertices, gr->edges, gr->triangles,root->size);

  
  matrix = (real **)malloc(n * sizeof(real *));
  for (int i = 0; i < n; i++) 
  {
      matrix[i] = (real *)malloc(n * sizeof(real));
  }
  filename = "1/matrixA1d.txt";  
  readmatrixdata(filename, matrix);
  for(int i=0;i<n;i++)
  {
    for(int j=0;j<n;j++)
    {
      x1_=gcoord[i][0];
      y1_=gcoord[i][1];
      z1_=gcoord[i][2];

      x2_=gcoord[j][0];
      y2_=gcoord[j][1];
      z2_=gcoord[j][2];

      
      //R=sqrt((x2_-x1_)*(x2_-x1_)+(y2_-y1_)*(y2_-y1_)+(z2_-z1_)*(z2_-z1_))+0.1;
      //G->a[j+G->ld*i]=1.0/R;
    G->a[j+G->ld*i]=matrix[j][i];
      // if(i%100==0 && j%100==0)
      //  printf("%d %d %f\n",i,j,G->a[j+G->ld*i]);
    }
  }


  

  clear_hmatrix(Gh);

  Gp = new_amatrix(root->size, root->size);
  printf("root->size: %d,n: %d \n", root->size,n);

   for(int j=0; j<root->size; j++)
   {
     //printf("j: %d,root->idx: %d \n", j,root->idx[j]);
     for(int i=0; i<root->size; i++)
     {
        Gp->a[i+j*Gp->ld] = G->a[root->idx[i]+root->idx[j]*G->ld];
     }
   }

  tm = new_releucl_truncmode();
  add_amatrix_hmatrix(1.0,false,Gp,tm,eps,Gh);
  

  tm2 = new_blockreleucl_truncmode();
  rbf = buildrowbasis_amatrix(G, broot, tm2, eps * 0.25);
  cbf = buildcolbasis_amatrix(G, broot, tm2, eps * 0.25);
  G2 = build_projected_amatrix_h2matrix(G, broot, rbf, cbf);

  normG = norm2_amatrix(G);
  error=norm2diff_amatrix_h2matrix(G2, G)/normG;


  float error1=norm2diff_amatrix_hmatrix(Gh, G)/normG;

  (void) printf("error %.20f %.20f\n",error,error1);

x = new_avector(G->cols);
//random_avector(x);
for(int i=0;i<n;i++)
  {
    x->v[i]=0.001;
  }

y = new_avector(G->rows);
yh = new_avector(G->rows);
yh2 = new_avector(G->rows);
//fill_avector(x, 1.0 / G->rows);
clear_avector(y);
clear_avector(yh);
clear_avector(yh2);



struct timeval start, end;
gettimeofday(&start, NULL);
addeval_amatrix_avector(1.0, G, x, y);
gettimeofday(&end, NULL);
double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
printf("cpu_time_amatrix: %.22lf 秒\n", time_taken);



gettimeofday(&start, NULL);

addeval_hmatrix_avector(1.0, Gh, x, yh);
gettimeofday(&end, NULL);
time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
printf("cpu_time_hmatrix: %.22lf 秒\n", time_taken);


gettimeofday(&start, NULL);
addeval_h2matrix_avector(1.0, G2, x, yh2);
//addeval_hmatrix_avector(1.0, Gh, x, yh);
gettimeofday(&end, NULL);
time_taken = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
printf("cpu_time_h2matrix: %.22lf 秒\n", time_taken);

  // Your code here

    

  
  for(int i=0;i<20;i++)
  {
    // for(int j=0;j<50;j++)
    //     printf("%d %d %.12f\n",i,j,G->a[i*n+j]);
    printf("%.12f %.12f %.12f %.12f\n",y->v[i],yh->v[i],yh2->v[i],G->a[i]);


    //printf("%.12f %.12f\n",G->a[i],Gh->f->a[i]);
  }



  return 0;



}