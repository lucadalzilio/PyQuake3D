#include <stdio.h>

#include "basic.h"
#include "krylovsolvers.h"
#include "laplacebem3d.h"
#include "ddcluster.h"




struct prStruct {
    uint vertices;
    uint edges;
    uint triangles;
    real *x;
    uint *e;
    uint *t;
    uint *s;
};

// 用于存储所有对象的动态数组
#define MAX_OBJECTS 100
phmatrix objectsHm[MAX_OBJECTS];
int object_count = 0;

// int MAX_OBJECTS=10;

// phmatrix* objectsHm = NULL;
// int object_count = 0;


phmatrix  Ghg;			/* H-matrix */
pamatrix  G,Gp;
pcluster  root;
pblock    broot;
real      eps;
int n, m;
ptruncmode tm;
pavector x,yh;


void createHmatrixstructure(struct prStruct *temp)
{
  psurface3d gr;
  uint q_reg, q_sing, clf;
  basisfunctionbem3d basis;
  pbem3d    bem_slp;
  
  real      eta;

  /* Number of quadrature points for regular integrals. */
  q_reg = 2;
  /* Number of quadrature points for singular integrals. */
  q_sing = q_reg + 2;
  eta=2;
  m=4;
  clf = 2 * m * m * m;
  n=temp->triangles;
  eps=5e-5;
  basis = BASIS_CONSTANT_BEM3D;

  printf("vertices = %d, edges = %d, triangles = %d\n", temp->vertices, temp->edges, temp->triangles);
  gr = new_surface3d(temp->vertices, temp->edges, temp->triangles);
  
  for (int i = 0; i < temp->vertices; i++)
  {
    //printf("%f\n",temp->x[i][0]);
    gr->x[i][0]=temp->x[i*3];
    gr->x[i][1]=temp->x[i*3+1];
    gr->x[i][2]=temp->x[i*3+2];
    //printf("%f \n",temp->x[i]);
  }
  
  for (int i = 0; i <temp->edges; i++) 
  {
    gr->e[i][0]=temp->e[2*i];
    gr->e[i][1]=temp->e[2*i+1];
    //printf("%d \n",temp->e[i]);
  }

  for (int i = 0; i <temp->triangles; i++) 
  {
    gr->t[i][0]=temp->t[i*3];
    gr->t[i][1]=temp->t[i*3+1];
    gr->t[i][2]=temp->t[i*3+2];
    gr->s[i][0]=temp->s[i*3];
    gr->s[i][1]=temp->s[i*3+1];
    gr->s[i][2]=temp->s[i*3+2];

    //printf("%d \n",temp->s[i*3]);
  }
  prepare_surface3d(gr);
  bem_slp = new_slp_laplace_bem3d(gr, q_reg, q_sing, basis, basis);
  root = build_bem3d_cluster(bem_slp, clf, basis);
  broot = build_nonstrict_block(root, root, &eta, admissible_2_min_cluster);
  Ghg = build_from_block_hmatrix(broot, m);
  G = new_amatrix(n, n);
  Gp = new_amatrix(n, n);
  tm = new_releucl_truncmode();

  x = new_avector(n);
  yh = new_avector(n);

  //objectsHm = (phmatrix*)malloc(MAX_OBJECTS * sizeof(phmatrix));
}





// 创建对象并保存
void create_Hmvalue(float* arrayA) {
    if (object_count < MAX_OBJECTS) 
    {
        //phmatrix* new_objectHm = (phmatrix*)malloc(sizeof(phmatrix));
        phmatrix new_objectHm;
        //new_objectHm=Ghg;
        //memcpy(new_objectHm, Ghg, sizeof(phmatrix));
        new_objectHm = build_from_block_hmatrix(broot, m);
        
         for(int i=0;i<n;i++)
        {
          for(int j=0;j<n;j++)
          {
            G->a[j+G->ld*i]=arrayA[i+G->ld*j];
          }
        }
        for(int j=0; j<n; j++)
        {
          //printf("j: %d,root->idx: %d \n", j,root->idx[j]);
          for(int i=0; i<n; i++)
          {
              Gp->a[i+j*Gp->ld] = G->a[root->idx[i]+root->idx[j]*G->ld];
          }
        }
        
        // new_object->id = id;
        // new_object->value = value;
        clear_hmatrix(new_objectHm);
        add_amatrix_hmatrix(1.0,false,Gp,tm,eps,new_objectHm);

        //objectsHm[object_count] = (phmatrix)malloc(sizeof(phmatrix));
        objectsHm[object_count] = new_objectHm;

        object_count++;
        printf("Created object with id: %d\n", object_count);
        


    }

    else 
    {
        printf("Error: Maximum hmatrix object limit reached.\n");
    }
}


float* Hmatrix_dot_X(int k_,float *X_)
{
  float* y_ = (float*)malloc(n * sizeof(float));
  for(int i=0;i<n;i++)
  {
    x->v[i]=X_[i];
  }
  //printf("!!!!!!!!!!!!!!!!!!!!!!%d",k_);
  //phmatrix  obj;
  //obj=build_from_block_hmatrix(broot, m);
  //copy_hmatrix(objectsHm[k_],obj);
  clear_avector(yh);
  addeval_hmatrix_avector(1.0, objectsHm[k_], x, yh);
  for(int i=0;i<n;i++)
  {
    y_[i]=yh->v[i];
  }
  
  return y_;
}






