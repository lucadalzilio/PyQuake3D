# ------------------------------------------------------------
# 编译器与编译标志
# ------------------------------------------------------------
CC = gcc
CFLAGS = -Wall -O2 -fPIC -fopenmp -I Library   # -fPIC 用于生成位置无关代码
#LDFLAGS = -L. -lm   # 如果需要 libh2.so，保留 -lh2
LDFLAGS = -L. -lm -fopenmp
# ------------------------------------------------------------
# 目标文件和库
# ------------------------------------------------------------
DYNAMIC_TEST_TARGET = hm.so   # 生成动态库的目标
TEST_SOURCE = mytest/hm.c         # 3dtest的源文件
TEST_TARGET = mytest/hm             # 测试程序3dtest的可执行文件

# ------------------------------------------------------------
# 源文件和目标文件列表（根据实际依赖简化）
# ------------------------------------------------------------
# ------------------------------------------------------------
# Components of the main library
# ------------------------------------------------------------

H2LIB_CORE0 = \
	Library/basic.c \
	Library/settings.c \
	Library/parameters.c \
	Library/opencl.c

H2LIB_CORE1 = \
	Library/avector.c \
	Library/realavector.c \
	Library/amatrix.c \
	Library/factorizations.c \
	Library/eigensolvers.c \
	Library/sparsematrix.c \
	Library/sparsepattern.c \
	Library/gaussquad.c \
	Library/krylov.c

H2LIB_CORE2 = \
	Library/cluster.c \
	Library/clustergeometry.c \
	Library/block.c \
	Library/clusterbasis.c \
	Library/clusteroperator.c \
	Library/uniform.c \
	Library/h2matrix.c \
	Library/rkmatrix.c \
	Library/hmatrix.c \
	Library/krylovsolvers.c

H2LIB_CORE3 = \
	Library/truncation.c \
	Library/harith.c \
	Library/harith2.c \
	Library/hcoarsen.c \
	Library/h2compression.c \
	Library/h2update.c \
	Library/h2arith.c \
	Library/aca.c \
	Library/visualize.c \
	Library/matrixnorms.c

H2LIB_DIRECTIONAL = \
	Library/dcluster.c \
	Library/dblock.c \
	Library/dclusterbasis.c \
	Library/duniform.c \
	Library/dh2matrix.c \
	Library/dh2compression.c \
	Library/dclusteroperator.c

H2LIB_FEM = \
	Library/tri2d.c \
	Library/tri2dp1.c \
	Library/tet3d.c \
	Library/tet3dp1.c\
	Library/ddcluster.c\
	Library/tri2drt0.c\
	Library/tet3drt0.c

H2LIB_BEM = \
	Library/curve2d.c \
	Library/singquad1d.c \
	Library/bem2d.c \
	Library/laplacebem2d.c \
	Library/surface3d.c \
	Library/macrosurface3d.c \
	Library/singquad2d.c \
	Library/bem3d.c \
	Library/oclbem3d.c \
	Library/laplacebem3d.c \
	Library/laplaceoclbem3d.c \
	Library/helmholtzbem3d.c \
	Library/helmholtzoclbem3d.c

SOURCES_libh2 := \
	$(H2LIB_CORE0) \
	$(H2LIB_CORE1) \
	$(H2LIB_CORE2) \
	$(H2LIB_CORE3) \
	$(H2LIB_DIRECTIONAL) \
	$(H2LIB_FEM) \
	$(H2LIB_BEM)

OBJECTS_libh2 := $(SOURCES_libh2:.c=.o)

# 添加3dtest的源文件
SOURCES_3DTEST := $(SOURCES_libh2) $(TEST_SOURCE)
OBJECTS_3DTEST := $(SOURCES_3DTEST:.c=.o)

# ------------------------------------------------------------
# 规则：编译3dtest动态库
# ------------------------------------------------------------
$(DYNAMIC_TEST_TARGET): $(OBJECTS_3DTEST)
	@echo "Building shared library $@"
	$(CC) -shared -fPIC -o $@ $(OBJECTS_3DTEST) $(CFLAGS) $(LDFLAGS)

# ------------------------------------------------------------
# 规则：编译3dtest可执行文件
# ------------------------------------------------------------
$(TEST_TARGET): $(OBJECTS_3DTEST)
	@echo "Building test executable $@"
	$(CC) -o $@ $(OBJECTS_3DTEST) $(LDFLAGS)

# ------------------------------------------------------------
# 编译.c文件为.o文件的规则
# ------------------------------------------------------------
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ------------------------------------------------------------
# 清理编译生成的文件
# ------------------------------------------------------------
.PHONY: clean
clean:
	rm -f $(OBJECTS_3DTEST) $(DYNAMIC_TEST_TARGET) $(TEST_TARGET)
