#/bin/bash
CUDA_PATH=/usr/local/cuda-10.0
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
echo $CUDA_PATH
echo $TF_CFLAGS
echo $TF_LFLAGS

${CUDA_PATH}/bin/nvcc \
  -I${CUDA_PATH}/include \
  tf_nndistance_g.cu \
  -o tf_nndistance_g.cu.o \
  -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ \
  -std=c++11 \
  tf_nndistance.cpp tf_nndistance_g.cu.o \
  -o tf_nndistance_so.so \
  -shared -fPIC \
  ${TF_CFLAGS} \
  -I${CUDA_PATH}/include \
  ${TF_LFLAGS} \
  -L${CUDA_PATH}/lib64/ -lcudart \
  -O2
