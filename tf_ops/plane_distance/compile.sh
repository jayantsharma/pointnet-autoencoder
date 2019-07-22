CUDA_PATH=/usr/local/cuda-10.0
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
echo $CUDA_PATH
echo $TF_CFLAGS
echo $TF_LFLAGS

${CUDA_PATH}/bin/nvcc \
  -std=c++11 \
  -I${CUDA_PATH}/include \
  plane_distance.cu \
  -o plane_distance.cu.o \
  -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# /usr/local/cuda-8.0/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
g++ \
  -std=c++11 \
  -shared \
  plane_distance.cpp plane_distance.cu.o \
  -o tf_planedistance_so.so \
  -fPIC \
  ${TF_CFLAGS} \
  -I${CUDA_PATH}/include \
  ${TF_LFLAGS} \
  -L${CUDA_PATH}/lib64/ -lcudart \
  -O2
  # -I /home/jayant/miniconda3/lib/python3.6/site-packages/tensorflow/include/external/nsync/public \
# g++ \
#   -std=c++11 \
#   tf_nndistance.cpp tf_nndistance_g.cu.o \
#   -o tf_nndistance_so.so \
#   -shared -fPIC \
#   -I /usr/local/lib/python2.7/dist-packages/tensorflow/include \
#   -I /usr/local/cuda-8.0/include \
#   -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public \
#   -lcudart \
#   -L /usr/local/cuda-8.0/lib64/ \
#   -L /usr/local/lib/python2.7/dist-packages/tensorflow \
#   -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
