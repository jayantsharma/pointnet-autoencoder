#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
REGISTER_OP("PlaneDistance")
    .Input("xyz: float32")
    .Output("dist: float32")
    .Output("offset: float32")
    .Output("normal: float32")
    .Output("idx: int32");
REGISTER_OP("PlaneDistanceGrad")
    .Input("offset: float32")
    .Input("normal: float32")
    .Output("grad: float32");
using namespace tensorflow;


// CPU Ops
const int NUM_NBRS = 10;
int dsvd(float a[][NUM_NBRS], int m, int n, float w[3], float v[][3]);
static void calculate_plane_distance(int b,int n,const float *xyz,float *dist, float *offset, float *normal, int *idx){
    for (int i=0;i<b;i++){
      for (int j=0;j<n;j++){
        float x1=xyz[(i*n+j)*3+0];
        float y1=xyz[(i*n+j)*3+1];
        float z1=xyz[(i*n+j)*3+2];
        // priority_queue<pair<float, int> > nn_dist;
        // pair<float, int> max_dist;
        float nn_dist [NUM_NBRS];
        int nn_idx [NUM_NBRS];
        int insert_idx = 0;
        int max_idx = 0;
        for (int k=0;k<n;k++){
          if (k==j){
            continue;
          }
          float x2=xyz[(i*n+k)*3+0]-x1;
          float y2=xyz[(i*n+k)*3+1]-y1;
          float z2=xyz[(i*n+k)*3+2]-z1;
          double d=x2*x2+y2*y2+z2*z2;
          if(insert_idx < NUM_NBRS){   // (LOOP) counter value less than queue capacity, so append
            nn_dist[insert_idx] = d;
            nn_idx[insert_idx] = k;
            if(d > nn_dist[max_idx]){
              max_idx = insert_idx;
            }
            insert_idx++;
          }
          else if(d < nn_dist[max_idx]){ // LOOP
            if(d < nn_dist[max_idx]){
              nn_dist[max_idx] = d;
              nn_idx[max_idx] = k;
              // Find new max_idx
              max_idx = 0;
              for(int l=1; l < NUM_NBRS; l++){
                if(nn_dist[l] > nn_dist[max_idx]){
                  max_idx = l;
                }
              }
            }
          }
        }

        // Store k nearest nbr indices
        for(int k=0; k<NUM_NBRS; k++){
          idx[(i*n+j)*NUM_NBRS+k] = nn_idx[k];
        }

        // Init matrices to hold SVD results
        // float **a = new float *[NUM_NBRS];
        // for(int k=0; k<NUM_NBRS; k++)
        //   a[k] = new float[NUM_NBRS];
        // float **v = new float *[3];
        // for(int k=0; k<3; k++)
        //   v[k] = new float[3];
        float a[NUM_NBRS][NUM_NBRS];
        float v[3][3];
        float w[3];

        // Copy over nearest nbrs to a
        for(int k=0; k<NUM_NBRS; k++)
          for(int l=0; l<3; l++)
            a[k][l] = xyz[(i*n+nn_idx[k])*3+l];

        // calculate centroid
        float centroidx = 0;
        float centroidy = 0;
        float centroidz = 0;
        for(int k=0; k<NUM_NBRS; k++){
          centroidx += a[k][0];
          centroidy += a[k][1];
          centroidz += a[k][2];
        }
        centroidx /= NUM_NBRS;
        centroidy /= NUM_NBRS;
        centroidz /= NUM_NBRS;

        // subtract centroid
        for(int k=0; k<NUM_NBRS; k++){
          a[k][0] -= centroidx;
          a[k][1] -= centroidy;
          a[k][2] -= centroidz;
        }

        // Calculate SVD
        dsvd(a, NUM_NBRS, 3, w, v);
        // Find smallest singular value
        int minidx = (w[0] < w[1]) ? 0 : 1 ;
        minidx = (w[minidx] < w[2]) ? minidx : 2 ;

        float nrm[3];
        nrm[0] = v[0][minidx];
        nrm[1] = v[1][minidx];
        nrm[2] = v[2][minidx];
        normal[(i*n+j)*3+0] = nrm[0];
        normal[(i*n+j)*3+1] = nrm[1];
        normal[(i*n+j)*3+2] = nrm[2];

        // Calculate offset
        float o = (x1-centroidx)*nrm[0] + (y1-centroidy)*nrm[1] + (z1-centroidz)*nrm[2];
        offset[i*n+j] = o;
        dist[i*n+j] = 0.5*o*o;

        // for(int k=0; k<NUM_NBRS; k++)
        //   free((void*) a[k]);
        // free((void**) a);
        // for(int k=0; k<3; k++)
        //   free((void*) v[k]);
        // free((void**) v);
      }
    }
}
static void calculate_plane_distance_grad(int b,int n,const float *offset,const float *normals,float *grad){
    for (int i=0; i<b; i+=1){
      for (int j=0; j<n; j+=1){
        float g=offset[i*n+j];
        float nx=normals[(i*n+j)*3+0];
        float ny=normals[(i*n+j)*3+1];
        float nz=normals[(i*n+j)*3+2];
        grad[(i*n+j)*3+0] = g*nx;
        grad[(i*n+j)*3+1] = g*ny;
        grad[(i*n+j)*3+2] = g*nz;
      }
    }
}

class PlaneDistanceOp : public OpKernel{
    public:
        explicit PlaneDistanceOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext * context)override{
            const Tensor& xyz_tensor=context->input(0);
            OP_REQUIRES(context,xyz_tensor.dims()==3,errors::InvalidArgument("PlaneDistance requires xyz be of shape (batch,#points,3)"));
            OP_REQUIRES(context,xyz_tensor.shape().dim_size(2)==3,errors::InvalidArgument("PlaneDistance only accepts 3d point set xyz"));
            int b=xyz_tensor.shape().dim_size(0);
            int n=xyz_tensor.shape().dim_size(1);
            auto xyz_flat=xyz_tensor.flat<float>();
            const float * xyz=&xyz_flat(0);

            Tensor *dist_tensor=NULL;
            Tensor *offset_tensor=NULL;
            Tensor *plane_tensor=NULL;
            Tensor *idx_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&dist_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&offset_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,n,3},&plane_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape{b,n,NUM_NBRS},&idx_tensor));
            auto dist_flat=dist_tensor->flat<float>();
            auto offset_flat=offset_tensor->flat<float>();
            auto plane_flat=plane_tensor->flat<float>();
            auto idx_flat=idx_tensor->flat<int>();
            float * dist=&(dist_flat(0));
            float * offset=&(offset_flat(0));
            float * plane=&(plane_flat(0));
            int * idx=&(idx_flat(0));
            calculate_plane_distance(b,n,xyz,dist,offset,plane,idx);
        }
};
REGISTER_KERNEL_BUILDER(Name("PlaneDistance").Device(DEVICE_CPU), PlaneDistanceOp);

class PlaneDistanceGradOp : public OpKernel{
    public:
        explicit PlaneDistanceGradOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext * context)override{
            const Tensor& offset_tensor=context->input(0);
            const Tensor& plane_tensor=context->input(1);
            OP_REQUIRES(context,offset_tensor.dims()==2,errors::InvalidArgument("PlaneDistanceGrad requires offset be of shape (batch,#points)"));
            int b=offset_tensor.shape().dim_size(0);
            int n=offset_tensor.shape().dim_size(1);
            OP_REQUIRES(context,plane_tensor.shape()==(TensorShape{b,n,3}),errors::InvalidArgument("PlaneDistanceGrad requires plane be of shape(batch,#points,3)"));
            auto offset_flat=offset_tensor.flat<float>();
            const float * offset=&offset_flat(0);
            auto plane_flat=plane_tensor.flat<float>();
            const float * plane=&plane_flat(0);

            Tensor * grad_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&grad_tensor));
            auto grad_flat=grad_tensor->flat<float>();
            float * grad=&grad_flat(0);
            calculate_plane_distance_grad(b,n,offset,plane,grad);
        }
};
REGISTER_KERNEL_BUILDER(Name("PlaneDistanceGrad").Device(DEVICE_CPU), PlaneDistanceGradOp);


// GPU Ops
void PlaneDistanceKernelLauncher(int b, int n, const float *xyz, float *dist, float *offset, float *normal, int*idx);
class PlaneDistanceGpuOp : public OpKernel{
    public:
        explicit PlaneDistanceGpuOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext * context)override{
            const Tensor& xyz_tensor=context->input(0);
            OP_REQUIRES(context,xyz_tensor.dims()==3,errors::InvalidArgument("PlaneDistance requires xyz be of shape (batch,#points,3)"));
            OP_REQUIRES(context,xyz_tensor.shape().dim_size(2)==3,errors::InvalidArgument("PlaneDistance only accepts 3d point set xyz"));
            int b=xyz_tensor.shape().dim_size(0);
            int n=xyz_tensor.shape().dim_size(1);
            auto xyz_flat=xyz_tensor.flat<float>();
            const float * xyz=&xyz_flat(0);

            Tensor *dist_tensor=NULL;
            Tensor *offset_tensor=NULL;
            Tensor *plane_tensor=NULL;
            Tensor *idx_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&dist_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&offset_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,n,3},&plane_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape{b,n,NUM_NBRS},&idx_tensor));
            auto dist_flat=dist_tensor->flat<float>();
            auto offset_flat=offset_tensor->flat<float>();
            auto plane_flat=plane_tensor->flat<float>();
            auto idx_flat=idx_tensor->flat<int>();
            float * dist=&(dist_flat(0));
            float * offset=&(offset_flat(0));
            float * plane=&(plane_flat(0));
            int * idx=&(idx_flat(0));
            PlaneDistanceKernelLauncher(b,n,xyz,dist,offset,plane,idx);
        }
};
REGISTER_KERNEL_BUILDER(Name("PlaneDistance").Device(DEVICE_GPU), PlaneDistanceGpuOp);

void PlaneDistanceGradKernelLauncher(int b, int n, const float *offset, const float *normals, float *grad);
class PlaneDistanceGradGpuOp : public OpKernel{
    public:
        explicit PlaneDistanceGradGpuOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext * context)override{
            const Tensor& offset_tensor=context->input(0);
            const Tensor& plane_tensor=context->input(1);
            OP_REQUIRES(context,offset_tensor.dims()==2,errors::InvalidArgument("PlaneDistanceGrad requires offset be of shape (batch,#points)"));
            int b=offset_tensor.shape().dim_size(0);
            int n=offset_tensor.shape().dim_size(1);
            OP_REQUIRES(context,plane_tensor.shape()==(TensorShape{b,n,3}),errors::InvalidArgument("PlaneDistanceGrad requires plane be of shape(batch,#points,3)"));
            auto offset_flat=offset_tensor.flat<float>();
            const float * offset=&offset_flat(0);
            auto plane_flat=plane_tensor.flat<float>();
            const float * plane=&plane_flat(0);

            Tensor * grad_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,3},&grad_tensor));
            auto grad_flat=grad_tensor->flat<float>();
            float * grad=&grad_flat(0);
            PlaneDistanceGradKernelLauncher(b,n,offset,plane,grad);
        }
};
REGISTER_KERNEL_BUILDER(Name("PlaneDistanceGrad").Device(DEVICE_GPU), PlaneDistanceGradGpuOp);
