#include <queue>
#include <bits/stdc++.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
REGISTER_OP("PlaneDistance")
    .Input("xyz: float32")
    .Output("dist: float32")
    .Output("idx: int32");
using namespace tensorflow;
using namespace std;

template<class Vector3>
std::pair<Vector3, Vector3> best_plane_from_points(const std::vector<Vector3> & c){
  // copy coordinates to  matrix in Eigen format
  size_t num_atoms = c.size();
  Eigen::Matrix< Vector3::Scalar, Eigen::Dynamic, Eigen::Dynamic > coord(3, num_atoms);
  for (size_t i = 0; i < num_atoms; ++i) coord.col(i) = c[i];

  // calculate centroid
  Vector3 centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

  // subtract centroid
  coord.row(0).array() -= centroid(0); coord.row(1).array() -= centroid(1); coord.row(2).array() -= centroid(2);

  // we only need the left-singular matrix here
  //  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
  auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  Vector3 plane_normal = svd.matrixU().rightCols<1>();
  return std::make_pair(centroid, plane_normal);
}

static void planesearch(int b,int n,const float * xyz,float * dist,int * idx){
    for (int i=0;i<b;i++){
      num_nbrs = 5;
      std::priority_queue<pair<float, int> > nn_dist;
      // std::priority_queue<float> nn_dist;
      // float nn_dist [num_nbrs];
      // float nn_idx [num_nbrs];
      // float max_idx;
      for (int j=0;j<n;j++){
        float x1=xyz[(i*n+j)*3+0];
        float y1=xyz[(i*n+j)*3+1];
        float z1=xyz[(i*n+j)*3+2];
        for (int k=0;k<n;k++){
          if (k==j){
            continue;
          }
          float x2=xyz[(i*n+k)*3+0]-x1;
          float y2=xyz[(i*n+k)*3+1]-y1;
          float z2=xyz[(i*n+k)*3+2]-z1;
          double d=x2*x2+y2*y2+z2*z2;
          if(k < num_nbrs){
            nn_dist.push(std::make_pair(d,k));
            pair<float, int> max_dist = nn_dist.top();
            // nn_dist[k] = d;
            // nn_idx[k] = k;
            // if(k == 0 || d > nn_dist[max_idx]){
            //   max_idx = k;
            // }
          }
          else if(d < max_dist.first){
            nn_dist.pop();
            nn_dist.push(std::make_pair(d,k));
            pair<float, int> max_dist = nn_dist.top();
            // if(d < nn_dist[max_idx]){
            //   nn_dist[max_idx] = d;
            //   nn_idx[max_idx] = k;
            //   // Find new max_idx
            //   max_idx = 0;
            //   for(int l=1; l < num_nbrs; l++){
            //     if(nn_dist[l] > nn_dist[max_idx]){
            //       max_idx = l;
            //     }
            //   }
            // }
          }
        }
        dist[i*n+j]=best;
        idx[i*n+j]=besti;

        // Find best-fit plane now
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
            Tensor * dist_tensor=NULL;
            Tensor * idx_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&dist_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&idx_tensor));
            auto dist_flat=dist_tensor->flat<float>();
            auto idx_flat=idx_tensor->flat<int>();
            float * dist=&(dist_flat(0));
            int * idx=&(idx_flat(0));
            planesearch(b,n,xyz,dist,idx);
        }
};
REGISTER_KERNEL_BUILDER(Name("PlaneDistance").Device(DEVICE_CPU), PlaneDistanceOp);

void NmDistanceKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i);
class PlaneDistanceGpuOp : public OpKernel{
    public:
        explicit PlaneDistanceGpuOp(OpKernelConstruction* context):OpKernel(context){}
        void Compute(OpKernelContext * context)override{
            const Tensor& xyz1_tensor=context->input(0);
            const Tensor& xyz2_tensor=context->input(1);
            OP_REQUIRES(context,xyz1_tensor.dims()==3,errors::InvalidArgument("PlaneDistance requires xyz1 be of shape (batch,#points,3)"));
            OP_REQUIRES(context,xyz1_tensor.shape().dim_size(2)==3,errors::InvalidArgument("PlaneDistance only accepts 3d point set xyz1"));
            int b=xyz1_tensor.shape().dim_size(0);
            int n=xyz1_tensor.shape().dim_size(1);
            OP_REQUIRES(context,xyz2_tensor.dims()==3,errors::InvalidArgument("PlaneDistance requires xyz2 be of shape (batch,#points,3)"));
            OP_REQUIRES(context,xyz2_tensor.shape().dim_size(2)==3,errors::InvalidArgument("PlaneDistance only accepts 3d point set xyz2"));
            int m=xyz2_tensor.shape().dim_size(1);
            OP_REQUIRES(context,xyz2_tensor.shape().dim_size(0)==b,errors::InvalidArgument("PlaneDistance expects xyz1 and xyz2 have same batch size"));
            auto xyz1_flat=xyz1_tensor.flat<float>();
            const float * xyz1=&xyz1_flat(0);
            auto xyz2_flat=xyz2_tensor.flat<float>();
            const float * xyz2=&xyz2_flat(0);
            Tensor * dist1_tensor=NULL;
            Tensor * idx1_tensor=NULL;
            Tensor * dist2_tensor=NULL;
            Tensor * idx2_tensor=NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n},&dist1_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{b,n},&idx1_tensor));
            auto dist1_flat=dist1_tensor->flat<float>();
            auto idx1_flat=idx1_tensor->flat<int>();
            OP_REQUIRES_OK(context,context->allocate_output(2,TensorShape{b,m},&dist2_tensor));
            OP_REQUIRES_OK(context,context->allocate_output(3,TensorShape{b,m},&idx2_tensor));
            auto dist2_flat=dist2_tensor->flat<float>();
            auto idx2_flat=idx2_tensor->flat<int>();
            float * dist1=&(dist1_flat(0));
            int * idx1=&(idx1_flat(0));
            float * dist2=&(dist2_flat(0));
            int * idx2=&(idx2_flat(0));
            NmDistanceKernelLauncher(b,n,xyz1,m,xyz2,dist1,idx1,dist2,idx2);
        }
};
REGISTER_KERNEL_BUILDER(Name("PlaneDistance").Device(DEVICE_GPU), PlaneDistanceGpuOp);
