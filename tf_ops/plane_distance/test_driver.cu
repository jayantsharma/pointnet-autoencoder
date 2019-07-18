// #include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdio.h>
#include <queue>

using namespace std;
void read_clouds(float * result) 
{ 
  // File pointer 
  fstream fin; 

  // Open an existing file 
  fin.open("clouds.txt", ios::in); 

  // Read the Data from the file 
  // as String Vector 
  string line, word, temp; 

  int i=0;
  while (fin >> line) { 
    // used for breaking words 
    stringstream s(line); 

    while (getline(s, word, ',')) { 
      float num = stof(word);
      result[i] = num; 
      i++;
    } 
  } 
} 

static void nnsearch(int b,int n,int m,const float * xyz1,const float * xyz2,float * dist,int * idx){
    for (int i=0;i<b;i++){
        for (int j=0;j<n;j++){
            float x1=xyz1[(i*n+j)*3+0];
            float y1=xyz1[(i*n+j)*3+1];
            float z1=xyz1[(i*n+j)*3+2];
            double best=100;
            int besti=0;
            for (int k=0;k<m;k++){
                // if(k == j)  continue;
                float x2=xyz2[(i*m+k)*3+0]-x1;
                float y2=xyz2[(i*m+k)*3+1]-y1;
                float z2=xyz2[(i*m+k)*3+2]-z1;
                double d=x2*x2+y2*y2+z2*z2;
                if (d<best){
                    best=d;
                    besti=k;
                }
            }
            dist[i*n+j]=best;
            idx[i*n+j]=besti;
        }
    }
}

static void knearestnbr(int b,int n,const float * xyz,float * dist, int *idx){
    for (int i=0;i<b;i++){
      int num_nbrs = 10;
      // priority_queue<float> nn_dist;
      // float nn_dist [num_nbrs];
      // float nn_idx [num_nbrs];
      // float max_idx;
      for (int j=0;j<n;j++){
        float x1=xyz[(i*n+j)*3+0];
        float y1=xyz[(i*n+j)*3+1];
        float z1=xyz[(i*n+j)*3+2];
        priority_queue<pair<float, int> > nn_dist;
        pair<float, int> max_dist;
        for (int k=0;k<n;k++){
          if (k==j){
            continue;
          }
          float x2=xyz[(i*n+k)*3+0]-x1;
          float y2=xyz[(i*n+k)*3+1]-y1;
          float z2=xyz[(i*n+k)*3+2]-z1;
          double d=x2*x2+y2*y2+z2*z2;
          // cout << "Idx: " << k << ", Dist: " << d << "\n";
          if((j < num_nbrs && k <= num_nbrs) || (k < num_nbrs)){
            // cout << "Inserted\n";
            nn_dist.push(make_pair(d,k));
            max_dist = nn_dist.top();
            // nn_dist[k] = d;
            // nn_idx[k] = k;
            // if(k == 0 || d > nn_dist[max_idx]){
            //   max_idx = k;
            // }
          }
          else if(d < max_dist.first){
            // cout << "Replaced pt: " << max_dist.second << "with " << k << ", dist: " << d << "\n";
            nn_dist.pop();
            nn_dist.push(make_pair(d,k));
            max_dist = nn_dist.top();
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

        for(int k=num_nbrs-1; k>=0; k--){
          int l = nn_dist.top().second;
          nn_dist.pop();
          idx[(i*n+j)*num_nbrs+k] = l;
        }
        // // Find best-fit plane now
        // vector<Vector3f> points;
        // cout << "Point " << j << "\n";
        // while(!nn_dist.empty()){
        //   int k = nn_dist.top().second;
        //   nn_dist.pop();
        //   Vector3f point;
        //   point(0) = xyz[(i*n+k)*3+0];
        //   point(1) = xyz[(i*n+k)*3+1];
        //   point(2) = xyz[(i*n+k)*3+2];
        //   std::cout << "Nbr: " << k << std::endl;
        //   points.push_back(point);
        // }
        // cout << "\n";
        // pair<Vector3f, Vector3f> plane_pair = best_plane_from_points(points);
        // Vector3f centroid = plane_pair.first;
        // Vector3f plane_normal = plane_pair.second;
        // float dist_from_plane = abs((x1-centroid(0))*plane_normal(0) + (y1-centroid(1))*plane_normal(1) + (z1-centroid(2))*plane_normal(2));
        // dist[i*n+j]=dist_from_plane;
        // // idx[i*n+j]=besti;
      }
    }
}

__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
    const int batch=512;
    __shared__ float buf[batch*3];
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int k2=0;k2<m;k2+=batch){
            int end_k=min(m,k2+batch)-k2;
            for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
                buf[j]=xyz2[(i*m+k2)*3+j];
            }
            __syncthreads();
            for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
                float x1=xyz[(i*n+j)*3+0];
                float y1=xyz[(i*n+j)*3+1];
                float z1=xyz[(i*n+j)*3+2];
                int best_i=0;
                float best=0;
                int end_ka=end_k-(end_k&3);
                if (end_ka==batch){
                    for (int k=0;k<batch;k+=4){
                        {
                            float x2=buf[k*3+0]-x1;
                            float y2=buf[k*3+1]-y1;
                            float z2=buf[k*3+2]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (k==0 || d<best){
                                best=d;
                                best_i=k+k2;
                            }
                        }
                        {
                            float x2=buf[k*3+3]-x1;
                            float y2=buf[k*3+4]-y1;
                            float z2=buf[k*3+5]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+1;
                            }
                        }
                        {
                            float x2=buf[k*3+6]-x1;
                            float y2=buf[k*3+7]-y1;
                            float z2=buf[k*3+8]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+2;
                            }
                        }
                        {
                            float x2=buf[k*3+9]-x1;
                            float y2=buf[k*3+10]-y1;
                            float z2=buf[k*3+11]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+3;
                            }
                        }
                    }
                }else{
                    for (int k=0;k<end_ka;k+=4){
                        {
                            float x2=buf[k*3+0]-x1;
                            float y2=buf[k*3+1]-y1;
                            float z2=buf[k*3+2]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (k==0 || d<best){
                                best=d;
                                best_i=k+k2;
                            }
                        }
                        {
                            float x2=buf[k*3+3]-x1;
                            float y2=buf[k*3+4]-y1;
                            float z2=buf[k*3+5]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+1;
                            }
                        }
                        {
                            float x2=buf[k*3+6]-x1;
                            float y2=buf[k*3+7]-y1;
                            float z2=buf[k*3+8]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+2;
                            }
                        }
                        {
                            float x2=buf[k*3+9]-x1;
                            float y2=buf[k*3+10]-y1;
                            float z2=buf[k*3+11]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if (d<best){
                                best=d;
                                best_i=k+k2+3;
                            }
                        }
                    }
                }
                for (int k=end_ka;k<end_k;k++){
                    float x2=buf[k*3+0]-x1;
                    float y2=buf[k*3+1]-y1;
                    float z2=buf[k*3+2]-z1;
                    float d=x2*x2+y2*y2+z2*z2;
                    if (k==0 || d<best){
                        best=d;
                        best_i=k+k2;
                    }
                }
                if (k2==0 || result[(i*n+j)]>best){
                    result[(i*n+j)]=best;
                    result_i[(i*n+j)]=best_i;
                }
            }
            __syncthreads();
        }
    }
}
__global__ void JNotEqKNmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
    const int batch=512;
    __shared__ float buf[batch*3];
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int k2=0;k2<m;k2+=batch){
            int end_k=min(m,k2+batch)-k2;
            for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
                buf[j]=xyz2[(i*m+k2)*3+j];
            }
            __syncthreads();
            for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
                float x1=xyz[(i*n+j)*3+0];
                float y1=xyz[(i*n+j)*3+1];
                float z1=xyz[(i*n+j)*3+2];
                int best_i=-10;
                float best=100;
                int end_ka=end_k-(end_k&3);
                if (end_ka==batch){
                    for (int k=0;k<batch;k+=4){
                        {
                            float x2=buf[k*3+0]-x1;
                            float y2=buf[k*3+1]-y1;
                            float z2=buf[k*3+2]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if ((k+k2!=j) && (d<best)){
                                best=d;
                                best_i=k+k2;
                            }
                        }
                        {
                            float x2=buf[k*3+3]-x1;
                            float y2=buf[k*3+4]-y1;
                            float z2=buf[k*3+5]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if ((k+k2+1 != j) && (d<best)){
                                best=d;
                                best_i=k+k2+1;
                            }
                        }
                        {
                            float x2=buf[k*3+6]-x1;
                            float y2=buf[k*3+7]-y1;
                            float z2=buf[k*3+8]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if ((k+k2+2 != j) && (d<best)){
                                best=d;
                                best_i=k+k2+2;
                            }
                        }
                        {
                            float x2=buf[k*3+9]-x1;
                            float y2=buf[k*3+10]-y1;
                            float z2=buf[k*3+11]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if ((k+k2+3 != j) && (d<best)){
                                best=d;
                                best_i=k+k2+3;
                            }
                        }
                    }
                }else{
                    for (int k=0;k<end_ka;k+=4){
                        {
                            float x2=buf[k*3+0]-x1;
                            float y2=buf[k*3+1]-y1;
                            float z2=buf[k*3+2]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if ((k+k2 != j) && (d<best)){
                                best=d;
                                best_i=k+k2;
                            }
                        }
                        {
                            float x2=buf[k*3+3]-x1;
                            float y2=buf[k*3+4]-y1;
                            float z2=buf[k*3+5]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if ((k+k2+1 != j) && (d<best)){
                                best=d;
                                best_i=k+k2+1;
                            }
                        }
                        {
                            float x2=buf[k*3+6]-x1;
                            float y2=buf[k*3+7]-y1;
                            float z2=buf[k*3+8]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if ((k+k2+2 != j) && (d<best)){
                                best=d;
                                best_i=k+k2+2;
                            }
                        }
                        {
                            float x2=buf[k*3+9]-x1;
                            float y2=buf[k*3+10]-y1;
                            float z2=buf[k*3+11]-z1;
                            float d=x2*x2+y2*y2+z2*z2;
                            if ((k+k2+3 != j) && (d<best)){
                                best=d;
                                best_i=k+k2+3;
                            }
                        }
                    }
                }
                for (int k=end_ka;k<end_k;k++){
                    float x2=buf[k*3+0]-x1;
                    float y2=buf[k*3+1]-y1;
                    float z2=buf[k*3+2]-z1;
                    float d=x2*x2+y2*y2+z2*z2;
                    if ((k+k2 != j) && (d<best)){
                        best=d;
                        best_i=k+k2;
                    }
                }
                if (k2==0 || result[(i*n+j)]>best){
                    result[(i*n+j)]=best;
                    result_i[(i*n+j)]=best_i;
                }
            }
            __syncthreads();
        }
    }
}
__global__ void MyNmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
  const int batch=512;
  __shared__ float buf[batch*3];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    // Compare in batches from xyz2
    for (int k2=0;k2<m;k2+=batch){
      // Read into shared buffer
      int end_k=min(m,k2+batch)-k2;
      for (int l=threadIdx.x;l<end_k*3;l+=blockDim.x){
        buf[l]=xyz2[(i*m+k2)*3+l];
      }
      __syncthreads();
      for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
        float x1=xyz[(i*n+j)*3+0];
        float y1=xyz[(i*n+j)*3+1];
        float z1=xyz[(i*n+j)*3+2];
        // Compare
        int best_i=0;
        float best=100;
        for (int k=0;k<end_k;k++){
          float x2=buf[k*3+0]-x1;
          float y2=buf[k*3+1]-y1;
          float z2=buf[k*3+2]-z1;
          float d=x2*x2+y2*y2+z2*z2;
          if (j!=k+k2 && d<best){
            best=d;
            best_i=k+k2;
          }
        }
        if (k2==0 || best < result[(i*n+j)]){
          result[(i*n+j)]=best;
          result_i[(i*n+j)]=best_i;
        }
      }
      __syncthreads();
    }
  }
}
void NmDistanceKernelLauncher(int b,int n,int m,const float * xyz,const float * xyz2,float * result,int * result_i){
    // NmDistanceKernel<<<dim3(32,16,1),512>>>(b,n,xyz,m,xyz2,result,result_i);
    JNotEqKNmDistanceKernel<<<dim3(32,16,1),512>>>(b,n,xyz,m,xyz2,result,result_i);
    // MyNmDistanceKernel<<<dim3(32,16,1),512>>>(b,n,xyz,m,xyz2,result,result_i);
}

int main(){
  int b=2;
  int n=2048;
  int m=n;
  // float* xyz = new float[b*n*3];
  // float* dist = new float[b*n];
  // int* idx = new int[b*n];
  float *xyz, *dist;
  int *idx;
  cudaMallocManaged(&xyz, b*n*3*sizeof(float));
  cudaMallocManaged(&dist, b*n*sizeof(float));
  cudaMallocManaged(&idx, b*n*sizeof(int));
  
  // Read point cloud
  read_clouds(xyz);

  // CPU
  // nnsearch(b, n, m, xyz, xyz, dist, idx);
  // GPU
  for(int i=0; i<10000; i++){
    NmDistanceKernelLauncher(b, n, m, xyz, xyz, dist, idx);
    cudaDeviceSynchronize();
  }

  FILE *pfile;
  pfile = fopen("computed_gpu.txt","w");
  for (int i=0; i < b*n; i+=1){
    fprintf(pfile, "%d\n", idx[i]+1);
  }
  fclose(pfile);

  pfile = fopen("computed_dist_gpu.txt","w");
  for (int i=0; i < b*n; i+=1){
    fprintf(pfile, "%f\n", dist[i]);
  }
  fclose(pfile);

  cudaFree(xyz);
  cudaFree(dist);
  cudaFree(idx);

  return 0;
}
