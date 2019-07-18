#if GOOGLE_CUDA
#define EIGEN_USE_GPU
//#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
  const int batch=512;
  __shared__ float buf[batch*3];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
      float x1=xyz[(i*n+j)*3+0];
      float y1=xyz[(i*n+j)*3+1];
      float z1=xyz[(i*n+j)*3+2];
      // Compare in batches from xyz2
      for (int k2=0;k2<m;k2+=batch){
        // Read into shared buffer
        int end_k=min(m,k2+batch)-k2;
        for (int l=threadIdx.x;l<end_k*3;l+=blockDim.x){
          buf[l]=xyz2[(i*m+k2)*3+l];
        }
        __syncthreads();
        // Compare
        int best_i=0;
        float best=0;
        for (int k=0;k<end_k;k++){
          float x2=buf[k*3+0]-x1;
          float y2=buf[k*3+1]-y1;
          float z2=buf[k*3+2]-z1;
          float d=x2*x2+y2*y2+z2*z2;
          if (k==0 || d<best){
            best=d;
            best_i=k+k2;
          }
        }
        if (k2==0 || best < result[(i*n+j)]){
          result[(i*n+j)]=best;
          result_i[(i*n+j)]=best_i;
        }
        __syncthreads();
      }
    }
  }
}
void NmDistanceKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i){
    NmDistanceKernel<<<dim3(32,16,1),512>>>(b,n,xyz,m,xyz2,result,result_i);
    NmDistanceKernel<<<dim3(32,16,1),512>>>(b,m,xyz2,n,xyz,result2,result2_i);
}

#endif


void read_record() 
{ 
  // File pointer 
  fstream fin; 

  // Open an existing file 
  fin.open("foo.csv", ios::in); 

  // Read the Data from the file 
  // as String Vector 
  vector<string> row; 
  string line, word, temp; 

  while (fin >> temp) { 
    row.clear(); 

    // read an entire row and 
    // store it in a string variable 'line' 
    getline(fin, line); 
    // used for breaking words 
    stringstream s(line); 

    while (getline(s, word, ', ')) { 
      num = stof(word);
      cout << num << "\n";
      row.push_back(num); 
    } 
  } 
} 


int main(){
  read_record();
}

