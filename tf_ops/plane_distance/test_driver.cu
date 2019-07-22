#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int32_t

// #include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <list>
#include <unistd.h>

using namespace std;

#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define MAX(x,y) ((x)>(y)?(x):(y))

const int NUM_NBRS = 10;

///////////////////////////////////////////////////////

__host__ __device__ static double PYTHAG(double a, double b)
{
    double at = fabs(a), bt = fabs(b), ct, result;

    if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
    else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
    else result = 0.0;
    return(result);
}

///////////////////////////////////////////////////////

__host__ __device__
int dsvd(float a[][NUM_NBRS], int m, int n, float w[3], float v[][3]){
    int flag, i, its, j, jj, k, l, nm;
    double c, f, h, s, x, y, z;
    double anorm = 0.0, g = 0.0, scale = 0.0;
    double *rv1;
  
    if (m < n) 
    {
        // fprintf(stderr, "#rows must be > #cols \n");
        return(0);
    }
  
    rv1 = (double *)malloc((unsigned int) n*sizeof(double));

/* Householder reduction to bidiagonal form */
    for (i = 0; i < n; i++) 
    {
        /* left-hand reduction */
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < m) 
        {
            for (k = i; k < m; k++) 
                scale += fabs((double)a[k][i]);
            if (scale) 
            {
                for (k = i; k < m; k++) 
                {
                    a[k][i] = (float)((double)a[k][i]/scale);
                    s += ((double)a[k][i] * (double)a[k][i]);
                }
                f = (double)a[i][i];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[i][i] = (float)(f - g);
                if (i != n - 1) 
                {
                    for (j = l; j < n; j++) 
                    {
                        for (s = 0.0, k = i; k < m; k++) 
                            s += ((double)a[k][i] * (double)a[k][j]);
                        f = s / h;
                        for (k = i; k < m; k++) 
                            a[k][j] += (float)(f * (double)a[k][i]);
                    }
                }
                for (k = i; k < m; k++) 
                    a[k][i] = (float)((double)a[k][i]*scale);
            }
        }
        w[i] = (float)(scale * g);
    
        /* right-hand reduction */
        g = s = scale = 0.0;
        if (i < m && i != n - 1) 
        {
            for (k = l; k < n; k++) 
                scale += fabs((double)a[i][k]);
            if (scale) 
            {
                for (k = l; k < n; k++) 
                {
                    a[i][k] = (float)((double)a[i][k]/scale);
                    s += ((double)a[i][k] * (double)a[i][k]);
                }
                f = (double)a[i][l];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[i][l] = (float)(f - g);
                for (k = l; k < n; k++) 
                    rv1[k] = (double)a[i][k] / h;
                if (i != m - 1) 
                {
                    for (j = l; j < m; j++) 
                    {
                        for (s = 0.0, k = l; k < n; k++) 
                            s += ((double)a[j][k] * (double)a[i][k]);
                        for (k = l; k < n; k++) 
                            a[j][k] += (float)(s * rv1[k]);
                    }
                }
                for (k = l; k < n; k++) 
                    a[i][k] = (float)((double)a[i][k]*scale);
            }
        }
        anorm = MAX(anorm, (fabs((double)w[i]) + fabs(rv1[i])));
    }
  
    /* accumulate the right-hand transformation */
    for (i = n - 1; i >= 0; i--) 
    {
        if (i < n - 1) 
        {
            if (g) 
            {
                for (j = l; j < n; j++)
                    v[j][i] = (float)(((double)a[i][j] / (double)a[i][l]) / g);
                    /* double division to avoid underflow */
                for (j = l; j < n; j++) 
                {
                    for (s = 0.0, k = l; k < n; k++) 
                        s += ((double)a[i][k] * (double)v[k][j]);
                    for (k = l; k < n; k++) 
                        v[k][j] += (float)(s * (double)v[k][i]);
                }
            }
            for (j = l; j < n; j++) 
                v[i][j] = v[j][i] = 0.0;
        }
        v[i][i] = 1.0;
        g = rv1[i];
        l = i;
    }
  
    /* accumulate the left-hand transformation */
    for (i = n - 1; i >= 0; i--) 
    {
        l = i + 1;
        g = (double)w[i];
        if (i < n - 1) 
            for (j = l; j < n; j++) 
                a[i][j] = 0.0;
        if (g) 
        {
            g = 1.0 / g;
            if (i != n - 1) 
            {
                for (j = l; j < n; j++) 
                {
                    for (s = 0.0, k = l; k < m; k++) 
                        s += ((double)a[k][i] * (double)a[k][j]);
                    f = (s / (double)a[i][i]) * g;
                    for (k = i; k < m; k++) 
                        a[k][j] += (float)(f * (double)a[k][i]);
                }
            }
            for (j = i; j < m; j++) 
                a[j][i] = (float)((double)a[j][i]*g);
        }
        else 
        {
            for (j = i; j < m; j++) 
                a[j][i] = 0.0;
        }
        ++a[i][i];
    }

    /* diagonalize the bidiagonal form */
    for (k = n - 1; k >= 0; k--) 
    {                             /* loop over singular values */
        for (its = 0; its < 30; its++) 
        {                         /* loop over allowed iterations */
            flag = 1;
            for (l = k; l >= 0; l--) 
            {                     /* test for splitting */
                nm = l - 1;
                if (fabs(rv1[l]) + anorm == anorm) 
                {
                    flag = 0;
                    break;
                }
                if (fabs((double)w[nm]) + anorm == anorm) 
                    break;
            }
            if (flag) 
            {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++) 
                {
                    f = s * rv1[i];
                    if (fabs(f) + anorm != anorm) 
                    {
                        g = (double)w[i];
                        h = PYTHAG(f, g);
                        w[i] = (float)h; 
                        h = 1.0 / h;
                        c = g * h;
                        s = (- f * h);
                        for (j = 0; j < m; j++) 
                        {
                            y = (double)a[j][nm];
                            z = (double)a[j][i];
                            a[j][nm] = (float)(y * c + z * s);
                            a[j][i] = (float)(z * c - y * s);
                        }
                    }
                }
            }
            z = (double)w[k];
            if (l == k) 
            {                  /* convergence */
                if (z < 0.0) 
                {              /* make singular value nonnegative */
                    w[k] = (float)(-z);
                    for (j = 0; j < n; j++) 
                        v[j][k] = (-v[j][k]);
                }
                break;
            }
            if (its >= 30) {
                free((void*) rv1);
                // fprintf(stderr, "No convergence after 30,000! iterations \n");
                return(0);
            }
    
            /* shift from bottom 2 x 2 minor */
            x = (double)w[l];
            nm = k - 1;
            y = (double)w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = PYTHAG(f, 1.0);
            f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
          
            /* next QR transformation */
            c = s = 1.0;
            for (j = l; j <= nm; j++) 
            {
                i = j + 1;
                g = rv1[i];
                y = (double)w[i];
                h = s * g;
                g = c * g;
                z = PYTHAG(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++) 
                {
                    x = (double)v[jj][j];
                    z = (double)v[jj][i];
                    v[jj][j] = (float)(x * c + z * s);
                    v[jj][i] = (float)(z * c - x * s);
                }
                z = PYTHAG(f, h);
                w[j] = (float)z;
                if (z) 
                {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++) 
                {
                    y = (double)a[jj][j];
                    z = (double)a[jj][i];
                    a[jj][j] = (float)(y * c + z * s);
                    a[jj][i] = (float)(z * c - y * s);
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = (float)x;
        }
    }
    free((void*) rv1);
    return(1);
}

void read_clouds(float * result){
  // File pointer 
  fstream fin; 

  // Open an existing file 
  fin.open("clouds2.txt", ios::in); 

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
      for (int j=0;j<n;j++){
        float x1=xyz[(i*n+j)*3+0];
        float y1=xyz[(i*n+j)*3+1];
        float z1=xyz[(i*n+j)*3+2];
        // priority_queue<pair<float, int> > nn_dist;
        // pair<float, int> max_dist;
        float nn_dist [num_nbrs];
        int nn_idx [num_nbrs];
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
          if(insert_idx < num_nbrs){   // (LOOP) counter value less than queue capacity, so append
          // if((j < num_nbrs && k <= num_nbrs) || (k < num_nbrs)){  // QUEUE
            // nn_dist.push(make_pair(d,k));
            // max_dist = nn_dist.top();
            nn_dist[insert_idx] = d;
            nn_idx[insert_idx] = k;
            if(d > nn_dist[max_idx]){
              max_idx = insert_idx;
            }
            insert_idx++;
          }
          else if(d < nn_dist[max_idx]){ // LOOP
          // else if(d < max_dist.first){ // QUEUE
            // nn_dist.pop();
            // nn_dist.push(make_pair(d,k));
            // max_dist = nn_dist.top();
            if(d < nn_dist[max_idx]){
              nn_dist[max_idx] = d;
              nn_idx[max_idx] = k;
              // Find new max_idx
              max_idx = 0;
              for(int l=1; l < num_nbrs; l++){
                if(nn_dist[l] > nn_dist[max_idx]){
                  max_idx = l;
                }
              }
            }
          }
        }

        // LOOP - Sort nn_idx by nn_dist
        // Bubble sort nn_idx
        for(int k=0; k<num_nbrs; k++){
          for(int l=num_nbrs-1; l>k; l--){
            if(nn_dist[l] < nn_dist[l-1]){
              // Swap in BOTH nn_idx, nn_dist
              int tmp = nn_idx[l-1];
              nn_idx[l-1] = nn_idx[l];
              nn_idx[l] = tmp;
              float tmpd = nn_dist[l-1];
              nn_dist[l-1] = nn_dist[l];
              nn_dist[l] = tmpd;
            }
          }
        }
        for(int k=0; k<num_nbrs; k++){
          idx[(i*n+j)*num_nbrs+k] = nn_idx[k];
        }
        // for(int k=num_nbrs-1; k>=0; k--){
          // int l = nn_dist.top().second;
          // nn_dist.pop();
          // idx[(i*n+j)*num_nbrs+k] = l;
        // }
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

static void simpleplanedistance(int b, int n, const float *xyz, float *dist, float *offset, float *normal){
  const int num_nbrs = 10;
  for (int i=0; i<b; i+=1){
    for (int j=0; j<n; j+=1){
      float x1=xyz[(i*n+j)*3+0];
      float y1=xyz[(i*n+j)*3+1];
      float z1=xyz[(i*n+j)*3+2];
      // Queue via loop
      float nn_dist [num_nbrs];
      int nn_idx [num_nbrs];
      int insert_idx = 0;
      int max_idx = 0;
      for (int k=0; k < n; k+=1){
        // Compare
        if(j == k)
          continue;
        float x2=xyz[(i*n+k)*3+0]-x1;
        float y2=xyz[(i*n+k)*3+1]-y1;
        float z2=xyz[(i*n+k)*3+2]-z1;
        float d=x2*x2+y2*y2+z2*z2;
        if(insert_idx < num_nbrs){
          nn_dist[insert_idx] = d;
          nn_idx[insert_idx] = k;
          if(d > nn_dist[max_idx]){
            max_idx = insert_idx;
          }
          insert_idx++;
        }
        else if(d < nn_dist[max_idx]){
          nn_dist[max_idx] = d;
          nn_idx[max_idx] = k;
          // Find new max_idx
          max_idx = 0;
          for(int l=1; l < num_nbrs; l++){
            if(nn_dist[l] > nn_dist[max_idx]){
              max_idx = l;
            }
          }
        }
      }

      // Store k nearest nbr indices
      // for(int k=0; k<num_nbrs; k++){
      //   result_i[(i*n+j)*num_nbrs+k] = nn_idx[k];
      // }

      // Init matrices to hold SVD results
      float **a = new float *[num_nbrs];
      for(int k=0; k<num_nbrs; k++)
        a[k] = new float[num_nbrs];
      float **v = new float *[3];
      for(int k=0; k<3; k++)
        v[k] = new float[3];
      float w[3];

      // Copy over nearest nbrs to a
      for(int k=0; k<num_nbrs; k++)
        for(int l=0; l<3; l++)
          a[k][l] = xyz[(i*n+nn_idx[k])*3+l];

      // calculate centroid
      float centroidx = 0;
      float centroidy = 0;
      float centroidz = 0;
      for(int k=0; k<num_nbrs; k++){
        centroidx += a[k][0];
        centroidy += a[k][1];
        centroidz += a[k][2];
      }
      centroidx /= num_nbrs;
      centroidy /= num_nbrs;
      centroidz /= num_nbrs;

      // subtract centroid
      for(int k=0; k<num_nbrs; k++){
        a[k][0] -= centroidx;
        a[k][1] -= centroidy;
        a[k][2] -= centroidz;
      }

      // // Calculate SVD
      // dsvd(a, num_nbrs, 3, w, v);
      // // Find smallest singular value
      // int minidx = (w[0] < w[1]) ? 0 : 1 ;
      // minidx = (w[minidx] < w[2]) ? minidx : 2 ;

      // float nrm[3];
      // nrm[0] = v[0][minidx];
      // nrm[1] = v[1][minidx];
      // nrm[2] = v[2][minidx];
      // normal[(i*n+j)*3+0] = nrm[0];
      // normal[(i*n+j)*3+1] = nrm[1];
      // normal[(i*n+j)*3+2] = nrm[2];

      // // Calculate offset
      // float o = (x1-centroidx)*nrm[0] + (y1-centroidy)*nrm[1] + (z1-centroidz)*nrm[2];
      // offset[i*n+j] = o;
      // dist[i*n+j] = 0.5*o*o;
    }
  }
}

__global__ void PlaneDistanceKernel(int b, int n, const float *xyz, float *dist, float *offset, float *normal){
  const int batch=512;
  // const int num_nbrs = 10;
  __shared__ float buf[batch*3];
  if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
    printf("Hi\t");
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
      float x1=xyz[(i*n+j)*3+0];
      float y1=xyz[(i*n+j)*3+1];
      float z1=xyz[(i*n+j)*3+2];
      // Queue via loop
      float nn_dist [NUM_NBRS];
      int nn_idx [NUM_NBRS];
      int insert_idx = 0;
      int max_idx = 0;
      for (int k2=0; k2 < n; k2+=batch){
        // Read into shared buffer
        int end_k=min(n,k2+batch)-k2;
        for (int l=threadIdx.x;l<end_k*3;l+=blockDim.x){
          buf[l]=xyz[(i*n+k2)*3+l];
        }
        __syncthreads();
        // Compare
        for (int k=0;k<end_k;k++){
          if(j == k+k2)
            continue;
          float x2=buf[k*3+0]-x1;
          float y2=buf[k*3+1]-y1;
          float z2=buf[k*3+2]-z1;
          float d=x2*x2+y2*y2+z2*z2;
          if(insert_idx < NUM_NBRS){
            nn_dist[insert_idx] = d;
            nn_idx[insert_idx] = k+k2;
            if(d > nn_dist[max_idx]){
              max_idx = insert_idx;
            }
            insert_idx++;
          }
          else if(d < nn_dist[max_idx]){
            nn_dist[max_idx] = d;
            nn_idx[max_idx] = k+k2;
            // Find new max_idx
            max_idx = 0;
            for(int l=1; l < NUM_NBRS; l++){
              if(nn_dist[l] > nn_dist[max_idx]){
                max_idx = l;
              }
            }
          }
        }
        __syncthreads();
      }

      // Store k nearest nbr indices
      // for(int k=0; k<NUM_NBRS; k++){
      //   result_i[(i*n+j)*NUM_NBRS+k] = nn_idx[k];
      // }

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
void PlaneDistanceKernelLauncher(int b,int n,const float * xyz,float * dist,float * offset, float *normal){
    PlaneDistanceKernel<<<dim3(32,16,1),512>>>(b,n,xyz,dist,offset,normal);
}
__global__ void PlaneDistanceGradKernel(int b,int n,const float *dist,const float *normals,float *grad){
    for (int i=blockIdx.x;i<b;i+=gridDim.x){
      for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
        float g=dist[i*n+j];
        float nx=normals[(i*n+j)*3+0];
        float ny=normals[(i*n+j)*3+1];
        float nz=normals[(i*n+j)*3+2];
        grad[(i*n+j)*3+0] = g*nx;
        grad[(i*n+j)*3+1] = g*ny;
        grad[(i*n+j)*3+2] = g*nz;
      }
    }
}
void PlaneDistanceGradKernelLauncher(int b,int n,const float *dist,const float *normals,float *grad){
    cudaMemset(grad,0,b*n*3*4);
    PlaneDistanceGradKernel<<<dim3(1,16,1),256>>>(b,n,dist,normals,grad);
}

int main(){
  int b=32;
  int n=4096;
  int m=n;
  float *xyz;
  int *idx;
  cudaMallocManaged(&xyz, b*n*3*sizeof(float));
  // Read point cloud
  read_clouds(xyz);

  for(int foo=0; foo<10; foo++){
    printf("%d:\t", foo);
    // usleep(10000);
    float *dist, *offset, *plane, *grad;
    cudaError_t err = cudaMallocManaged(&dist, b*n*sizeof(float));
    if(dist == nullptr)
      printf("dist null %d\t", err);
    err = cudaMallocManaged(&offset, b*n*sizeof(float));
    if(offset == nullptr)
      printf("offset null %d\t", err);
    err = cudaMallocManaged(&plane, b*n*3*sizeof(float));
    if(plane == nullptr)
      printf("plane null %d\t", err);
    err = cudaMallocManaged(&grad, b*n*3*sizeof(float));
    if(grad == nullptr)
      printf("grad null  %d\t", err);
    printf("Malloc,%f\t", xyz[12287]);
    // usleep(10000);

    // CPU
    // for(int i=0; i<100; i++){
    //   knearestnbr(b, n, xyz, dist, idx);
    // }
    plane[12287] = 0;
    printf("NBefore: %.6f\t", plane[12287]);
    // simpleplanedistance(b, n, xyz, dist, offset, plane);
    // GPU
    PlaneDistanceKernelLauncher(b, n, xyz, dist, offset, plane);
    cudaDeviceSynchronize();
    printf("NAfter: %.6f\t", plane[12287]);
    // printf("Normal: %.6f\t", plane[0]);
    // usleep(10000);
    PlaneDistanceGradKernelLauncher(b, n, offset, plane, grad);
    cudaDeviceSynchronize();
    printf("Grad: %.6f\t", grad[0]);
    // usleep(10000);

    cudaFree(dist);
    cudaFree(offset);
    cudaFree(plane);
    cudaFree(grad);
    printf("Free\n");
    // usleep(10000);
  }

  // FILE *pfile;
  // // pfile = fopen("ans_knearestnbr_gpu.txt","w");
  // // for (int i=0; i < b*n; i+=1){
  // //   for (int k=0; k < num_nbrs; k++){
  // //     fprintf(pfile, "%d ", idx[i*num_nbrs + k]+1);
  // //   }
  // //   fprintf(pfile, "\n");
  // // }
  // // fclose(pfile);

  // pfile = fopen("ans_planedist_gpu.txt","w");
  // for (int i=0; i < b*n; i+=1){
  //   fprintf(pfile, "%f\n", dist[i]);
  // }
  // fclose(pfile);

  // pfile = fopen("ans_planenormals_gpu.txt","w");
  // for (int i=0; i < b*n; i+=1){
  //   fprintf(pfile, "%.3f %.3f %.3f\n", plane[i*3+0], plane[i*3+1], plane[i*3+2]);
  // }
  // fclose(pfile);

  // pfile = fopen("ans_grad_gpu.txt","w");
  // for (int i=0; i < b*n; i+=1){
  //   fprintf(pfile, "%.5f %.5f %.5f\n", grad[i*3+0], grad[i*3+1], grad[i*3+2]);
  // }
  // fclose(pfile);

  cudaFree(xyz);

  return 0;
}
