// #include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdio.h>

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
                if(k == j)  continue;
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

int main(){
  int b=2;
  int n=2048;
  int m=n;
  float* xyz = new float[b*n*3];
  float* dist = new float[b*n];
  int* idx = new int[b*n];
  read_clouds(xyz);
  nnsearch(b, n, m, xyz, xyz, dist, idx);

  FILE *pfile;
  pfile = fopen("computed.txt","w");
  for (int i=0; i < b*n; i++){
    fprintf(pfile, "%d\n", idx[i]+1);
  }
  fclose(pfile);

  pfile = fopen("computed_dist.txt","w");
  for (int i=0; i < b*n; i++){
    fprintf(pfile, "%f\n", dist[i]);
  }
  fclose(pfile);

  return 0;
}
