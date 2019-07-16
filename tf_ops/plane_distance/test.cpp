#include <queue>
#include <Eigen/Dense>
#include <bits/stdc++.h>
using namespace std;
using namespace Eigen;

template<class Vector3>
pair<Vector3, Vector3> best_plane_from_points(const vector<Vector3> & c){
  // copy coordinates to  matrix in Eigen format
  size_t num_atoms = c.size();
  Matrix<float, Dynamic, Dynamic> coord(3, num_atoms);
  for (size_t i = 0; i < num_atoms; ++i) coord.col(i) = c[i];

  // calculate centroid
  Vector3 centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

  // subtract centroid
  coord.row(0).array() -= centroid(0); coord.row(1).array() -= centroid(1); coord.row(2).array() -= centroid(2);

  // we only need the left-singular matrix here
  //  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
  auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  Vector3 plane_normal = svd.matrixU().rightCols<1>();
  return make_pair(centroid, plane_normal);
}

static void planesearch(int b,int n,const float * xyz,float * dist){
    for (int i=0;i<b;i++){
      int num_nbrs = 3;
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
          cout << "Idx: " << k << ", Dist: " << d << "\n";
          if((j < num_nbrs && k <= num_nbrs) || (k < num_nbrs)){
            cout << "Inserted\n";
            nn_dist.push(make_pair(d,k));
            max_dist = nn_dist.top();
            // nn_dist[k] = d;
            // nn_idx[k] = k;
            // if(k == 0 || d > nn_dist[max_idx]){
            //   max_idx = k;
            // }
          }
          else if(d < max_dist.first){
            cout << "Replaced pt: " << max_dist.second << "with " << k << ", dist: " << d << "\n";
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

        // Find best-fit plane now
        vector<Vector3f> points;
        cout << "Point " << j << "\n";
        while(!nn_dist.empty()){
          int k = nn_dist.top().second;
          nn_dist.pop();
          Vector3f point;
          point(0) = xyz[(i*n+k)*3+0];
          point(1) = xyz[(i*n+k)*3+1];
          point(2) = xyz[(i*n+k)*3+2];
          std::cout << "Nbr: " << k << std::endl;
          points.push_back(point);
        }
        cout << "\n";
        pair<Vector3f, Vector3f> plane_pair = best_plane_from_points(points);
        Vector3f centroid = plane_pair.first;
        Vector3f plane_normal = plane_pair.second;
        float dist_from_plane = abs((x1-centroid(0))*plane_normal(0) + (y1-centroid(1))*plane_normal(1) + (z1-centroid(2))*plane_normal(2));
        dist[i*n+j]=dist_from_plane;
        // idx[i*n+j]=besti;
      }
    }
}

int main(){
  float xyz[] = {1.2187, 0.7219, 0.3273,
                 0.5155, 0.2572, 0.7773,
                 1.9986, 0.8065, -0.6539,
                 3.0743, -0.5451, -0.2765,
                 4., 1., 1.,
                 10., 0., 0.,
                 3.7704, 1.0210, 0.1445,
                 2.5171, 1.3666, 1.5375,
                 1.9513, 2.0431, 0.7668,
                 2.8391, 1.1393, 1.1670};
  float dist[10];
  planesearch(2, 5, xyz, dist);

  std::cout << "The contents of dist are:";
  for (auto x:dist) std::cout << ' ' << x;
  std::cout << '\n';

  return 0;
}
