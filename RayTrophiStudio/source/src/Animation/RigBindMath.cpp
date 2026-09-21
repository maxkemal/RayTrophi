#include "Animation/RigBindMath.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
namespace RigAuthoring {
bool bindAffineInverse(const Matrix4x4& m,Matrix4x4& result) {
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(!std::isfinite(m.m[r][c]))return false;
    for(int c=0;c<4;++c)if(std::fabs(m.m[3][c]-(c==3?1.f:0.f))>1e-6f)return false;
    const double a=m.m[0][0],b=m.m[0][1],c=m.m[0][2],d=m.m[1][0],e=m.m[1][1],f=m.m[1][2],g=m.m[2][0],h=m.m[2][1],i=m.m[2][2];
    const double det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    if(!std::isfinite(det) || det<=0)return false; // No singular/reflected baseline bind.
    const double values[3][3]={{(e*i-f*h)/det,(c*h-b*i)/det,(b*f-c*e)/det},
        {(f*g-d*i)/det,(a*i-c*g)/det,(c*d-a*f)/det},
        {(d*h-e*g)/det,(b*g-a*h)/det,(a*e-b*d)/det}};
    result=Matrix4x4::identity();
    for(int r=0;r<3;++r) {
        double translation=0;
        for(int k=0;k<3;++k){result.m[r][k]=static_cast<float>(values[r][k]);translation-=values[r][k]*m.m[k][3];}
        result.m[r][3]=static_cast<float>(translation);
    }
    for(int r=0;r<4;++r)for(int k=0;k<4;++k)if(!std::isfinite(result.m[r][k]))return false;
    // Reject numerically unstable inverses instead of shipping a guessed identity.
    for(int r=0;r<3;++r)for(int k=0;k<3;++k) {
        double product=0;for(int j=0;j<3;++j)product+=double(result.m[r][j])*m.m[j][k];
        if(std::fabs(product-(r==k?1.:0.))>1e-4)return false;
    }
    return true;
}
double segmentDistanceSquared(const Vec3& p,const SkinSegment& s) {
    const double dx=double(s.end.x)-s.start.x,dy=double(s.end.y)-s.start.y,dz=double(s.end.z)-s.start.z;
    const double px=double(p.x)-s.start.x,py=double(p.y)-s.start.y,pz=double(p.z)-s.start.z;
    const double length=dx*dx+dy*dy+dz*dz;
    const double t=length>0?std::clamp((px*dx+py*dy+pz*dz)/length,0.,1.):0.;
    return (px-t*dx)*(px-t*dx)+(py-t*dy)*(py-t*dy)+(pz-t*dz)*(pz-t*dz);
}
VertexInfluences distanceSkinWeights(const Vec3& p,const std::vector<SkinSegment>& segments,double floor,double& nearest) {
    std::unordered_map<int,double> distances;
    nearest=std::numeric_limits<double>::infinity();
    for(const auto& segment:segments) {
        const double distance=segmentDistanceSquared(p,segment);nearest=std::min(nearest,distance);
        auto found=distances.find(segment.bone);
        if(found==distances.end())distances.emplace(segment.bone,distance);else found->second=std::min(found->second,distance);
    }
    std::vector<std::pair<int,double>> sorted(distances.begin(),distances.end());
    std::sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.second!=b.second?a.second<b.second:a.first<b.first;});
    if(sorted.size()>4)sorted.resize(4);
    VertexInfluences result;
    const double regularizer=std::max(floor*floor,std::numeric_limits<double>::min());
    if(!sorted.empty())for(const auto& bone:sorted)
        result.emplace_back(bone.first,static_cast<float>((sorted.front().second+regularizer)/(bone.second+regularizer)));
    canonicalizeInfluences(result);
    result.erase(std::remove_if(result.begin(),result.end(),[](const auto& w){return w.second<1e-6f;}),result.end());
    canonicalizeInfluences(result);
    nearest=std::sqrt(nearest);return result;
}
}
