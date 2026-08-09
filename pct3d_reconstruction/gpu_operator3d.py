"""CuPy/CUDA Schulte-MLP projector with paired trilinear adjoint."""

from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
struct Mat2 { double a,b,c,d; };
__device__ __forceinline__ Mat2 add2(Mat2 x,Mat2 y){return{x.a+y.a,x.b+y.b,x.c+y.c,x.d+y.d};}
__device__ __forceinline__ Mat2 mul2(Mat2 x,Mat2 y){
  return{x.a*y.a+x.b*y.c,x.a*y.b+x.b*y.d,x.c*y.a+x.d*y.c,x.c*y.b+x.d*y.d};
}
__device__ __forceinline__ Mat2 inv2(Mat2 x){
  double det=x.a*x.d-x.b*x.c;if(fabs(det)<1e-30)det=copysign(1e-30,det+1e-30);
  return{x.d/det,-x.b/det,-x.c/det,x.a/det};
}
__device__ __forceinline__ void integrals(double u,double &th,double &tth,double &t){
  const double c[6]={7.444724e-6,5.463937e-8,-9.986645e-10,2.026409e-11,-1.420501e-13,3.899100e-16};
  th=tth=t=0.;double p=u;
  #pragma unroll
  for(int k=0;k<6;++k){th+=c[k]*p/(k+1.);tth+=c[k]*p*u/(k+2.);t+=c[k]*p*u*u/(k+3.);p*=u;}
}
__device__ __forceinline__ double scatter(double distance){
  distance=fmax(distance,1e-3);double q=1.+0.038*log(distance/361.);
  return(13.6*13.6/361.)*q*q;
}
__device__ __forceinline__ bool cylinder_forward(
    const double *p,const double *d,double radius,double half_y,double &enter,double &leave){
  double n=sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);if(n<=1e-12)return false;
  double ux=d[0]/n,uy=d[1]/n,uz=d[2]/n;
  double a=ux*ux+uz*uz,b=2.*(p[0]*ux+p[2]*uz),c=p[0]*p[0]+p[2]*p[2]-radius*radius;
  double disc=b*b-4.*a*c;if(a<=1e-12||disc<0.)return false;
  double root=sqrt(fmax(disc,0.)),rlo=(-b-root)/(2.*a),rhi=(-b+root)/(2.*a);
  double ylo=-1e300,yhi=1e300;
  if(fabs(uy)<=1e-12){if(fabs(p[1])>half_y)return false;}
  else{double a1=(-half_y-p[1])/uy,a2=(half_y-p[1])/uy;ylo=fmin(a1,a2);yhi=fmax(a1,a2);}
  enter=fmax(0.,fmax(rlo,ylo));leave=fmin(rhi,yhi);return leave>enter;
}
__device__ __forceinline__ void mlp(
    double z,const double *entry,const double *exitp,const double *din,const double *dout,
    double thtot,double tthtot,double ttot,double &x,double &y){
  double length=exitp[2]-entry[2],u=fmin(fmax(z-entry[2],1e-6),length-1e-6),rem=length-u;
  double th1,tth1,t1;integrals(u,th1,tth1,t1);
  double q1=u*th1-tth1;Mat2 s1={u*(2.*q1-u*th1)+t1,q1,q1,th1};double c1=scatter(u);
  s1={s1.a*c1,s1.b*c1,s1.c*c1,s1.d*c1};
  double th2=thtot-th1,q2=length*th2-tthtot+tth1;
  Mat2 s2={length*(2.*q2-length*th2)+ttot-t1,q2,q2,th2};double c2=scatter(rem);
  s2={s2.a*c2,s2.b*c2,s2.c*c2,s2.d*c2};
  Mat2 r0={1.,u,0.,1.},r1={1.,rem,0.,1.},r1i={1.,-rem,0.,1.};
  Mat2 r1t={1.,0.,rem,1.},r1it={1.,0.,-rem,1.};
  Mat2 p1=mul2(mul2(mul2(r1i,s2),inv2(add2(mul2(r1i,s2),mul2(s1,r1t)))),r0);
  Mat2 p2=mul2(s1,inv2(add2(mul2(r1,s1),mul2(s2,r1it))));
  double aix=atan(din[0]/din[2]),aiy=atan(din[1]/din[2]);
  double aox=atan(dout[0]/dout[2]),aoy=atan(dout[1]/dout[2]);
  x=p1.a*entry[0]+p1.b*aix+p2.a*exitp[0]+p2.b*aox;
  y=p1.a*entry[1]+p1.b*aiy+p2.a*exitp[1]+p2.b*aoy;
}
extern "C" __global__ void debug_mlp_points(
    const float *pin,const float *pout,const float *din_f,const float *dout_f,
    const float *z_values,int rays,double radius,double half_y,float *points,unsigned char *valid){
  int ray=blockDim.x*blockIdx.x+threadIdx.x;if(ray>=rays)return;
  const float *pi=pin+3*ray,*po=pout+3*ray,*dif=din_f+3*ray,*dof=dout_f+3*ray;
  double p0[3]={pi[0],pi[1],pi[2]},p1[3]={po[0],po[1],po[2]};
  double di[3]={dif[0],dif[1],dif[2]},doo[3]={dof[0],dof[1],dof[2]},neg[3]={-doo[0],-doo[1],-doo[2]};
  double ti,li,to,lo;if(!cylinder_forward(p0,di,radius,half_y,ti,li)||!cylinder_forward(p1,neg,radius,half_y,to,lo)){valid[ray]=0;return;}
  double ndi=sqrt(di[0]*di[0]+di[1]*di[1]+di[2]*di[2]),ndo=sqrt(doo[0]*doo[0]+doo[1]*doo[1]+doo[2]*doo[2]);
  double entry[3],exitp[3];
  #pragma unroll
  for(int k=0;k<3;++k){entry[k]=p0[k]+ti*di[k]/ndi;exitp[k]=p1[k]-to*doo[k]/ndo;}
  double z=z_values[ray];if(exitp[2]<=entry[2]+1e-5||z<=entry[2]||z>=exitp[2]){valid[ray]=0;return;}
  double th,tth,t;integrals(exitp[2]-entry[2],th,tth,t);double x,y;
  mlp(z,entry,exitp,di,doo,th,tth,t,x,y);
  points[3*ray]=(float)x;points[3*ray+1]=(float)y;points[3*ray+2]=(float)z;valid[ray]=1;
}
extern "C" __global__ void debug_cylinder_intervals(
    const float *position,const float *direction,int rays,double radius,double half_y,
    double *enter,double *leave,unsigned char *valid){
  int ray=blockDim.x*blockIdx.x+threadIdx.x;if(ray>=rays)return;
  const float *p=position+3*ray,*d=direction+3*ray;
  double pd[3]={p[0],p[1],p[2]},dd[3]={d[0],d[1],d[2]},ti,tl;
  bool ok=cylinder_forward(pd,dd,radius,half_y,ti,tl);
  enter[ray]=ok?ti:0.;leave[ray]=ok?tl:0.;valid[ray]=ok?1:0;
}
extern "C" __global__ void build_paths(
    const float *pin,const float *pout,const float *din_f,const float *dout_f,
    int rays,int samples,double z0,double step,double radius,double half_y,
    int nx,int ny,int nz,double ox,double oy,double oz,double sx,double sy,double sz,
    double ca,double sa,int *pixels,float *weights,float *row_sum){
  int ray=blockDim.x*blockIdx.x+threadIdx.x;if(ray>=rays)return;
  const float *pi=pin+3*ray,*po=pout+3*ray,*dif=din_f+3*ray,*dof=dout_f+3*ray;
  double p0[3]={pi[0],pi[1],pi[2]},p1[3]={po[0],po[1],po[2]};
  double di[3]={dif[0],dif[1],dif[2]},doo[3]={dof[0],dof[1],dof[2]},neg[3]={-doo[0],-doo[1],-doo[2]};
  double ti,li,to,lo;if(!cylinder_forward(p0,di,radius,half_y,ti,li)||!cylinder_forward(p1,neg,radius,half_y,to,lo)){row_sum[ray]=0.f;return;}
  double ndi=sqrt(di[0]*di[0]+di[1]*di[1]+di[2]*di[2]),ndo=sqrt(doo[0]*doo[0]+doo[1]*doo[1]+doo[2]*doo[2]);
  double entry[3],exitp[3];
  #pragma unroll
  for(int k=0;k<3;++k){entry[k]=p0[k]+ti*di[k]/ndi;exitp[k]=p1[k]-to*doo[k]/ndo;}
  if(exitp[2]<=entry[2]+1e-5){row_sum[ray]=0.f;return;}
  double length=exitp[2]-entry[2],tht,ttht,tt;integrals(length,tht,ttht,tt);
  double prevx=entry[0],prevy=entry[1],currx,curry,nextx,nexty;
  mlp(z0,entry,exitp,di,doo,tht,ttht,tt,currx,curry);
  mlp(z0+step,entry,exitp,di,doo,tht,ttht,tt,nextx,nexty);
  double sum=0.;
  for(int j=0;j<samples;++j){
    int base=(ray*samples+j)*8;
    #pragma unroll
    for(int q=0;q<8;++q){pixels[base+q]=-1;weights[base+q]=0.f;}
    double z=z0+j*step;
    if(z>entry[2]+1e-6&&z<exitp[2]-1e-6&&fabs(curry)<=half_y+1e-6&&currx*currx+z*z<=radius*radius+1e-6){
      double dx=(j==0?(nextx-currx):(j==samples-1?(currx-prevx):(nextx-prevx)*0.5))/step;
      double dy=(j==0?(nexty-curry):(j==samples-1?(curry-prevy):(nexty-prevy)*0.5))/step;
      double xo=ca*currx-sa*z,zo=sa*currx+ca*z,yo=curry;
      double cx=(xo-ox)/sx,cy=(yo-oy)/sy,cz=(zo-oz)/sz;
      int ix=(int)floor(cx),iy=(int)floor(cy),iz=(int)floor(cz);
      if(ix>=0&&ix<nx-1&&iy>=0&&iy<ny-1&&iz>=0&&iz<nz-1){
        double fx=cx-ix,fy=cy-iy,fz=cz-iz,ds=step*sqrt(1.+dx*dx+dy*dy);
        int q=0;
        for(int dz=0;dz<2;++dz)for(int dyv=0;dyv<2;++dyv)for(int dxv=0;dxv<2;++dxv){
          pixels[base+q]=((iz+dz)*ny+(iy+dyv))*nx+(ix+dxv);
          weights[base+q]=(float)(ds*(dxv?fx:1.-fx)*(dyv?fy:1.-fy)*(dz?fz:1.-fz));++q;
        }sum+=ds;
      }
    }
    prevx=currx;prevy=curry;currx=nextx;curry=nexty;
    if(j+2<samples)mlp(z0+(j+2)*step,entry,exitp,di,doo,tht,ttht,tt,nextx,nexty);
  }row_sum[ray]=(float)sum;
}
extern "C" __global__ void forward_rows(
    const float *image,const int *pixels,const float *weights,const float *row_sum,
    const float *wepl,int rays,int samples,float *normalized,float *res2,unsigned char *valid){
  int r=blockDim.x*blockIdx.x+threadIdx.x;if(r>=rays)return;
  if(row_sum[r]<=0.f){normalized[r]=res2[r]=0.f;valid[r]=0;return;}
  double pred=0.;int begin=r*samples*8,end=begin+samples*8;
  for(int k=begin;k<end;++k)if(pixels[k]>=0)pred+=(double)weights[k]*image[pixels[k]];
  float residual=wepl[r]-(float)pred;normalized[r]=residual/row_sum[r];res2[r]=residual*residual;valid[r]=1;
}
extern "C" __global__ void predict_rows(
    const float *image,const int *pixels,const float *weights,const float *row_sum,
    int rays,int samples,float *prediction,unsigned char *valid){
  int r=blockDim.x*blockIdx.x+threadIdx.x;if(r>=rays)return;
  if(row_sum[r]<=0.f){prediction[r]=0.f;valid[r]=0;return;}
  double pred=0.;int begin=r*samples*8,end=begin+samples*8;
  for(int k=begin;k<end;++k)if(pixels[k]>=0)pred+=(double)weights[k]*image[pixels[k]];
  prediction[r]=(float)pred;valid[r]=1;
}
extern "C" __global__ void back_rows(
    const int *pixels,const float *weights,const float *value,const unsigned char *valid,
    int rays,int samples,float *numerator,float *denominator,int add_denominator){
  int r=blockDim.x*blockIdx.x+threadIdx.x;if(r>=rays||!valid[r])return;
  int begin=r*samples*8,end=begin+samples*8;
  for(int k=begin;k<end;++k){int p=pixels[k];if(p>=0){float w=weights[k];atomicAdd(numerator+p,w*value[r]);if(add_denominator)atomicAdd(denominator+p,w);}}
}
"""


class GpuMlpProjector3D:
    def __init__(self, config: dict):
        import cupy as cp

        self.cp = cp
        grid = config["grid"]
        self.nx, self.ny, self.nz = (int(x) for x in grid["size_xyz"])
        self.sx, self.sy, self.sz = (float(x) for x in grid["spacing_xyz_mm"])
        self.ox, self.oy, self.oz = (float(x) for x in grid["origin_xyz_mm"])
        self.step = float(grid["path_step_mm"])
        self.radius = float(config["phantom_radius_mm"])
        self.half_y = float(config["phantom_half_length_y_mm"])
        self.samples = int(round(2 * self.radius / self.step))
        module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++11",))
        self.build_kernel = module.get_function("build_paths")
        self.debug_kernel = module.get_function("debug_mlp_points")
        self.intersection_kernel = module.get_function("debug_cylinder_intervals")
        self.forward_kernel = module.get_function("forward_rows")
        self.predict_kernel = module.get_function("predict_rows")
        self.back_kernel = module.get_function("back_rows")

    def debug_cylinder_intervals(self, position: np.ndarray, direction: np.ndarray):
        cp = self.cp
        position = cp.asarray(np.ascontiguousarray(position, dtype=np.float32))
        direction = cp.asarray(np.ascontiguousarray(direction, dtype=np.float32))
        if position.shape != direction.shape or position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position and direction must have shape (N,3)")
        n = len(position)
        enter, leave = cp.empty(n, cp.float64), cp.empty(n, cp.float64)
        valid = cp.empty(n, cp.uint8)
        threads, blocks = 128, ((n + 127) // 128,)
        self.intersection_kernel(
            blocks,
            (threads,),
            (position, direction, np.int32(n), np.float64(self.radius),
             np.float64(self.half_y), enter, leave, valid),
        )
        return cp.asnumpy(enter), cp.asnumpy(leave), cp.asnumpy(valid).astype(bool)

    def debug_mlp_points(self, batch: dict[str, np.ndarray], z_values: np.ndarray):
        cp = self.cp
        n = len(batch["wepl_mm"])
        inputs = [
            cp.asarray(np.ascontiguousarray(batch[key], dtype=np.float32))
            for key in ("position_in", "position_out", "direction_in", "direction_out")
        ]
        z = cp.asarray(np.ascontiguousarray(z_values, dtype=np.float32))
        points, valid = cp.empty((n, 3), cp.float32), cp.empty(n, cp.uint8)
        threads, blocks = 128, ((n + 127) // 128,)
        self.debug_kernel(
            blocks,
            (threads,),
            (*inputs, z, np.int32(n), np.float64(self.radius), np.float64(self.half_y), points, valid),
        )
        return cp.asnumpy(points), cp.asnumpy(valid).astype(bool)

    def _paths(self, batch: dict[str, np.ndarray], angle_deg: float):
        cp = self.cp
        n = len(batch["wepl_mm"])
        inputs = [
            cp.asarray(np.ascontiguousarray(batch[key], dtype=np.float32))
            for key in ("position_in", "position_out", "direction_in", "direction_out")
        ]
        entries = n * self.samples * 8
        pixels = cp.empty(entries, cp.int32)
        weights = cp.empty(entries, cp.float32)
        row_sum = cp.empty(n, cp.float32)
        threads, blocks = 128, ((n + 127) // 128,)
        angle = np.deg2rad(angle_deg)
        self.build_kernel(
            blocks,
            (threads,),
            (
                *inputs,
                np.int32(n),
                np.int32(self.samples),
                np.float64(-self.radius + 0.5 * self.step),
                np.float64(self.step),
                np.float64(self.radius),
                np.float64(self.half_y),
                np.int32(self.nx),
                np.int32(self.ny),
                np.int32(self.nz),
                np.float64(self.ox),
                np.float64(self.oy),
                np.float64(self.oz),
                np.float64(self.sx),
                np.float64(self.sy),
                np.float64(self.sz),
                np.float64(np.cos(angle)),
                np.float64(np.sin(angle)),
                pixels,
                weights,
                row_sum,
            ),
        )
        return pixels, weights, row_sum, blocks, threads

    def build_paths(self, batch: dict[str, np.ndarray], angle_deg: float):
        """Expose an immutable path bundle for Stage 8C diagnostics."""
        pixels, weights, row_sum, blocks, threads = self._paths(batch, angle_deg)
        return pixels, weights, row_sum, blocks, threads, len(batch["wepl_mm"])

    def predict_from_paths(self, image, paths):
        cp = self.cp
        pixels, weights, row_sum, blocks, threads, n = paths
        result, valid = cp.empty(n, cp.float32), cp.empty(n, cp.uint8)
        self.predict_kernel(
            blocks,
            (threads,),
            (image, pixels, weights, row_sum, np.int32(n), np.int32(self.samples), result, valid),
        )
        return result, valid

    def accumulate_from_paths(self, image, measured, paths, numerator, denominator):
        cp = self.cp
        pixels, weights, row_sum, blocks, threads, n = paths
        measured = cp.asarray(measured, dtype=cp.float32)
        normalized, res2 = cp.empty(n, cp.float32), cp.empty(n, cp.float32)
        valid = cp.empty(n, cp.uint8)
        self.forward_kernel(
            blocks,
            (threads,),
            (image, pixels, weights, row_sum, measured, np.int32(n), np.int32(self.samples),
             normalized, res2, valid),
        )
        self.back_kernel(
            blocks,
            (threads,),
            (pixels, weights, normalized, valid, np.int32(n), np.int32(self.samples),
             numerator, denominator, np.int32(1)),
        )
        return float(cp.sum(res2, dtype=cp.float64).get()), int(cp.sum(valid, dtype=cp.int64).get())

    def coverage_from_paths(self, paths):
        cp = self.cp
        output = cp.zeros(self.nx * self.ny * self.nz, cp.float32)
        self.accumulate_coverage_from_paths(paths, output)
        return output.reshape(self.nz, self.ny, self.nx)

    def accumulate_coverage_from_paths(self, paths, output):
        cp = self.cp
        pixels, weights, row_sum, blocks, threads, n = paths
        flat = output.reshape(-1)
        valid = (row_sum > 0).astype(cp.uint8)
        values = cp.ones(n, cp.float32)
        self.back_kernel(
            blocks,
            (threads,),
            (pixels, weights, values, valid, np.int32(n), np.int32(self.samples),
             flat, flat, np.int32(0)),
        )
        return output

    def accumulate(self, image, batch, angle_deg, numerator, denominator):
        cp = self.cp
        n = len(batch["wepl_mm"])
        if n == 0:
            return 0.0, 0
        pixels, weights, row_sum, blocks, threads = self._paths(batch, angle_deg)
        wepl = cp.asarray(np.ascontiguousarray(batch["wepl_mm"], dtype=np.float32))
        normalized, res2 = cp.empty(n, cp.float32), cp.empty(n, cp.float32)
        valid = cp.empty(n, cp.uint8)
        self.forward_kernel(
            blocks,
            (threads,),
            (image, pixels, weights, row_sum, wepl, np.int32(n), np.int32(self.samples), normalized, res2, valid),
        )
        self.back_kernel(
            blocks,
            (threads,),
            (pixels, weights, normalized, valid, np.int32(n), np.int32(self.samples), numerator, denominator, np.int32(1)),
        )
        return float(cp.sum(res2, dtype=cp.float64).get()), int(cp.sum(valid, dtype=cp.int64).get())

    def predict(self, image, batch, angle_deg):
        paths = self.build_paths(batch, angle_deg)
        result, valid = self.predict_from_paths(image, paths)
        return result, valid, paths[:3]

    def transpose(self, values, valid, paths):
        cp = self.cp
        pixels, weights, _ = paths
        n = len(values)
        output = cp.zeros(self.nx * self.ny * self.nz, cp.float32)
        dummy = cp.zeros_like(output)
        threads, blocks = 128, ((n + 127) // 128,)
        self.back_kernel(
            blocks,
            (threads,),
            (pixels, weights, values, valid, np.int32(n), np.int32(self.samples), output, dummy, np.int32(0)),
        )
        return output.reshape(self.nz, self.ny, self.nx)

    def evaluate(self, image, batch, angle_deg):
        cp = self.cp
        prediction, valid, _ = self.predict(image, batch, angle_deg)
        measured = cp.asarray(np.ascontiguousarray(batch["wepl_mm"], dtype=np.float32))
        residual = measured - prediction
        mask = valid.astype(cp.bool_)
        values = residual[mask]
        return {
            "squared": float(cp.sum(values * values, dtype=cp.float64).get()),
            "absolute": float(cp.sum(cp.abs(values), dtype=cp.float64).get()),
            "signed": float(cp.sum(values, dtype=cp.float64).get()),
            "count": int(values.size),
        }
