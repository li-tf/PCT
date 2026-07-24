"""Fused CuPy/CUDA Schulte-MLP projector for list-mode OS-SART."""

from __future__ import annotations

import numpy as np


CUDA_SOURCE = r"""
struct Mat2 { double a, b, c, d; };

__device__ __forceinline__ Mat2 add2(Mat2 x, Mat2 y) {
  return {x.a+y.a, x.b+y.b, x.c+y.c, x.d+y.d};
}
__device__ __forceinline__ Mat2 mul2(Mat2 x, Mat2 y) {
  return {x.a*y.a+x.b*y.c, x.a*y.b+x.b*y.d,
          x.c*y.a+x.d*y.c, x.c*y.b+x.d*y.d};
}
__device__ __forceinline__ Mat2 inv2(Mat2 x) {
  double det = x.a*x.d-x.b*x.c;
  if (fabs(det) < 1.0e-30) det = copysign(1.0e-30, det+1.0e-30);
  return {x.d/det, -x.b/det, -x.c/det, x.a/det};
}

__device__ __forceinline__ void integrals(double u, double &theta,
                                           double &ttheta, double &t) {
  const double coeff[6] = {7.444724e-6, 5.463937e-8, -9.986645e-10,
                           2.026409e-11, -1.420501e-13, 3.899100e-16};
  theta=0.0; ttheta=0.0; t=0.0;
  double power=u;
  #pragma unroll
  for (int k=0; k<6; ++k) {
    theta += coeff[k]*power/(double)(k+1);
    ttheta += coeff[k]*power*u/(double)(k+2);
    t += coeff[k]*power*u*u/(double)(k+3);
    power *= u;
  }
}

__device__ __forceinline__ double scatter_constant(double ux, double uy) {
  double distance=fmax(uy-ux, 1.0e-3);
  double correction=1.0+0.038*log(distance/361.0);
  return (13.6*13.6/361.0)*correction*correction;
}

__device__ __forceinline__ void mlp_position(
    double z, const double *entry, const double *exit,
    const double *din, const double *dout,
    double theta_total, double ttheta_total, double t_total,
    double &x, double &y) {
  double length=exit[2]-entry[2];
  double u=z-entry[2];
  double uc=fmin(fmax(u, 1.0e-6), length-1.0e-6);
  double remaining=length-uc;
  double th1,tth1,t1;
  integrals(uc,th1,tth1,t1);

  double s101=uc*th1-tth1;
  Mat2 s1={uc*(2.0*s101-uc*th1)+t1, s101, s101, th1};
  double c1=scatter_constant(0.0,uc);
  s1={s1.a*c1,s1.b*c1,s1.c*c1,s1.d*c1};

  double th2=theta_total-th1;
  double s201=length*th2-ttheta_total+tth1;
  Mat2 s2={length*(2.0*s201-length*th2)+t_total-t1,s201,s201,th2};
  double c2=scatter_constant(uc,length);
  s2={s2.a*c2,s2.b*c2,s2.c*c2,s2.d*c2};

  Mat2 r0={1.0,uc,0.0,1.0};
  Mat2 r1={1.0,remaining,0.0,1.0};
  Mat2 r1i={1.0,-remaining,0.0,1.0};
  Mat2 r1t={1.0,0.0,remaining,1.0};
  Mat2 r1ti={1.0,0.0,-remaining,1.0};
  Mat2 sum1=add2(mul2(r1i,s2),mul2(s1,r1t));
  Mat2 sum2=add2(mul2(r1,s1),mul2(s2,r1ti));
  Mat2 part1=mul2(mul2(mul2(r1i,s2),inv2(sum1)),r0);
  Mat2 part2=mul2(s1,inv2(sum2));

  double ainx=atan(din[0]/din[2]), ainy=atan(din[1]/din[2]);
  double aoutx=atan(dout[0]/dout[2]), aouty=atan(dout[1]/dout[2]);
  x=part1.a*entry[0]+part1.b*ainx+part2.a*exit[0]+part2.b*aoutx;
  y=part1.a*entry[1]+part1.b*ainy+part2.a*exit[1]+part2.b*aouty;
}

extern "C" __global__ void build_paths(
    const float *pin, const float *pout, const float *din_f, const float *dout_f,
    int rays, int samples, double z0, double step, double radius,
    int size, double spacing, double origin, double cos_angle, double sin_angle,
    int *pixels, float *weights, float *row_sum) {
  int ray=blockDim.x*blockIdx.x+threadIdx.x;
  if (ray>=rays) return;
  const float *pi=pin+3*ray, *po=pout+3*ray, *di_f=din_f+3*ray, *do_f=dout_f+3*ray;
  double p_i[3]={pi[0],pi[1],pi[2]}, p_o[3]={po[0],po[1],po[2]};
  double di[3]={di_f[0],di_f[1],di_f[2]}, dout[3]={do_f[0],do_f[1],do_f[2]};
  double a=di[0]*di[0]+di[2]*di[2];
  double b=2.0*(p_i[0]*di[0]+p_i[2]*di[2]);
  double c=p_i[0]*p_i[0]+p_i[2]*p_i[2]-radius*radius;
  double disc_i=b*b-4.0*a*c;
  double ao=dout[0]*dout[0]+dout[2]*dout[2];
  double bo=2.0*(p_o[0]*dout[0]+p_o[2]*dout[2]);
  double co=p_o[0]*p_o[0]+p_o[2]*p_o[2]-radius*radius;
  double disc_o=bo*bo-4.0*ao*co;
  if (disc_i<0.0 || disc_o<0.0) { row_sum[ray]=0.0f; return; }
  double ri=sqrt(fmax(disc_i,0.0)), ro=sqrt(fmax(disc_o,0.0));
  double in_near=(-b-ri)/(2.0*a), in_far=(-b+ri)/(2.0*a);
  double out_near=(-bo-ro)/(2.0*ao), out_far=(-bo+ro)/(2.0*ao);
  double ti=in_near>=0.0?in_near:in_far;
  double to=out_far<=0.0?out_far:out_near;
  double entry[3],exitp[3];
  #pragma unroll
  for(int k=0;k<3;++k){entry[k]=p_i[k]+ti*di[k];exitp[k]=p_o[k]+to*dout[k];}
  if (ti<0.0 || to>0.0 || exitp[2]<=entry[2]) { row_sum[ray]=0.0f; return; }
  double length=exitp[2]-entry[2], tht,ttht,tt;
  integrals(length,tht,ttht,tt);

  double prevprev_x,prevprev_y,prev_x,prev_y,curr_x,curr_y,next_x,next_y,next2_x,next2_y;
  mlp_position(z0,entry,exitp,di,dout,tht,ttht,tt,curr_x,curr_y);
  mlp_position(z0+step,entry,exitp,di,dout,tht,ttht,tt,next_x,next_y);
  mlp_position(z0+2.0*step,entry,exitp,di,dout,tht,ttht,tt,next2_x,next2_y);
  prevprev_x=prev_x=curr_x; prevprev_y=prev_y=curr_y;
  double sum=0.0;
  for(int j=0;j<samples;++j){
    int base=(ray*samples+j)*4;
    pixels[base]=pixels[base+1]=pixels[base+2]=pixels[base+3]=-1;
    double z=z0+j*step;
    double dx,d_y;
    if(j==0){dx=(-3.0*curr_x+4.0*next_x-next2_x)/(2.0*step);d_y=(-3.0*curr_y+4.0*next_y-next2_y)/(2.0*step);}
    else if(j==samples-1){dx=(3.0*curr_x-4.0*prev_x+prevprev_x)/(2.0*step);d_y=(3.0*curr_y-4.0*prev_y+prevprev_y)/(2.0*step);}
    else {dx=(next_x-prev_x)/(2.0*step);d_y=(next_y-prev_y)/(2.0*step);}
    double u=z-entry[2];
    if(u>1.0e-6 && u<length-1.0e-6){
      // OpenGATE active +y rotation in x-z is [[c,s],[-s,c]].
      // Map scanner coordinates with r = R(theta) F s = F R(-theta) s.
      double xr=cos_angle*curr_x-sin_angle*z;
      double zr=-sin_angle*curr_x-cos_angle*z;
      double cx=(xr-origin)/spacing, cz=(zr-origin)/spacing;
      int ix=(int)floor(cx), iz=(int)floor(cz);
      if(ix>=0 && ix<size-1 && iz>=0 && iz<size-1){
        double fx=cx-ix,fz=cz-iz;
        double ds=step*sqrt(1.0+dx*dx+d_y*d_y);
        pixels[base]=iz*size+ix; pixels[base+1]=iz*size+ix+1;
        pixels[base+2]=(iz+1)*size+ix; pixels[base+3]=(iz+1)*size+ix+1;
        weights[base]=(float)(ds*(1.0-fx)*(1.0-fz));
        weights[base+1]=(float)(ds*fx*(1.0-fz));
        weights[base+2]=(float)(ds*(1.0-fx)*fz);
        weights[base+3]=(float)(ds*fx*fz);
        sum+=ds;
      }
    }
    prevprev_x=prev_x;prevprev_y=prev_y;prev_x=curr_x;prev_y=curr_y;
    curr_x=next_x;curr_y=next_y;next_x=next2_x;next_y=next2_y;
    if(j+3<samples) mlp_position(z0+(j+3)*step,entry,exitp,di,dout,tht,ttht,tt,next2_x,next2_y);
  }
  row_sum[ray]=(float)sum;
}

extern "C" __global__ void forward_rows(
    const float *image, const int *pixels, const float *weights,
    const float *row_sum, const float *wepl, int rays, int samples,
    float *normalized, float *residual_squared, unsigned char *valid) {
  int ray=blockDim.x*blockIdx.x+threadIdx.x;
  if(ray>=rays)return;
  if(row_sum[ray]<=0.0f){normalized[ray]=0.0f;residual_squared[ray]=0.0f;valid[ray]=0;return;}
  double predicted=0.0;
  int begin=ray*samples*4,end=begin+samples*4;
  for(int k=begin;k<end;++k){int p=pixels[k];if(p>=0)predicted+=(double)weights[k]*image[p];}
  float residual=wepl[ray]-(float)predicted;
  normalized[ray]=residual/row_sum[ray];
  residual_squared[ray]=residual*residual;valid[ray]=1;
}

extern "C" __global__ void evaluate_rows(
    const float *image, const int *pixels, const float *weights,
    const float *row_sum, const float *wepl, int rays, int samples,
    float *residual, float *residual_squared, float *residual_absolute,
    unsigned char *valid) {
  int ray=blockDim.x*blockIdx.x+threadIdx.x;
  if(ray>=rays)return;
  if(row_sum[ray]<=0.0f){
    residual[ray]=0.0f;residual_squared[ray]=0.0f;
    residual_absolute[ray]=0.0f;valid[ray]=0;return;
  }
  double predicted=0.0;
  int begin=ray*samples*4,end=begin+samples*4;
  for(int k=begin;k<end;++k){int p=pixels[k];if(p>=0)predicted+=(double)weights[k]*image[p];}
  float value=wepl[ray]-(float)predicted;
  residual[ray]=value;
  residual_squared[ray]=value*value;
  residual_absolute[ray]=fabsf(value);
  valid[ray]=1;
}

extern "C" __global__ void backproject_rows(
    const int *pixels, const float *weights, const float *normalized,
    const unsigned char *valid, int rays, int samples,
    float *numerator, float *denominator) {
  int ray=blockDim.x*blockIdx.x+threadIdx.x;
  if(ray>=rays || !valid[ray])return;
  float value=normalized[ray];
  int begin=ray*samples*4,end=begin+samples*4;
  for(int k=begin;k<end;++k){int p=pixels[k];if(p>=0){float w=weights[k];atomicAdd(numerator+p,w*value);atomicAdd(denominator+p,w);}}
}
"""


class GpuMlpProjector:
    """Own CUDA kernels and accumulate one list-mode batch into a subset."""

    def __init__(self, size: int, spacing_mm: float, step_mm: float, radius_mm: float):
        import cupy as cp

        self.cp = cp
        self.size = int(size)
        self.spacing = float(spacing_mm)
        self.step = float(step_mm)
        self.radius = float(radius_mm)
        self.origin = -0.5 * (self.size - 1) * self.spacing
        self.samples = int(round(2.0 * self.radius / self.step))
        if self.samples < 3 or not np.isclose(self.samples * self.step, 2.0 * self.radius):
            raise ValueError("path step must divide the support diameter and produce at least 3 samples")
        module = cp.RawModule(code=CUDA_SOURCE, options=("--std=c++11",))
        self.build_kernel = module.get_function("build_paths")
        self.forward_kernel = module.get_function("forward_rows")
        self.evaluate_kernel = module.get_function("evaluate_rows")
        self.back_kernel = module.get_function("backproject_rows")

    def accumulate(self, image, batch: dict[str, np.ndarray], angle_deg: float, numerator, denominator):
        cp = self.cp
        n = int(len(batch["wepl_mm"]))
        if n == 0:
            return 0.0, 0
        inputs = [
            cp.asarray(np.ascontiguousarray(batch[name], dtype=np.float32))
            for name in ("position_in", "position_out", "direction_in", "direction_out")
        ]
        wepl = cp.asarray(np.ascontiguousarray(batch["wepl_mm"], dtype=np.float32))
        entries = n * self.samples * 4
        pixels = cp.empty(entries, dtype=cp.int32)
        weights = cp.empty(entries, dtype=cp.float32)
        row_sum = cp.empty(n, dtype=cp.float32)
        normalized = cp.empty(n, dtype=cp.float32)
        residual_squared = cp.empty(n, dtype=cp.float32)
        valid = cp.empty(n, dtype=cp.uint8)
        threads = 128
        blocks = ((n + threads - 1) // threads,)
        angle = np.deg2rad(angle_deg)
        self.build_kernel(
            blocks,
            (threads,),
            (*inputs, np.int32(n), np.int32(self.samples), np.float64(-self.radius + 0.5 * self.step),
             np.float64(self.step), np.float64(self.radius), np.int32(self.size), np.float64(self.spacing),
             np.float64(self.origin), np.float64(np.cos(angle)), np.float64(np.sin(angle)), pixels, weights, row_sum),
        )
        self.forward_kernel(
            blocks, (threads,),
            (image, pixels, weights, row_sum, wepl, np.int32(n), np.int32(self.samples),
             normalized, residual_squared, valid),
        )
        self.back_kernel(
            blocks, (threads,),
            (pixels, weights, normalized, valid, np.int32(n), np.int32(self.samples), numerator, denominator),
        )
        residual_sum = float(cp.sum(residual_squared, dtype=cp.float64).get())
        valid_count = int(cp.sum(valid, dtype=cp.int64).get())
        return residual_sum, valid_count

    def evaluate_many(
        self,
        images: dict[str, object],
        batch: dict[str, np.ndarray],
        angle_deg: float,
    ) -> dict[str, dict[str, float | int]]:
        """Evaluate several fixed images while building each MLP batch once.

        Returned residuals use the measurement convention ``WEPL - A x``.
        No backprojection or image update is performed.
        """

        cp = self.cp
        n = int(len(batch["wepl_mm"]))
        if n == 0:
            return {
                name: {"squared": 0.0, "absolute": 0.0, "signed": 0.0, "count": 0}
                for name in images
            }
        inputs = [
            cp.asarray(np.ascontiguousarray(batch[name], dtype=np.float32))
            for name in ("position_in", "position_out", "direction_in", "direction_out")
        ]
        wepl = cp.asarray(np.ascontiguousarray(batch["wepl_mm"], dtype=np.float32))
        entries = n * self.samples * 4
        pixels = cp.empty(entries, dtype=cp.int32)
        weights = cp.empty(entries, dtype=cp.float32)
        row_sum = cp.empty(n, dtype=cp.float32)
        residual = cp.empty(n, dtype=cp.float32)
        residual_squared = cp.empty(n, dtype=cp.float32)
        residual_absolute = cp.empty(n, dtype=cp.float32)
        valid = cp.empty(n, dtype=cp.uint8)
        threads = 128
        blocks = ((n + threads - 1) // threads,)
        angle = np.deg2rad(angle_deg)
        self.build_kernel(
            blocks,
            (threads,),
            (*inputs, np.int32(n), np.int32(self.samples), np.float64(-self.radius + 0.5 * self.step),
             np.float64(self.step), np.float64(self.radius), np.int32(self.size), np.float64(self.spacing),
             np.float64(self.origin), np.float64(np.cos(angle)), np.float64(np.sin(angle)), pixels, weights, row_sum),
        )
        result: dict[str, dict[str, float | int]] = {}
        for name, image in images.items():
            self.evaluate_kernel(
                blocks,
                (threads,),
                (image, pixels, weights, row_sum, wepl, np.int32(n), np.int32(self.samples),
                 residual, residual_squared, residual_absolute, valid),
            )
            result[name] = {
                "squared": float(cp.sum(residual_squared, dtype=cp.float64).get()),
                "absolute": float(cp.sum(residual_absolute, dtype=cp.float64).get()),
                "signed": float(cp.sum(residual, dtype=cp.float64).get()),
                "count": int(cp.sum(valid, dtype=cp.int64).get()),
            }
        return result
