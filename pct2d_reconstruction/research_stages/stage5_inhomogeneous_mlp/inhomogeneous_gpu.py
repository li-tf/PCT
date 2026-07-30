"""GPU inhomogeneous-MLP projector used only by Stage 5."""

from __future__ import annotations

import numpy as np

from gpu_mlp_operator import CUDA_SOURCE as WATER_CUDA
from robust_gpu import RobustGpuMlpProjector


INHOMOGENEOUS_CUDA = r"""
__device__ __forceinline__ double water_stopping(double energy) {
  const double mp=938.27208816, me=0.51099895, ion=78.0e-6;
  energy=fmax(energy,0.001);
  double beta2=1.0-(mp/(energy+mp))*(mp/(energy+mp));
  double k=4.0*3.141592653589793*2.8179403262e-12*2.8179403262e-12*
           me*3.343e23/1000.0;
  return k*(log(2.0*me/ion*beta2/(1.0-beta2))-beta2)/beta2;
}
__device__ __forceinline__ double energy_scatter(double energy) {
  const double mp=938.27208816;
  energy=fmax(energy,0.1);
  double pv=energy*(energy+2.0*mp)/(energy+mp);
  double ref=200.0*(200.0+2.0*mp)/(200.0+mp);
  return 3.645061873086788e-6*(ref/pv)*(ref/pv);
}
__device__ __forceinline__ float bilinear_map(
    const float *map, int size, double spacing, double origin,
    double x, double z, float outside) {
  double cx=(x-origin)/spacing, cz=(z-origin)/spacing;
  int ix=(int)floor(cx), iz=(int)floor(cz);
  if(ix<0 || iz<0 || ix>=size-1 || iz>=size-1)return outside;
  double fx=cx-ix,fz=cz-iz;
  int p=iz*size+ix;
  return (float)((1.0-fx)*(1.0-fz)*map[p]+fx*(1.0-fz)*map[p+1]+
                 (1.0-fx)*fz*map[p+size]+fx*fz*map[p+size+1]);
}
__device__ __forceinline__ bool endpoints(
    const float *pi, const float *po, const float *di_f, const float *do_f,
    double radius, double *entry, double *exitp, double *di, double *dout) {
  for(int k=0;k<3;++k){di[k]=di_f[k];dout[k]=do_f[k];}
  double a=di[0]*di[0]+di[2]*di[2];
  double b=2.0*(pi[0]*di[0]+pi[2]*di[2]);
  double c=pi[0]*pi[0]+pi[2]*pi[2]-radius*radius;
  double ao=dout[0]*dout[0]+dout[2]*dout[2];
  double bo=2.0*(po[0]*dout[0]+po[2]*dout[2]);
  double co=po[0]*po[0]+po[2]*po[2]-radius*radius;
  double disci=b*b-4.0*a*c, disco=bo*bo-4.0*ao*co;
  if(disci<0.0 || disco<0.0 || a<=0.0 || ao<=0.0)return false;
  double ri=sqrt(disci),ro=sqrt(disco);
  double ni=(-b-ri)/(2.0*a),fi=(-b+ri)/(2.0*a);
  double no=(-bo-ro)/(2.0*ao),fo=(-bo+ro)/(2.0*ao);
  double ti=ni>=0.0?ni:fi,to=fo<=0.0?fo:no;
  for(int k=0;k<3;++k){entry[k]=pi[k]+ti*di[k];exitp[k]=po[k]+to*dout[k];}
  return ti>=0.0 && to<=0.0 && exitp[2]>entry[2];
}
extern "C" __global__ void initialize_inhomogeneous(
    const float *pin,const float *pout,const float *din_f,const float *dout_f,
    int rays,int samples,double z0,double step,double radius,
    float *pathx,float *pathy,unsigned char *ray_valid) {
  int ray=blockDim.x*blockIdx.x+threadIdx.x;if(ray>=rays)return;
  const float *pi=pin+3*ray,*po=pout+3*ray,*df=din_f+3*ray,*of=dout_f+3*ray;
  double en[3],ex[3],di[3],dout[3];
  bool ok=endpoints(pi,po,df,of,radius,en,ex,di,dout);ray_valid[ray]=ok;
  if(!ok)return;
  double len=ex[2]-en[2],th,tth,tt;integrals(len,th,tth,tt);
  for(int j=0;j<samples;++j){
    double x,y;mlp_position(z0+j*step,en,ex,di,dout,th,tth,tt,x,y);
    pathx[ray*samples+j]=(float)x;pathy[ray*samples+j]=(float)y;
  }
}
extern "C" __global__ void integrate_inhomogeneous(
    const float *pin,const float *pout,const float *din_f,const float *dout_f,
    const float *ein,const float *eout,int rays,int samples,double z0,double step,
    double radius,const float *rsp_map,const float *rscp_map,int map_size,
    double map_spacing,double map_origin,double ca,double sa,
    const float *pathx,const unsigned char *ray_valid,
    float *rsp,float *rscp,float *energy,float *p0,float *p1,float *p2) {
  int ray=blockDim.x*blockIdx.x+threadIdx.x;if(ray>=rays||!ray_valid[ray])return;
  const float *pi=pin+3*ray,*po=pout+3*ray,*df=din_f+3*ray,*of=dout_f+3*ray;
  double en[3],ex[3],di[3],dout[3];if(!endpoints(pi,po,df,of,radius,en,ex,di,dout))return;
  int base=ray*samples;double ef=fmax((double)ein[ray],0.1);
  for(int j=0;j<samples;++j){
    double z=z0+j*step,x=pathx[base+j];
    double xr=ca*x-sa*z,zr=-sa*x-ca*z;
    float rv=bilinear_map(rsp_map,map_size,map_spacing,map_origin,xr,zr,0.0f);
    float sv=bilinear_map(rscp_map,map_size,map_spacing,map_origin,xr,zr,0.0f);
    bool inside=z>en[2]&&z<ex[2];
    rsp[base+j]=inside?rv:0.0f;rscp[base+j]=inside?sv:0.0f;energy[base+j]=(float)ef;
    if(inside&&j+1<samples)ef=fmax(0.1,ef-rv*water_stopping(ef)*step);
  }
  double eb=fmax((double)eout[ray],0.1);
  for(int j=samples-1;j>=0;--j){
    double f=(double)j/(double)(samples-1);
    double blended=(1.0-f)*(double)energy[base+j]+f*eb;
    p0[base+j]=(float)((double)rscp[base+j]*energy_scatter(blended));
    if(rsp[base+j]>0.0f&&j>0)eb+=rsp[base+j]*water_stopping(eb)*step;
  }
  double c0=0.0,c1=0.0,c2=0.0;
  for(int j=0;j<samples;++j){
    double s=(z0+j*step)-en[2],t=p0[base+j];
    c0+=t*step;c1+=t*s*step;c2+=t*s*s*step;
    p0[base+j]=(float)c0;p1[base+j]=(float)c1;p2[base+j]=(float)c2;
  }
}
extern "C" __global__ void update_inhomogeneous(
    const float *pin,const float *pout,const float *din_f,const float *dout_f,
    int rays,int samples,double z0,double step,double radius,
    const unsigned char *ray_valid,const float *p0,const float *p1,const float *p2,
    float *pathx,float *pathy) {
  int ray=blockDim.x*blockIdx.x+threadIdx.x;if(ray>=rays||!ray_valid[ray])return;
  const float *pi=pin+3*ray,*po=pout+3*ray,*df=din_f+3*ray,*of=dout_f+3*ray;
  double en[3],ex[3],di[3],dout[3];if(!endpoints(pi,po,df,of,radius,en,ex,di,dout))return;
  int base=ray*samples;double length=ex[2]-en[2];
  double t0=p0[base+samples-1],t1=p1[base+samples-1],t2=p2[base+samples-1];
  for(int j=0;j<samples;++j){
    double z=z0+j*step,s=z-en[2],rem=length-s;
    if(s<=1e-6||s>=length-1e-6)continue;
    double a0=p0[base+j],pp1=p1[base+j],pp2=p2[base+j];
    double a1=s*a0-pp1,a2=s*s*a0-2.0*s*pp1+pp2;
    double q0=t0-a0,q1=t1-pp1,q2=t2-pp2;
    double b0=q0,b1=length*q0-q1,b2=length*length*q0-2.0*length*q1+q2;
    Mat2 s1={a2+1e-18,a1,a1,a0+1e-18},s2={b2+1e-18,b1,b1,b0+1e-18};
    Mat2 r0={1.0,s,0.0,1.0},r1={1.0,rem,0.0,1.0};
    Mat2 r1i={1.0,-rem,0.0,1.0},r1t={1.0,0.0,rem,1.0},r1ti={1.0,0.0,-rem,1.0};
    Mat2 part1=mul2(mul2(mul2(r1i,s2),inv2(add2(mul2(r1i,s2),mul2(s1,r1t)))),r0);
    Mat2 part2=mul2(s1,inv2(add2(mul2(r1,s1),mul2(s2,r1ti))));
    double ainx=atan(di[0]/di[2]),ainy=atan(di[1]/di[2]);
    double aox=atan(dout[0]/dout[2]),aoy=atan(dout[1]/dout[2]);
    pathx[base+j]=(float)(part1.a*en[0]+part1.b*ainx+part2.a*ex[0]+part2.b*aox);
    pathy[base+j]=(float)(part1.a*en[1]+part1.b*ainy+part2.a*ex[1]+part2.b*aoy);
  }
}
extern "C" __global__ void weights_inhomogeneous(
    int rays,int samples,double z0,double step,int size,double spacing,double origin,double radius,
    double ca,double sa,const float *pathx,const float *pathy,
    const unsigned char *ray_valid,int *pixels,float *weights,float *row_sum) {
  int ray=blockDim.x*blockIdx.x+threadIdx.x;if(ray>=rays)return;
  if(!ray_valid[ray]){row_sum[ray]=0.0f;return;}int rb=ray*samples;double sum=0.0;
  for(int j=0;j<samples;++j){
    int base=(rb+j)*4;for(int k=0;k<4;++k){pixels[base+k]=-1;weights[base+k]=0.0f;}
    double z=z0+j*step,x=pathx[rb+j];
    double xr=ca*x-sa*z,zr=-sa*x-ca*z,cx=(xr-origin)/spacing,cz=(zr-origin)/spacing;
    if(xr*xr+zr*zr>radius*radius)continue;
    int ix=(int)floor(cx),iz=(int)floor(cz);
    if(ix<0||iz<0||ix>=size-1||iz>=size-1)continue;
    int jm=j>0?j-1:j,jp=j<samples-1?j+1:j;double dz=(jp-jm)*step;
    double dx=(pathx[rb+jp]-pathx[rb+jm])/dz,dy=(pathy[rb+jp]-pathy[rb+jm])/dz;
    double ds=step*sqrt(1.0+dx*dx+dy*dy),fx=cx-ix,fz=cz-iz;
    pixels[base]=iz*size+ix;pixels[base+1]=iz*size+ix+1;
    pixels[base+2]=(iz+1)*size+ix;pixels[base+3]=(iz+1)*size+ix+1;
    weights[base]=(float)(ds*(1-fx)*(1-fz));weights[base+1]=(float)(ds*fx*(1-fz));
    weights[base+2]=(float)(ds*(1-fx)*fz);weights[base+3]=(float)(ds*fx*fz);sum+=ds;
  }row_sum[ray]=(float)sum;
}
"""


class InhomogeneousGpuMlpProjector(RobustGpuMlpProjector):
    """Numerically integrate RSP/RScP-dependent paths on the GPU."""

    def __init__(self, size, spacing_mm, step_mm, radius_mm, rsp_map, rscp_map, iterations=2):
        super().__init__(size, spacing_mm, step_mm, radius_mm)
        cp = self.cp
        module = cp.RawModule(code=WATER_CUDA + INHOMOGENEOUS_CUDA, options=("--std=c++11",))
        self.init_kernel = module.get_function("initialize_inhomogeneous")
        self.integrate_kernel = module.get_function("integrate_inhomogeneous")
        self.update_kernel = module.get_function("update_inhomogeneous")
        self.weights_kernel = module.get_function("weights_inhomogeneous")
        self.iterations = int(iterations)
        self.update_maps(rsp_map, rscp_map)

    def update_maps(self, rsp_map, rscp_map):
        cp = self.cp
        rsp = np.ascontiguousarray(rsp_map, dtype=np.float32)
        rscp = np.ascontiguousarray(rscp_map, dtype=np.float32)
        if (
            rsp.ndim != 2
            or rsp.shape != rscp.shape
            or rsp.shape != (self.size, self.size)
        ):
            raise ValueError(
                "RSP/RScP maps must match the reconstruction grid"
            )
        if not np.isfinite(rsp).all() or not np.isfinite(rscp).all():
            raise ValueError("material maps contain non-finite values")
        self.map_size = self.size
        self.map_spacing = self.spacing
        self.map_origin = self.origin
        self.rsp_map = cp.asarray(rsp)
        self.rscp_map = cp.asarray(rscp)

    def _device_paths(self, batch, angle_deg):
        cp = self.cp
        n = len(batch["wepl_mm"])
        inputs = [cp.asarray(np.ascontiguousarray(batch[k], dtype=np.float32)) for k in
                  ("position_in", "position_out", "direction_in", "direction_out")]
        ein = cp.asarray(np.ascontiguousarray(batch["energy_in"], dtype=np.float32))
        eout = cp.asarray(np.ascontiguousarray(batch["energy_out"], dtype=np.float32))
        shape = n * self.samples
        x, y = cp.empty(shape, cp.float32), cp.empty(shape, cp.float32)
        valid = cp.empty(n, cp.uint8)
        rsp, rscp, energy = (cp.empty(shape, cp.float32) for _ in range(3))
        p0, p1, p2 = (cp.empty(shape, cp.float32) for _ in range(3))
        threads = 128
        blocks = ((n + threads - 1) // threads,)
        common = (np.int32(n), np.int32(self.samples), np.float64(-self.radius + .5*self.step),
                  np.float64(self.step), np.float64(self.radius))
        self.init_kernel(blocks, (threads,), (*inputs, *common, x, y, valid))
        a = np.deg2rad(angle_deg)
        for _ in range(self.iterations):
            self.integrate_kernel(blocks, (threads,), (
                *inputs, ein, eout, *common, self.rsp_map, self.rscp_map,
                np.int32(self.map_size), np.float64(self.map_spacing), np.float64(self.map_origin),
                np.float64(np.cos(a)), np.float64(np.sin(a)), x, valid,
                rsp, rscp, energy, p0, p1, p2))
            self.update_kernel(blocks, (threads,), (*inputs, *common, valid, p0, p1, p2, x, y))
        entries = shape * 4
        pixels, weights = cp.empty(entries, cp.int32), cp.empty(entries, cp.float32)
        row_sum = cp.empty(n, cp.float32)
        self.weights_kernel(blocks, (threads,), (
            np.int32(n), np.int32(self.samples), np.float64(-self.radius+.5*self.step),
            np.float64(self.step), np.int32(self.size), np.float64(self.spacing),
            np.float64(self.origin), np.float64(self.radius),
            np.float64(np.cos(a)), np.float64(np.sin(a)),
            x, y, valid, pixels, weights, row_sum))
        return blocks, threads, pixels, weights, row_sum, valid

    def _paths_and_forward(self, image, batch, angle_deg):
        cp = self.cp
        n = len(batch["wepl_mm"])
        blocks, threads, pixels, weights, row_sum, valid = self._device_paths(batch, angle_deg)
        wepl = cp.asarray(np.ascontiguousarray(batch["wepl_mm"], dtype=np.float32))
        normalized, squared = cp.empty(n, cp.float32), cp.empty(n, cp.float32)
        self.forward_kernel(blocks, (threads,), (
            image, pixels, weights, row_sum, wepl, np.int32(n), np.int32(self.samples),
            normalized, squared, valid))
        return blocks, threads, pixels, weights, row_sum, normalized, squared, valid

    def residuals(self, image, batch, angle_deg):
        cp = self.cp
        n = len(batch["wepl_mm"])
        blocks, threads, pixels, weights, row_sum, valid = self._device_paths(batch, angle_deg)
        wepl = cp.asarray(np.ascontiguousarray(batch["wepl_mm"], dtype=np.float32))
        residual, squared, absolute = (cp.empty(n, cp.float32) for _ in range(3))
        self.evaluate_kernel(blocks, (threads,), (
            image, pixels, weights, row_sum, wepl, np.int32(n), np.int32(self.samples),
            residual, squared, absolute, valid))
        return cp.asnumpy(residual[valid.astype(cp.bool_)])
