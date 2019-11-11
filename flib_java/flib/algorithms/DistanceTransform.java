package flib.algorithms;

import flib.math.VectorFun;
import flib.math.VectorAccess;

public class DistanceTransform {
	static private final double LARGE_NUMBER = Double.MAX_VALUE;
	// squared 1d distance transform
	public static double[] dt1ds(final double[] x, int type){
		// type = 0 indicates that the input still needs to be transformed to the 
		// appropriate form
		double[] y = x.clone();
		int l = x.length;
		if (type==0){
			for (int i=0; i<l; i++){
				if (x[i]<=0){
					y[i] = 0;
				}
				else {
					y[i] = LARGE_NUMBER;
				}
			}
		}
		int k = 0;
		int[] v = new int[l];
		double[] z = new double[l+1];
		double[] dt = new double[l];
		z[0] = -Double.MAX_VALUE;
		z[1] = Double.MAX_VALUE;
		double s;
		for (int i=1; i<l; i++){
			s = ((y[i]-y[v[k]])+(i*i-v[k]*v[k]))/(2*(i-v[k]));
			while (s<=z[k]){
				k--;
				s = ((y[i]-y[v[k]])+(i*i-v[k]*v[k]))/(2*(i-v[k]));
			}
			k++;
			v[k] = i;
			z[k] = s;
			z[k+1] = Double.MAX_VALUE;
		}
		k = 0;
		for (int i=0; i<l; i++){
			while (z[k+1]<i){
				k++;
			}
			dt[i] = (i-v[k])*(i-v[k])+y[v[k]];
		}
		return dt;
	}
	
	public static double[] dt1d(final double[] x, int type){
		return VectorFun.sqrt(dt1ds(x,type));
	}
	
	public static double[] dt2ds(int w, int h, final double[] x, int type){
		double[] y = x.clone();
		if (type==0){
			for (int i=0; i<w*h; i++){
				if (x[i]<=0){
					y[i] = 0;
				}
				else {
					y[i] = LARGE_NUMBER;
				}
			}
		}
		int[] a = new int[w];
		for (int i=0; i<w; i++){
			a[i] = i;
		}
		int[] b = new int[h];
		for (int i=0; i<h; i++){
			b[i] = i*w;
		}
		for (int i=0; i<h; i++){
			int[] c = VectorFun.add(a,i*w);
			VectorAccess.write(y,c,dt1ds(VectorAccess.access(y,c),1));
		}
		for (int i=0; i<w; i++){
			int[] c = VectorFun.add(b,i);
			VectorAccess.write(y,c,dt1ds(VectorAccess.access(y,c),1));
		}
		return y;
	}
	
	public static double[] dt2d(int w, int h, final double[] x, int type){
		return VectorFun.sqrt(dt2ds(w,h,x,type));
	}
}