package flib.complexkernels;

import flib.math.VectorFun;
import flib.fftfunctions.Convolution;
import flib.complexkernels.ShapeFilters;

public class PowerMaxFilters {
	
	public static double[] powerMax(final double[] x, final double[] s, final int[] w, double p){
		double a = VectorFun.min(x)[0];
		return VectorFun.add(VectorFun.pow((new Convolution(w)).convolveKernel(VectorFun.pow(VectorFun.add(x,-a),p),s),1/p),a);
	}
	
	public static double[] expMax(final double[] x, final double[] s, final int[] w, double alpha){
		double a = VectorFun.sum(x)/x.length;
		return VectorFun.add(VectorFun.mult(VectorFun.log((new Convolution(w)).convolveKernel(VectorFun.exp(VectorFun.mult(VectorFun.add(x,-a),alpha)),s)),1/alpha),a);
	}
}