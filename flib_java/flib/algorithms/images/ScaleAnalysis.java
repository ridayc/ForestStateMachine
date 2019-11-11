package flib.algorithms.images;

import java.lang.Math;
import flib.fftfunctions.Convolution;
import flib.complexkernels.LogGabor;
import flib.complexkernels.Gaussian;
import flib.complexkernels.ShapeFilters;
import flib.math.VectorFun;

public class ScaleAnalysis {
	
	public static double[][] scaleDecomposition(final double[] im, int w, int h, double lambda, double sigma, double fact, int n, int num_ang, double ang){
		int l = im.length; // = w*h
		int[] s = {w,h};
		// initiation of the scale representation
		double[][] decomp = new double[n*num_ang][l];
		// create the convolution object
		Convolution conv = new Convolution(w,h);
		double lambda2 = lambda;
		for (int i=0; i<n; i++){
			// generate the complex log gabor kernel
			for (int j=0; j<num_ang; j++){
				double x = Math.cos(j/(double)num_ang*Math.PI/2);
				double y = Math.sin(j/(double)num_ang*Math.PI/2);
				LogGabor LG = new LogGabor(s,lambda2,sigma,0,new double[]{x,y},ang);
				decomp[i*num_ang+j] = conv.convolveComplexKernel(im,LG.getKernelr());
			}
			lambda2*=fact;
		}
		return decomp;
	}
	
	public static double[] singleDecomposition(final double[] im, int w, int h, double lambda, double sigma, double num_ang, int counter, double ang_width){
		int l = im.length; // = w*h
		int[] s = {w,h};
		Convolution conv = new Convolution(w,h);
		double x = Math.cos(counter/(double)num_ang*Math.PI/2);
		double y = Math.sin(counter/(double)num_ang*Math.PI/2);
		LogGabor LG = new LogGabor(s,lambda,sigma,0,new double[]{x,y},ang_width);
		double[] decomp = conv.convolveComplexKernel(im,LG.getKernelr());
		return decomp;
	}
	
	public static double[][] gaussianDecomposition(final double[] im, int w, int h, double sigma, double fact, int n){
		int l = im.length; // = w*h
		int[] s = {w,h};
		// initiation of the scale representation
		double[][] decomp = new double[n][l];
		// create the convolution object
		Convolution conv = new Convolution(w,h);
		double lambda = sigma;
		for (int i=0; i<n; i++){
			// generate the complex log gabor kernel
			Gaussian G = new Gaussian(s,lambda);
			decomp[i] = conv.convolveComplexKernel(im,G.getKernelr());
			lambda*=fact;
		}
		return decomp;
	}
	
	//public static double[][] imageSpectrogram(final double[][] im, int w, int h,
	
	public static double[] normalize(final double[] im, int w, int h, double sigma){
		int[] s = {w,h};
		if (sigma<0){
			return im.clone();
		}
		else {
			// create the convolution object
			Convolution conv = new Convolution(w,h);
			// generate the complex Gaussian kernel
			Gaussian G = new Gaussian(s,sigma);
			//double[] shape = ShapeFilters.nSphereFilter(s,sigma, new double[2]);
			// mean value
			double[] x = conv.convolveComplexKernel(im,G.getKernelr());
			// - mean of squares
			return VectorFun.div(VectorFun.sub(im,x),VectorFun.sqrt(VectorFun.sub(conv.convolveComplexKernel(VectorFun.mult(im,im),G.getKernelr()),VectorFun.mult(x,x))));
		}
	}
	
	public static double[][] scaleNormalization(final double[][] im, int w, int h, double sigma, double fact){
		int l = im[0].length;
		int n = im.length;
		double[][] decomp = new double[n][l];
		double lambda = sigma;
		for (int i=0; i<n; i++){
			decomp[i] = normalize(im[i],w,h,lambda);
			if (lambda<w*0.3/fact&&lambda*0.3<h/fact){
				lambda*=fact;
			}
		}
		return decomp;
	}
}