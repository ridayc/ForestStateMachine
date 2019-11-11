package flib.ij.featureextraction;

import java.lang.Math;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.complexkernels.LogGabor;
import flib.complexkernels.LogGaborBand;
import flib.fftfunctions.FFTWrapper;
import flib.math.ComplexMath;
import flib.fftfunctions.Convolution;
import flib.ij.stack.StackOperations;

public class ScaleFeatureStack {
	
	public static ImagePlus scaleFeatures(final ImagePlus imp, double l, double factor, int numstep, int numang, double angwidth, int numphase, double sigma){
		int w = imp.getWidth();
		int h = imp.getHeight();
		double[] x = VectorConv.float2double((float[])(imp.getProcessor().convertToFloat().getPixels()));
		double[] y,c,cr,ci;
		LogGabor LG;
		Convolution conv = new Convolution(w,h);
		ImageStack stack = new ImageStack(w,h);
		double l2,t;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for (int i=0; i<numstep; i++){
			for (int j=0; j<numang; j++){
				for (int k=0; k<numphase; k++){
					l2 = l*Math.pow(factor,i);
					LG = new LogGabor(new int[]{w,h},l2,sigma,(double)k/numphase*Math.PI,new double[]{Math.cos((double)j/numang*Math.PI),Math.sin((double)j/numang*Math.PI)},angwidth);
					c = LG.getKernel();
					cr = FFTWrapper.fftshift2(w,h,ComplexMath.getReal(c));
					ci = FFTWrapper.fftshift2(w,h,ComplexMath.getComplex(c));
					c = ComplexMath.complexVector(cr,ci);
					//c = ComplexMath.complexVector(cr);
					y = conv.convolveComplexKernel(x,c);
					t = VectorFun.min(y)[0];
					if (t<min){
						min = t;
					}
					t = VectorFun.max(y)[0];
					if (t>max){
						max = t;
					}
					stack.addSlice("None", new FloatProcessor(w,h,y));
				}
			}
		}
		ImagePlus featimp = new ImagePlus("Scale Features",stack);
		featimp.getProcessor().setMinAndMax(min,max);
		return featimp;
	}
	
	public static ImagePlus scaleFeatures(final ImagePlus imp, double l, double factor, int numstep){
		return scaleFeatures(imp,l,factor,numstep,1,1,1,2);
	}
	
	public static ImagePlus bandScaleFeatures(final ImagePlus imp, double l1, double l2, double factor, int numstep, int numang, double angwidth, int numphase, double sigma){
		int w = imp.getWidth();
		int h = imp.getHeight();
		double[] x = VectorConv.float2double((float[])(imp.getProcessor().convertToFloat().getPixels()));
		double[] y,c,cr,ci;
		LogGaborBand LG;
		Convolution conv = new Convolution(w,h);
		ImageStack stack = new ImageStack(w,h);
		double l3,t;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for (int i=0; i<numstep; i++){
			for (int j=0; j<numang; j++){
				for (int k=0; k<numphase; k++){
					l3 = l2*Math.pow(factor,i);
					LG = new LogGaborBand(new int[]{w,h},l1,l3,sigma,(double)k/numphase*Math.PI,new double[]{Math.cos((double)j/numang*Math.PI),Math.sin((double)j/numang*Math.PI)},angwidth);
					c = LG.getKernel();
					cr = FFTWrapper.fftshift2(w,h,ComplexMath.getReal(c));
					ci = FFTWrapper.fftshift2(w,h,ComplexMath.getComplex(c));
					c = ComplexMath.complexVector(cr,ci);
					//c = ComplexMath.complexVector(cr);
					y = conv.convolveComplexKernel(x,c);
					t = VectorFun.min(y)[0];
					if (t<min){
						min = t;
					}
					t = VectorFun.max(y)[0];
					if (t>max){
						max = t;
					}
					stack.addSlice("None", new FloatProcessor(w,h,y));
				}
			}
		}
		ImagePlus featimp = new ImagePlus("Band Scale Features",stack);
		featimp.getProcessor().setMinAndMax(min,max);
		return featimp;
	}
	
	public static ImagePlus maxDim(final ImagePlus imp, int numstep, int numang, int numphase, int type){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = numstep*numang*numphase;
		boolean[] dim = new boolean[3];
		dim[0] = (type/4!=0);
		dim[1] = ((type/2)%2!=0);
		dim[2] = (type%2!=0);
		double[][] stackim = StackOperations.stack2PixelArrays(imp);
		double[][] maxim;
		if (dim[0]){
			n/=numstep;
			maxim = new double[n][w*h];
			// get the initial max values
			for (int i=0; i<n; i++){
				maxim[i] = stackim[i].clone();
			}
			// compare with all other values
			for (int i=0; i<numstep; i++){
				for (int j=0; j<n; j++){
					int a = i*n+j;
					for (int k=0; k<w*h; k++){
						if (Math.abs(stackim[a][k])>Math.abs(maxim[j][k])){
							maxim[j][k] = stackim[a][k];
						}
					}
				}
			}
			stackim = new double[n][w*h];
			for (int i=0; i<n; i++){
				stackim[i] = maxim[i].clone();
			}
			numstep = 1;
		}
		if (dim[1]){
			n/=numang;
			maxim = new double[n][w*h];
			// get the initial max values
			for (int i=0; i<numstep; i++){
				for (int j=0; j<numphase; j++){
					maxim[i*numphase+j] = stackim[i*numphase*numang+j].clone();
				}
			}
			// compare with all other values
			for (int i=0; i<numstep; i++){
				for (int j=0; j<numang; j++){
					for (int k =0; k<numphase; k++){
						int a = i*numang*numphase+j*numphase+k;
						int b = i*numphase+k;
						for (int l=0; l<w*h; l++){
							if (Math.abs(stackim[a][l])>Math.abs(maxim[b][l])){
								maxim[b][l] = stackim[a][l];
							}
						}
					}
				}
			}
			stackim = new double[n][w*h];
			for (int i=0; i<n; i++){
				stackim[i] = maxim[i].clone();
			}
			numang = 1;
		}
		if (dim[2]){
			n/=numphase;
			maxim = new double[n][w*h];
			// get the initial max values
			for (int i=0; i<numstep; i++){
				for (int j=0; j<numang; j++){
					maxim[i*numang+j] = stackim[i*numang*numphase+j*numphase].clone();
				}
			}
			// compare with all other values
			for (int i=0; i<numstep; i++){
				for (int j=0; j<numang; j++){
					for (int k =0; k<numphase; k++){
						int a = i*numang*numphase+j*numphase+k;
						int b = i*numang+j;
						for (int l=0; l<w*h; l++){
							if (Math.abs(stackim[a][l])>Math.abs(maxim[b][l])){
								maxim[b][l] = stackim[a][l];
							}
						}
					}
				}
			}
			stackim = new double[n][w*h];
			for (int i=0; i<n; i++){
				stackim[i] = maxim[i].clone();
			}
		}
		return StackOperations.convert2Stack(stackim,w,h);
	}
}