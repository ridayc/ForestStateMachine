package flib.ij.stack;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.fftfunctions.Convolution;
import flib.complexkernels.ConvertKernel;
import flib.complexkernels.Gaussian;
import flib.fftfunctions.FFTWrapper;
import flib.math.ComplexMath;
import flib.math.VectorFun;
import flib.math.VectorConv;

public class StackConvolve {
	private Convolution conv;
	
	public StackConvolve(int kw, int kh){
		conv = new Convolution(kw,kh);
	}
	
	public ImagePlus convolveComplexKernel(final ImagePlus imp, final double[] k){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int s = w*h;
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		double[] y;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		double[] t;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			y = conv.convolveComplexKernel(x,k);
			t = VectorFun.min(y);
			if (t[0]<min){
				min = t[0];
			}
			t = VectorFun.max(y);
			if (t[0]>max){
				max = t[0];
			}
			stack2.addSlice("None", new FloatProcessor(w,h,y));
		}
		ImagePlus imp2= new ImagePlus("convolved",stack2);
		imp2.getProcessor().setMinAndMax(min,max);
		return imp2;
	}
	
	public ImagePlus convolveKernel(final ImagePlus imp, final double[] k){
		int w = imp.getWidth();
		int h = imp.getHeight();
		return convolveComplexKernel(imp,ConvertKernel.generateComplexKernel(w,h,k));
	}

	
	public double[] convolveComplexKernel(final double[] im, final double[] k){
		return conv.convolveComplexKernel(im,k);
	}
	
	public ImagePlus gaussian(final ImagePlus imp, double sigma){
		int w = imp.getWidth();
		int h = imp.getHeight();
		return convolveComplexKernel(imp,(new Gaussian(new int[]{w,h},sigma)).getKernel());
	}
}