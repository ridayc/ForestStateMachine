package flib.complexkernels;

import flib.fftfunctions.FFTWrapper;
import flib.math.ComplexMath;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;

// class for conversion of 1,2 and 3 dimensional filters to complex representations that
// can be used for convolution
public class ConvertKernel {
	// a function to shift kernels generated in the frequency domain
	// 1D
	public static double[] shiftKernel(int w, final double[] x){
		// we need to shift the real and the imaginary components of the kernel
		// individually
		return ComplexMath.complexVector(FFTWrapper.fftshift1(w,ComplexMath.getReal(x)),FFTWrapper.fftshift1(w,ComplexMath.getComplex(x)));
	}
	// 2D
	public static double[] shiftKernel(int w, int h, final double[] x){
		return ComplexMath.complexVector(FFTWrapper.fftshift2(w,h,ComplexMath.getReal(x)),FFTWrapper.fftshift2(w,h,ComplexMath.getComplex(x)));
	}
	// 3D
	public static double[] shiftKernel(int w, int h, int z, final double[] x){
		return ComplexMath.complexVector(FFTWrapper.fftshift3(w,h,z,ComplexMath.getReal(x)),FFTWrapper.fftshift3(w,h,z,ComplexMath.getComplex(x)));
	}
	
	public static double[] shiftKernel(final int[] w, final double[] x){
		if (w.length==1){
			return shiftKernel(w[0],x);
		}
		else if (w.length==2){
			return shiftKernel(w[0],w[1],x);
		}
		else if (w.length==3){
			return shiftKernel(w[0],w[1],w[2],x);
		}
		else {
			return x;
		}
	}
	
	// a function to inverse shift kernels generated in the frequency domain
	// 1D
	public static double[] ishiftKernel(int w, final double[] x){
		// we need to shift the real and the imaginary components of the kernel
		// individually
		return ComplexMath.complexVector(FFTWrapper.ifftshift1(w,ComplexMath.getReal(x)),FFTWrapper.ifftshift1(w,ComplexMath.getComplex(x)));
	}
	// 2D
	public static double[] ishiftKernel(int w, int h, final double[] x){
		return ComplexMath.complexVector(FFTWrapper.ifftshift2(w,h,ComplexMath.getReal(x)),FFTWrapper.ifftshift2(w,h,ComplexMath.getComplex(x)));
	}
	// 3D
	public static double[] ishiftKernel(int w, int h, int z, final double[] x){
		return ComplexMath.complexVector(FFTWrapper.ifftshift3(w,h,z,ComplexMath.getReal(x)),FFTWrapper.ifftshift3(w,h,z,ComplexMath.getComplex(x)));
	}
	
	public static double[] ishiftKernel(final int[] w, final double[] x){
		if (w.length==1){
			return ishiftKernel(w[0],x);
		}
		else if (w.length==2){
			return ishiftKernel(w[0],w[1],x);
		}
		else if (w.length==3){
			return ishiftKernel(w[0],w[1],w[2],x);
		}
		else {
			return x;
		}
	}
	
	// a function to generate a frequency kernel from a spatial kernel
	// 1D
	public static double[] generateComplexKernel(int w, final double[] x){
		double[] c = ComplexMath.complexVector(x);
		(new DoubleFFT_1D(w)).complexForward(c);
		return c;
	}
	// 2D
	public static double[] generateComplexKernel(int w, int h, final double[] x){
		double[] c = ComplexMath.complexVector(x);
		(new DoubleFFT_2D(h,w)).complexForward(c);
		return c;
	}	
	// 3D
	public static double[] generateComplexKernel(int w, int h, int z, final double[] x){
		double[] c = ComplexMath.complexVector(x);
		(new DoubleFFT_3D(z,h,w)).complexForward(c);
		return c;
	}
	
	public static double[] generateComplexKernel(final int[] w, final double[] x){
		if (w.length==1){
			return generateComplexKernel(w[0],x);
		}
		else if (w.length==2){
			return generateComplexKernel(w[0],w[1],x);
		}
		else if (w.length==3){
			return generateComplexKernel(w[0],w[1],w[2],x);
		}
		else {
			return x;
		}
	}
}