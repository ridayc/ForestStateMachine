package flib.fftfunctions;

import java.lang.Object;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_1D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_2D;
import edu.emory.mathcs.jtransforms.fft.DoubleFFT_3D;
//import org.jtransforms.fft.DoubleFFT_1D;
//import org.jtransforms.fft.DoubleFFT_2D;
//import org.jtransforms.fft.DoubleFFT_3D;
import flib.math.ComplexMath;
import flib.math.VectorFun;
import flib.fftfunctions.FFTWrapper;

/* A wrapper function which provides different types of convolution.
This class is non static (partially due to DoubleFFT_2D not being static) 
and should be intialized for a specific image size
*/

public class Convolution {
	// FFT class used for convolution. It can be 1-,2- or 3D
	// this onject will contain the methods for the FFT calculations
	private Object FFT;
	private int fft_dim = 0;
	private int[] fft_k;
	
	// constructors for the different FFTs
	
	// constructor in 1D
	public Convolution(int kw){
		this.FFT = new DoubleFFT_1D(kw);
		this.fft_dim = 1;
		this.fft_k = new int[]{kw};
	}
	
	// constructor in 2D
	public Convolution(int kw, int kh){
		this.FFT = new DoubleFFT_2D(kh,kw);
		this.fft_dim = 2;
		this.fft_k = new int[]{kw,kh};
	}
	
	// constructor in 3D
	public Convolution(int kw, int kh, int kz){
		this.FFT = new DoubleFFT_3D(kz,kh,kw);
		this.fft_dim = 3;
		this.fft_k = new int[]{kw,kh,kz};
	}
	
	// general constructor
	public Convolution(final int[] k){
		this.fft_dim = k.length;
		this.fft_k = k.clone();
		if (this.fft_dim==1){
			this.FFT = new DoubleFFT_1D(k[0]);
		}
		if (this.fft_dim==2){
			this.FFT = new DoubleFFT_2D(k[1],k[0]);
		}
		if (this.fft_dim==3){
			this.FFT = new DoubleFFT_3D(k[2],k[1],k[0]);
		}
	}
	
	// This method takes a real image x and a complex kernel k and convolves these.
	// k must contain twice as many elements as x!
	public double[] convolveComplexKernel(final double[] x, final double[] k){
		// convert the real vector x to a complex vector
		double[] c = ComplexMath.complexVector(x);
		// depending on the dimensionality calculate the forward FFT of x
		if (this.fft_dim==1){
			((DoubleFFT_1D)this.FFT).complexForward(c);
		}
		else if (this.fft_dim==2){
			((DoubleFFT_2D)this.FFT).complexForward(c);
		}
		else if (this.fft_dim==3){
			((DoubleFFT_3D)this.FFT).complexForward(c);
		}
		// multiply in the frequency domain for convolution
		ComplexMath.complexMulti(c,k);
		// return the multiplied signal to the spatial domain
		if (this.fft_dim==1){
			((DoubleFFT_1D)this.FFT).complexInverse(c,true);
		}
		else if (this.fft_dim==2){
			((DoubleFFT_2D)this.FFT).complexInverse(c,true);
		}
		else if (this.fft_dim==3){
			((DoubleFFT_3D)this.FFT).complexInverse(c,true);
		}
		// return the real part of the calculated vector
		return ComplexMath.getReal(c);
	}
	
	// This method takes a a complex vector x and a complex kernel k and convolves these
	// returning a vector in the spatial domain
	public double[] convolveDoubleComplex(final double[] x, final double[] k){
		// multiply in the frequency domain for convolution
		double c[] = ComplexMath.complexMult(x,k);
		// return the multiplied signal to the spatial domain
		if (this.fft_dim==1){
			((DoubleFFT_1D)this.FFT).complexInverse(c,true);
		}
		else if (this.fft_dim==2){
			((DoubleFFT_2D)this.FFT).complexInverse(c,true);
		}
		else if (this.fft_dim==3){
			((DoubleFFT_3D)this.FFT).complexInverse(c,true);
		}
		// return the real part of the calculated vector
		return ComplexMath.getReal(c);
	}
	
	// method to convolve two real valued functions
	public double[] convolveKernel(final double[] x, final double[] k){
		double[] c = ComplexMath.complexVector(k);
		if (this.fft_dim==1){
			((DoubleFFT_1D)this.FFT).complexInverse(c,true);
		}
		else if (this.fft_dim==2){
			((DoubleFFT_2D)this.FFT).complexInverse(c,true);
		}
		else if (this.fft_dim==3){
			((DoubleFFT_3D)this.FFT).complexInverse(c,true);
		}
		c = FFTWrapper.ifftshiftn(fft_k,convolveComplexKernel(x,c));
		double a = VectorFun.cummult(fft_k)[fft_dim-1];
		return VectorFun.mult(c,a);
	}
}