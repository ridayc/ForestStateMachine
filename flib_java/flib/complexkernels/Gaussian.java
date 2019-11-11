package flib.complexkernels;

import java.lang.Math;
import flib.complexkernels.ConvertKernel;
import flib.math.VectorFun;

/* Implementation of a 1, 2 or 3D Gaussian filter in the frequency domain. 
The filter is radially symmetric. The function has the following parameters:
w: width of the filter in pixels
h: height of the filter in pixels
sigma: sigma (in regards to the real domain) of the Gaussian)
*/

public class Gaussian {
	private int[] w;
	private double sigma;
	
	// representation of the filter in a complex double vector
	private double[] c;
	
	public Gaussian(final int[] w, double sigma) {
		this.w = w.clone();
		this.sigma = sigma;
		this.generate();
	}
	
	// getter functions
	public int[] getSize() {
		return this.w.clone();
	}
	
	public int[] getSizer() {
		return this.w;
	}
	
	public double getSigma() {
		return this.sigma;
	}
	
	public double[] getKernel() {
		return this.c.clone();
	}
	
	public double[] getKernelr() {
		return this.c;
	}
	
	// setter functions
	public void setWidth(final int[] w) {
		this.w = w.clone();
		this.generate();
	}
	
	public void setSigma(double sigma) {
		this.sigma = sigma;
		this.generate();
	}
	
	// Gaussian filter generation
	private void generate() {
		int l = 1;
		double[] mw = new double[w.length];
		double[] wf = new double[w.length];
		for (int i=0; i<w.length; i++){
			l*=w[i];
			// center the coordinate system
			mw[i] = w[i]*0.5;
			// temporary calculation variable
			wf[i] = 2.0/w[i];
		}
		// constant variables to save calculation time
		double[] x = new double[w.length];
		int[] a = new int[w.length];
		double[] t = new double[1];
		t[0] = 2*Math.PI*Math.PI*this.sigma*this.sigma;
		// allocate memory for the filter
		this.c = new double[l*2];
		// for loop for value filling
		this.nestedLoop(this.w.length-1,x,t,a,VectorFun.cummult(w),mw,wf);
		if (w.length==1){
			this.c = ConvertKernel.shiftKernel(w[0],c);
		}
		else if(w.length==2){
			this.c = ConvertKernel.shiftKernel(w[0],w[1],c);
		}
		else if(w.length==3){
			this.c = ConvertKernel.shiftKernel(w[0],w[1],w[2],c);
		}
	}
	
	// a recursive nested for loop function to handle multidimensional cases
	private void nestedLoop(int l, double[] x, double[] t, int[] a, final int[] wcm, final double[] mw, final double[] wf){
		// inner most for loop
		// this hopefully takes more time than the function call for the recursive 
		// for loop
		if (l<1){
			for (int i=0; i<this.w[0]; i++){
				x[0] = (i-mw[0])*wf[0];
				this.c[2*(i+a[0])] = Math.exp(-VectorFun.sum(VectorFun.mult(x,x))*t[0]);
			}
		}
		// apply the recursion
		else {
			for (int i=0; i<this.w[l]; i++){
				// overwrite the lower level index offset
				a[l-1] = a[l];
				// set variables that depend on the value of the current level
				x[l] = (i-mw[l])*wf[l];
				// apply the recursion
				nestedLoop(l-1, x, t, a, wcm, mw, wf);
				// index offset due to the current level
				a[l]+=wcm[l-1];
			}
		}
	}
}