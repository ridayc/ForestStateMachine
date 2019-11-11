package flib.complexkernels;

import java.lang.Math;
import org.apache.commons.math3.special.Erf;
import flib.complexkernels.ConvertKernel;
import flib.math.VectorFun;

public class SymmetricLogGaborBand{
	private int[] w;
	private double lambda1, lambda2;
	private double sigma;
	
	// representation of the filter in a complex double vector
	private double[] c;
	
	public SymmetricLogGaborBand(final int[] w, double lambda1, double lambda2, double sigma) {
		this.w = w.clone();
		this.lambda1 = lambda1;
		this.lambda2 = lambda2;
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
	
	public double getLambda1() {
		return this.lambda1;
	}
	
	public double getLambda2() {
		return this.lambda2;
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
	
	public void setLambda1(double lambda1) {
		this.lambda1 = lambda1;
		this.generate();
	}
	
	public void setLambda2(double lambda2) {
		this.lambda2 = lambda2;
		this.generate();
	}
	
	// log Gabor filter generation
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
		double[] t = new double[2];
		t[0] = Math.PI/(Math.log(2)*this.sigma);
		// allocate memory for the filter
		this.c = new double[l*2];
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
				// calculate the radial part taking into consideration unequal image proportions
				t[1] = Math.sqrt(VectorFun.sum(VectorFun.mult(x,x)));
				this.c[2*(i+a[0])] = Erf.erf(-Math.log(t[1]*lambda1)*t[0])-Erf.erf(-Math.log(t[1]*lambda2)*t[0]);
				if (Double.isNaN(c[2*(i+a[0])])) c[2*(i+a[0])] = 0;
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