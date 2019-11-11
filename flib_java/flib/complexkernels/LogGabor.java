package flib.complexkernels;

import java.lang.Math;
import flib.complexkernels.ConvertKernel;
import flib.math.VectorFun;

public class LogGabor{
	private int[] w;
	private double lambda;
	private double sigma;
	private double phi;
	private double[] ang;
	private double ang_width;
	
	// representation of the filter in a complex double vector
	private double[] c;
	
	public LogGabor(final int[] w, double lambda, double sigma, double phi, final double[] ang, double ang_width) {
		this.w = w.clone();
		this.lambda = lambda;
		this.sigma = sigma;
		this.phi = phi;
		this.ang_width = ang_width;
		this.ang = VectorFun.div(ang,Math.sqrt(VectorFun.sum(VectorFun.mult(ang,ang))));
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
	
	public double getLambda() {
		return this.lambda;
	}
	
	public double getPhi(){
		return this.phi;
	}
	
	public double[] getAng() {
		return this.ang.clone();
	}
	
	public double[] getAngr() {
		return this.ang;
	}
	
	public double getAng_width(){
		return this.ang_width;
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
	
	public void setLambda(double lambda) {
		this.lambda = lambda;
		this.generate();
	}
	
	public void setPhi(double phi) {
		this.phi = phi;
		this.generate();
	}
	
	public void setAng(final double[] ang) {
		this.ang = VectorFun.div(ang,Math.sqrt(VectorFun.sum(VectorFun.mult(ang,ang))));
		this.generate();
	}
	
	public void setAng_width(double ang_width) {
		this.ang_width = ang_width;
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
		double[] t = new double[7];
		t[0] = Math.PI*Math.PI/(2*Math.pow(Math.log(2)*this.sigma,2));
		t[2] = Math.cos(this.phi);
		t[3] = Math.sin(this.phi);
		t[5] = 1./(2*this.ang_width*this.ang_width);
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
				// cosine to the main direction
				t[4] = VectorFun.sum(VectorFun.mult(x,ang))/t[1];
				t[1] = Math.log(t[1]*lambda);
				t[6] = Math.exp(Math.abs(t[4])*t[5]);
				t[1] = Math.exp(-t[1]*t[1]*t[0]);
				this.c[2*(i+a[0])] = t[1]*t[6]*t[2];
				if (Double.isNaN(c[2*(i+a[0])])) c[2*(i+a[0])] = 0;
				this.c[2*(i+a[0])+1] = t[1]*t[6]*t[3]*Math.signum(t[4]);
				if (Double.isNaN(c[2*(i+a[0])+1])) c[2*(i+a[0])+1] = 0;
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