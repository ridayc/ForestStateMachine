package flib.complexkernels;

import java.lang.Math;
import flib.complexkernels.ConvertKernel;
import flib.math.VectorFun;

public class DirectionalLogGabor{
	private int[] w;
	private double lambda;
	private double sigma;
	private int dim;
	
	// representation of the filter in a complex double vector
	private double[] c;
	
	public DirectionalLogGabor(final int[] w, double lambda, double sigma, int dim) {
		this.w = w.clone();
		this.lambda = lambda;
		this.sigma = sigma;
		this.dim = dim;
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
		
	public int getDim(){
		return this.dim;
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
	
	public void setLambda(int dim) {
		this.dim = dim;
		this.generate();
	}
	
	// log Gabor filter generation
	private void generate() {
		int l = VectorFun.cummult(w)[w.length-1];
		double mw = w[dim]*0.5;
		double wf = 2.0/w[dim];
		// constant variables to save calculation time
		double[] x = new double[w[dim]];
		int[] a = new int[w.length];
		double[] t = new double[2];
		t[0] = Math.PI*Math.PI/(2*Math.pow(Math.log(2)*this.sigma,2));
		// allocate memory for the filter
		this.c = new double[l*2];
		for (int i=0; i<w[dim]; i++){
			double y = (i-mw)*wf;
			x[i] = Math.log(Math.abs(y)*lambda);
			x[i] = Math.exp(-x[i]*x[i]*t[0]);
			if (Double.isNaN(x[i])) x[i] = 0;
		}
		this.nestedLoop(this.w.length-1,x,t,a,VectorFun.cummult(w));
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
	private void nestedLoop(int l, double[] x, double[] t, int[] a, final int[] wcm){
		// inner most for loop
		// this hopefully takes more time than the function call for the recursive 
		// for loop
		if (l<1){
			for (int i=0; i<this.w[0]; i++){
				if (l==dim){
					this.c[2*(i+a[0])] = x[i];
				}
				else {
					this.c[2*(i+a[0])] = t[1];
				}
			}
		}
		// apply the recursion
		else {
			for (int i=0; i<this.w[l]; i++){
				// overwrite the lower level index offset
				a[l-1] = a[l];
				// set variables that depend on the value of the current level
				if (l==dim){
					t[1] = x[i];
				}
				// apply the recursion
				nestedLoop(l-1, x, t, a, wcm);
				// index offset due to the current level
				a[l]+=wcm[l-1];
			}
		}
	}
}