package flib.math;
import java.lang.Math;

public class ComplexMath {
	// Functions for complex arrays. These are double arrays with each pair of
	// subsequent numbers corresponding to the real and complex parts of 
	// a single complex number
	
	// create a complex array from two double arrays
	public static double[] complexVector(final double[] x, final double[] y){
		int l = x.length;
		double[] c = new double[2*l];
		for (int i=0; i<l; i++) {
			c[i*2] = x[i];
			c[i*2+1] = y[i];
		}
	    return c;
	}
	
	// create a complex array from a single vector, assuming the vector has
	// all imaginary parts equal to zero
	public static double[] complexVector(final double[] x){
		int l = x.length;
		double[] c = new double[2*l];
		for (int i=0; i<l; i++) {
			c[2*i] = x[i];
			c[2*i+1] = 0;
		}
		return c;
	}
	
	// get the vector of real parts from a complex vector
	public static double[] getReal(final double[] c){
		int l = c.length;
		double[] x = new double[l/2];
		for (int i=0; i<l/2; i++) {
			x[i] = c[i*2];
		}
		return x;
	}
	
	// get the vector of imaginary parts from a complex vector
	public static double[] getComplex(final double[] c){
		int l = c.length;
		double[] y = new double[l/2];
		for (int i=0; i<l/2; i++) {
			y[i] = c[i*2+1];
		}
		return y;
	}
	
	// get the vector of absolute values from a complex vector
	public static double[] getAbs(final double [] c){
		int l =  c.length;
		double[] a = new double[l/2];
		for(int i=0; i<l/2; i++){
			a[i] = c[2*i]*c[2*i]+c[2*i+1]*c[2*i+1];
		}
		return a;
	}
	
	// get the vector of phase values from a complex vector
	public static double[] getPhase(final double [] c){
		int l =  c.length;
		double[] p = new double[l/2];
		for(int i=0; i<l/2; i++){
			p[i] = Math.atan2(c[2*i+1],c[2*i]);
		}
		return p;
	}
	
	// multiply two complex vectors with one another
	static public double[] complexMult(final double[] a, final double[] b) {
		if (a.length!=b.length) {
			return null;
		}
		double[] c = new double[a.length];
		for (int i=0; i<a.length/2;i++) {
			c[2*i] = a[2*i]*b[2*i]-a[2*i+1]*b[2*i+1];
			c[2*i+1] = a[2*i+1]*b[2*i]+a[2*i]*b[2*i+1];
		}
		return c;
	}
	
	// multiply in place of the first vector two complex vectors with one another
	static public void complexMulti(double[] a, final double[] b) {
		if (a.length!=b.length) {
			return;
		}
		double c,d;
		for (int i=0; i<a.length/2;i++) {
			c = a[2*i]*b[2*i]-a[2*i+1]*b[2*i+1];
			d = a[2*i+1]*b[2*i]+a[2*i]*b[2*i+1];
			a[2*i] = c;
			a[2*i+1] = d;
		}
	}
}