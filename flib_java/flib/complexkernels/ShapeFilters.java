package flib.complexkernels;

import flib.math.VectorFun;

public class ShapeFilters {
	
	public static double[] rectFilter(final int[] s, final double[] w, final double[] c){
		double[] mw = new double[s.length];
		int l = 1;
		for (int i=0; i<s.length; i++){
			l*=s[i];
			// center the coordinate system
			mw[i] = s[i]*0.5;
		}
		double[] x = new double[s.length];
		int[] a = new int[s.length];
		double[] r = new double[l];
		nestedLoopRect(s.length-1,r,s,w,c,x,a,VectorFun.cummult(s),mw);
		return VectorFun.mult(r,1/VectorFun.sum(VectorFun.abs(r)));
	}
	
	public static double[] nSphereFilter(final int[] s, double r, final double[] c){
		double[] mw = new double[s.length];
		int l = 1;
		for (int i=0; i<s.length; i++){
			l*=s[i];
			// center the coordinate system
			mw[i] = s[i]*0.5;
		}
		double[] x = new double[s.length];
		int[] a = new int[s.length];
		double[] ns = new double[l];
		double r2 = r*r;
		nestedLoopNSphere(s.length-1,ns,s,r2,c,x,a,VectorFun.cummult(s),mw);
		return VectorFun.mult(ns,1/VectorFun.sum(VectorFun.abs(ns)));
	}

	private static void nestedLoopRect(int l, double[] r, final int[] s, double[] w, final double[] c, double[] x, int[] a, final int[] wcm, final double[] mw){
		// inner most for loop
		// this hopefully takes more time than the function call for the recursive 
		// for loop
		if (l<1){
			for (int i=0; i<s[0]; i++){
				x[0] = (i-mw[0]);
				r[a[0]+i] = 1;
				// calculate the radial part taking into consideration unequal image proportions
				for (int j=0; j<w.length; j++){
					if (x[j]>c[j]+w[j]||x[j]<c[j]-w[j]){
						r[a[0]+i] = 0;
					}
				}
			}
		}
		// apply the recursion
		else {
			for (int i=0; i<s[l]; i++){
				// overwrite the lower level index offset
				a[l-1] = a[l];
				// set variables that depend on the value of the current level
				x[l] = (i-mw[l]);
				// apply the recursion
				nestedLoopRect(l-1,r,s,w,c,x,a,wcm,mw);
				// index offset due to the current level
				a[l]+=wcm[l-1];
			}
		}
	}
	
	private static void nestedLoopNSphere(int l, double[] ns, final int[] s, double r2, final double[] c, double[] x, int[] a, final int[] wcm, final double[] mw){
		// inner most for loop
		// this hopefully takes more time than the function call for the recursive 
		// for loop
		if (l<1){
			for (int i=0; i<s[0]; i++){
				x[0] = (i-mw[0]-c[0]);
				ns[a[0]+i] = 1;
				// calculate the radial part taking into consideration unequal image proportions
				if (VectorFun.sum(VectorFun.mult(x,x))>r2){
					ns[a[0]+i] = 0;
				}
			}
		}
		// apply the recursion
		else {
			for (int i=0; i<s[l]; i++){
				// overwrite the lower level index offset
				a[l-1] = a[l];
				// set variables that depend on the value of the current level
				x[l] = (i-mw[l]-c[l]);
				// apply the recursion
				nestedLoopNSphere(l-1,ns,s,r2,c,x,a,wcm,mw);
				// index offset due to the current level
				a[l]+=wcm[l-1];
			}
		}
	}
}