package flib.math;

public class BV {
	// A class for boolean[] comparison of vectors
	//as well as boolean[] operations for boolean[] vectors
	
	//
	// boolean[] comparators
	//
	// Greater than
	//
	public static boolean[] gt(final double[] x, final double[] y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]>y[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] gt(final double[] x, double y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]>y){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] gt(double y, final double[] x){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (y>x[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	// Lesser than
	//
	public static boolean[] lt(final double[] x, final double[] y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]<y[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] lt(final double[] x, double y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]<y){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] lt(double y, final double[] x){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (y<x[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	// Greater or equal
	//
	public static boolean[] gte(final double[] x, final double[] y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]>=y[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] gte(final double[] x, double y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]>=y){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] gte(double y, final double[] x){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (y>=x[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	// Lesser or equal
	//
	public static boolean[] lte(final double[] x, final double[] y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]>=y[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] lte(final double[] x, double y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]>=y){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] lte(double y, final double[] x){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (y>=x[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	// Equal
	//
	public static boolean[] eq(final double[] x, final double[] y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]==y[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] eq(final double[] x, double y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]==y){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] eq(double y, final double[] x){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (y==x[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	// Not equal
	//
	public static boolean[] neq(final double[] x, final double[] y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]!=y[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] neq(final double[] x, double y){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (x[i]!=y){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	public static boolean[] neq(double y, final double[] x){
		int l = x.length;
		boolean[] b = new boolean[l];
		for (int i=0; i<l; i++){
			if (y!=x[i]){
				b[i] = true;
			}
			else {
				b[i] = false;
			}
		}
		return b;
	}
	
	//
	// boolean[] operators
	//
	// and
	//
	public static boolean[] and(final boolean[] a, final boolean[] b){
		int l = a.length;
		boolean[] x = new boolean[l];
		for (int i=0; i<l; i++){
			if (a[i]&&b[i]){
				x[i] = true;
			}
			else {
				x[i] = false;
			}
		}
		return x;
	}
	
	// or
	//
	public static boolean[] or(final boolean[] a, final boolean[] b){
		int l = a.length;
		boolean[] x = new boolean[l];
		for (int i=0; i<l; i++){
			if (a[i]||b[i]){
				x[i] = true;
			}
			else {
				x[i] = false;
			}
		}
		return x;
	}
	
	// invert
	//
	public static boolean[] invert(final boolean[] a){
		int l = a.length;
		boolean[] x = new boolean[l];
		for (int i=0; i<l; i++){
			x[i] = !a[i];
		}
		return x;
	}
}