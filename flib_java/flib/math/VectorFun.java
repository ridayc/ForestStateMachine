package flib.math;
import java.lang.Math;
import java.util.Arrays;

public class VectorFun {
	//
	//
	// standard arithmics operations
	//
	//
	// Addition
	// sadly, to keep things quick (using primitive arrays) we need to define all different types of 
	// addition... :(
	// add two arrays
	public static double[] add(final double[] x, final double[] y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]+y[i];
		}
		return z;
	}
	// add a single value to an array
	public static double[] add(final double[] x, double y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]+y;
		}
		return z;
	}
	// the same for integers...
	public static int[] add(final int[] x, final int[] y){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]+y[i];
		}
		return z;
	}
	public static int[] add(final int[] x, int y){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]+y;
		}
		return z;
	}
	// in place operations. These overwrite the the first input vector, otherwise they're
	// the same as above
	public static void addi(final double[] x, final double[] y){
		for (int i=0; i<x.length; i++){
			x[i]+=y[i];
		}
	}
	public static void addi(final double[] x, double y){
		for (int i=0; i<x.length; i++){
			x[i]+=y;
		}
	}
	public static void addi(final int[] x, final int[] y){
		for (int i=0; i<x.length; i++){
			x[i]+=y[i];
		}
	}
	public static void addi(final int[] x, int y){
		for (int i=0; i<x.length; i++){
			x[i]+=y;
		}
	}
	
	//
	// Subtraction
	// sadly, to keep things quick enough we need to define all different types of 
	// subtraction... :(
	// to substract a single value from an array or similar we can just use addition
	// with negative numbers
	// subtract two arrays
	public static double[] sub(final double[] x, final double[] y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]-y[i];
		}
		return z;
	}
	public static double[] sub(double y, final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = y-x[i];
		}
		return z;
	}
	// the same for integers...
	public static int[] sub(final int[] x, final int[] y){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]-y[i];
		}
		return z;
	}
	// in place operations. These overwrite the the first input vector, otherwise they're
	// the same as above
	public static void subi(final double[] x, final double[] y){
		for (int i=0; i<x.length; i++){
			x[i]-=y[i];
		}
	}
	public static void subi(final int[] x, final int[] y){
		for (int i=0; i<x.length; i++){
			x[i]-=y[i];
		}
	}
	
	//
	// Multiplication
	//
	//
	// Multiplication
	// sadly, to keep things quick enough we need to define all different types of 
	// multiplication... :(
	// multiply two arrays
	public static double[] mult(final double[] x, final double[] y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]*y[i];
		}
		return z;
	}
	// multiply a single value times an array
	public static double[] mult(final double[] x, double y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]*y;
		}
		return z;
	}
	// the same for integers...
	public static int[] mult(final int[] x, final int[] y){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]*y[i];
		}
		return z;
	}
	public static int[] mult(final int[] x, int y){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]*y;
		}
		return z;
	}
	// in place operations. These overwrite the the first input vector, otherwise they're
	// the same as above
	public static void multi(final double[] x, final double[] y){
		for (int i=0; i<x.length; i++){
			x[i]*=y[i];
		}
	}
	public static void multi(final double[] x, double y){
		for (int i=0; i<x.length; i++){
			x[i]*=y;
		}
	}
	public static void multi(final int[] x, final int[] y){
		for (int i=0; i<x.length; i++){
			x[i]*=y[i];
		}
	}
	public static void multi(final int[] x, int y){
		for (int i=0; i<x.length; i++){
			x[i]*=y;
		}
	}
	
	//
	// Division
	//
	//
	// Division
	// sadly, to keep things quick enough we need to define all different types of 
	// division... :(
	// divide two arrays
	public static double[] div(final double[] x, final double[] y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]/y[i];
		}
		return z;
	}
	// this could easily be done using multiplication with 1/y
	public static double[] div(final double[] x, double y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]/y;
		}
		return z;
	}
	// the case of division by all values inside the array also needs to be considered
	public static double[] div(double y, final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = y/x[i];
		}
		return z;
	}
	// the same for integers...
	public static int[] div(final int[] x, final int[] y){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]/y[i];
		}
		return z;
	}
	public static int[] div(final int[] x, int y){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]/y;
		}
		return z;
	}
	public static int[] div(int y, final int[] x){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = y/x[i];
		}
		return z;
	}
	// in place operations. These overwrite the the first input vector, otherwise they're
	// the same as above
	public static void divi(final double[] x, final double[] y){
		for (int i=0; i<x.length; i++){
			x[i]/=y[i];
		}
	}
	public static void divi(double y, final double[] x){
		for (int i=0; i<x.length; i++){
			x[i] = y/x[i];
		}
	}
	public static void divi(final int[] x, final int[] y){
		for (int i=0; i<x.length; i++){
			x[i]/=y[i];
		}
	}
	public static void divi(int y, final int[] x){
		for (int i=0; i<x.length; i++){
			x[i] = y/x[i];
		}
	}
	
	//
	// Modulo
	//
	//
	// Modulo
	// sadly, to keep things quick enough we need to define all different types of 
	// modulo operations... :(
	// modulo of two arrays
	public static double[] mod(final double[] x, final double[] y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]%y[i];
		}
		return z;
	}
	public static double[] mod(final double[] x, double y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]%y;
		}
		return z;
	}
	public static double[] mod(double y, final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = y%x[i];
		}
		return z;
	}
	// the same for integers...
	public static int[] mod(final int[] x, final int[] y){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]%y[i];
		}
		return z;
	}
	public static int[] mod(final int[] x, int y){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = x[i]%y;
		}
		return z;
	}
	public static int[] mod(int y, final int[] x){
		int l = x.length;
		int[] z = new int[l];
		for (int i=0; i<l; i++){
			z[i] = y%x[i];
		}
		return z;
	}
	// in place operations. These overwrite the the first input vector, otherwise they're
	// the same as above
	public static void modi(final double[] x, final double[] y){
		for (int i=0; i<x.length; i++){
			x[i]%=y[i];
		}
	}
	
	public static void modi(double[] x, final double y){
		for (int i=0; i<x.length; i++){
			x[i]%=y;
		}
	}
	
	public static void modi(double y, final double[] x){
		for (int i=0; i<x.length; i++){
			x[i] = y%x[i];
		}
	}
	
	public static void modi(final int[] x, final int[] y){
		for (int i=0; i<x.length; i++){
			x[i]%=y[i];
		}
	}
	
	public static void modi(int y, final int[] x){
		for (int i=0; i<x.length; i++){
			x[i] = y%x[i];
		}
	}
	
	//
	// Potentiation
	// sadly, to keep things quick enough we need to define all different types of 
	// potentiation... :(
	// potentiate two arrays
	public static double[] pow(final double[] x, final double[] y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = Math.pow(x[i],y[i]);
		}
		return z;
	}
	// potentiate a single and an array
	public static double[] pow(double y, final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = Math.pow(y,x[i]);
		}
		return z;
	}
	public static double[] pow(final double[] x, double y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = Math.pow(x[i],y);
		}
		return z;
	}
	// in place operations. These overwrite the the first input vector, otherwise they're
	// the same as above
	public static void powi(final double[] x, final double[] y){
		for (int i=0; i<x.length; i++){
			x[i] = Math.pow(x[i],y[i]);
		}
	}
	public static void powi(final double[] x, double y){
		for (int i=0; i<x.length; i++){
			x[i] = Math.pow(x[i],y);
		}
	}
	public static void powi(double y, final double[] x){
		for (int i=0; i<x.length; i++){
			x[i] = Math.pow(y,x[i]);
		}
	}
	
	//
	// Elementary arithmic operations
	//
	// Square Root
	//
	public static double[] sqrt(final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = Math.sqrt(x[i]);
		}
		return z;
	}
	// in place
	public static void sqrti(final double[] x){
		for (int i=0; i<x.length; i++){
			x[i] = Math.sqrt(x[i]);
		}
	}
	
	// Exponentiation with basis e
	//
	public static double[] exp(final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = Math.exp(x[i]);
		}
		return z;
	}
	// in place
	public static void expi(final double[] x){
		for (int i=0; i<x.length; i++){
			x[i] = Math.exp(x[i]);
		}
	}
	
	// Logarithm Naturalis
	//
	public static double[] log(final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = Math.log(x[i]);
		}
		return z;
	}
	// in place
	public static void logi(final double[] x){
		for (int i=0; i<x.length; i++){
			x[i] = Math.log(x[i]);
		}
	}
	
	// Absolute value
	//
	public static double[] abs(final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = Math.abs(x[i]);
		}
		return z;
	}
	// in place
	public static void absi(final double[] x){
		for (int i=0; i<x.length; i++){
			x[i] = Math.abs(x[i]);
		}
	}
	
	// sign (signum) function
	public static double[] sgn(final double[] x){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = Math.signum(x[i]);
		}
		return z;
	}
	
	// Pairwise maximum of two vectors
	public static double[] max(final double[] x, final double[] y){
		int l = x.length;
		double[] m= new double[l];
		for (int i=0; i<l; i++){
			if (x[i]>y[i]){
				m[i] = x[i];
			}
			else {
				m[i] = y[i];
			}
		}
		return m;
	}
	// Pairwise minimum of two vectors
	public static double[] min(final double[] x, final double[] y){
		int l = x.length;
		double[] m= new double[l];
		for (int i=0; i<l; i++){
			if (x[i]<y[i]){
				m[i] = x[i];
			}
			else {
				m[i] = y[i];
			}
		}
		return m;
	}
	
	// arctan
	//
	public static double[] atan(final double[] x, final double[] y){
		int l = x.length;
		double[] z = new double[l];
		for (int i=0; i<l; i++){
			z[i] = Math.atan2(x[i],y[i]);
		}
		return z;
	}
	// in place
	public static void atani(final double[] x, final double[] y){
		for (int i=0; i<x.length; i++){
			x[i] = Math.atan2(x[i],y[i]);
		}
	}
	
	//
	// Global (combining information from different dimensions) vector operations
	//
	// normalize all values to range between 0 and 1 with linear rescaling
	//
	public static double[] norm(final double[] x){
		int l = x.length;
		double[] n = new double[l];
		double xmin = x[0];
		double xmax = x[0];
		for (int i=1; i<l; i++){
			if (x[i]<xmin) {
				xmin = x[i];
			}
			else if (x[i]>xmax){
				xmax = x[i];
			}
		}
		if (xmax==xmin) {
			return null;
		}
		else {
			double m = 1/(double)(xmax-xmin);
			for (int i=0; i<l; i++){
				n[i] = (x[i]-xmin)*m;
			}
			return n;
		}
	}
	// in place
	public static void normi(final double[] x){
		int l = x.length;
		double xmin = x[0];
		double xmax = x[0];
		for (int i=1; i<l; i++){
			if (x[i]<xmin) {
				xmin = x[i];
			}
			else if (x[i]>xmax){
				xmax = x[i];
			}
		}
		if (xmax==xmin) {
			return;
		}
		else {
			double m = 1/(double)(xmax-xmin);
			for (int i=0; i<l; i++){
				x[i] = (x[i]-xmin)*m;
			}
		}
	}
	
	//Max function
	// maximum value, as well as index location of the maximum value
	public static double[] max(final double[] x){
		int l = x.length;
		double[] m = new double[2];
		m[0] = x[0];
		m[1] = 0;
		for (int i=1; i<l; i++){
			if (x[i]>m[0]){
				m[0] = x[i];
				m[1] = i;
			}
		}
		return m;
	}
	
	public static int[] max(final int[] x){
		int l = x.length;
		int[] m = new int[2];
		m[0] = x[0];
		m[1] = 0;
		for (int i=1; i<l; i++){
			if (x[i]>m[0]){
				m[0] = x[i];
				m[1] = i;
			}
		}
		return m;
	}
	
	public static double[] max(final double[][] x){
		int l = x.length;
		double[] m = new double[l];
		for (int i=0; i<l; i++){
			m[i] = max(x[i])[0];
		}
		return m;
	}
	
	//Min function
	// minimum value, as well as index location of the minimum value
	public static double[] min(final double[] x){
		int l = x.length;
		double[] m = new double[2];
		m[0] = x[0];
		m[1] = 0;
		for (int i=0; i<l; i++){
			if (x[i]<m[0]){
				m[0] = x[i];
				m[1] = i;
			}
		}
		return m;
	}
	
	public static int[] min(final int[] x){
		int l = x.length;
		int[] m = new int[2];
		m[0] = x[0];
		m[1] = 0;
		for (int i=0; i<l; i++){
			if (x[i]<m[0]){
				m[0] = x[i];
				m[1] = i;
			}
		}
		return m;
	}
	
	public static double[] min(final double[][] x){
		int l = x.length;
		double[] m = new double[l];
		for (int i=0; i<l; i++){
			m[i] = min(x[i])[0];
		}
		return m;
	}
	
	// Index of the max value
	//
	public static int maxind(final double[] x){
		int l = x.length;
		double m = x[0];
		int ind = 0;
		for (int i=1; i<l; i++){
			if (x[i]>m){
				m = x[i];
				ind = i;
			}
		}
		return ind;
	}
	
	public static double[] maxind(final double[][] x){
		int l = x.length;
		double[] m = new double[l];
		for (int i=0; i<l; i++){
			m[i] = max(x[i])[1];
		}
		return m;
	}
	
	public static int maxind(final int[] x){
		int l = x.length;
		double m = x[0];
		int ind = 0;
		for (int i=0; i<l; i++){
			if (x[i]>m){
				m = x[i];
				ind = i;
			}
		}
		return ind;
	}
	
	// Sum of all vector entries
	//
	public static double sum(final double[] x){
		double sum = 0;
		for (int i=0; i<x.length; i++){
			sum+=x[i];
		}
		return sum;
	}
	
	public static int sum(final int[] x){
		int sum = 0;
		for (int i=0; i<x.length; i++){
			sum+=x[i];
		}
		return sum;
	}
	
	// Dot product
	//
	public static double dot(final double[] x, final double[] y){
		int sum = 0;
		for (int i=0; i<x.length; i++){
			sum+=x[i]*y[i];
		}
		return sum;
	}
	
	// Culmultative sum of vector entries until the current for each point in a vector
	//
	public static double[] cumsum(final double[] x){
		int l = x.length;
		double[] cumsum= new double[l];
		double current = 0;
		for (int i=0; i<l; i++){
			current+=x[i];
			cumsum[i] = current;
		}
		return cumsum;
	}
	
	public static int[] cumsum(final int[] x){
		int l = x.length;
		int[] cumsum= new int[l];
		int current = 0;
		for (int i=0; i<l; i++){
			current+=x[i];
			cumsum[i] = current;
		}
		return cumsum;
	}
	
	// Culmultative multiplication of vector entries until the current for the each point in a vector
	//
	public static double[] cummult(final double[] x){
		int l = x.length;
		double[] cummult= new double[l];
		double current = 1;
		for (int i=0; i<l; i++){
			current*=x[i];
			cummult[i] = current;
		}
		return cummult;
	}
	
	public static int[] cummult(final int[] x){
		int l = x.length;
		int[] cummult= new int[l];
		int current = 1;
		for (int i=0; i<l; i++){
			current*=x[i];
			cummult[i] = current;
		}
		return cummult;
	}
	
	//
	// matrix operations
	//
	
	public static double[][] abs(final double[][] x){
		double[][] y = new double[x.length][x[0].length];
		for (int i=0; i<x.length; i++){
			y[i] = VectorFun.abs(x[i]);
		}
		return y;
	}
	
	public static double[][] sgn(final double[][] x){
		double[][] y = new double[x.length][x[0].length];
		for (int i=0; i<x.length; i++){
			y[i] = VectorFun.sgn(x[i]);
		}
		return y;
	}
	
	public static double[][] mult(final double[][] x, final double[][] y){
		double[][] z = new double[x.length][x[0].length];
		for (int i=0; i<x.length; i++){
			z[i] = VectorFun.mult(x[i],y[i]);
		}
		return z;
	}
	
	public static double[][] mult(final double[][] x, double y){
		double[][] z = new double[x.length][x[0].length];
		for (int i=0; i<x.length; i++){
			z[i] = VectorFun.mult(x[i],y);
		}
		return z;
	}
	
	public static double[][] add(final double[][] x, final double[][] y){
		double[][] z = new double[x.length][x[0].length];
		for (int i=0; i<x.length; i++){
			z[i] = VectorFun.add(x[i],y[i]);
		}
		return z;
	}
	
	public static double[][] add(final double[][] x, double y){
		double[][] z = new double[x.length][x[0].length];
		for (int i=0; i<x.length; i++){
			z[i] = VectorFun.add(x[i],y);
		}
		return z;
	}
	
	public static double[] rowSum(final double[][] x){
		int l = x.length;
		double[] m = new double[l];
		for (int i=0; i<l; i++){
			m[i] = sum(x[i]);
		}
		return m;
	}
	
	public static double[] columnSum(final double[][] x){
		int l = x.length;
		int n = x[0].length;
		double[] m = new double[n];
		for (int i=0; i<l; i++){
			for (int j=0; j<n; j++){
				m[j]+=x[i][j];
			}
		}
		return m;
	}
	
	// for sorted arrays!
	public static int binarySearch(double[] a, double k){
		int b = Arrays.binarySearch(a,k);
		if (b<0){
			b = -b-1;
		}
		return b;
	}
	
	public static int binarySearch(int[] a, int k){
		int b = Arrays.binarySearch(a,k);
		if (b<0){
			b = -b-1;
		}
		return b;
	}
	
	public static int weightSearch(double[] a, double k){
		int n = a.length/2;
		int b = Arrays.binarySearch(a,n,2*n,k);
		if (b<0){
			b = -b-1;
		}
		return b-n;
	}
}
				