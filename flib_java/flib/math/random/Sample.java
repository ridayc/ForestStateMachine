package flib.math.random;

import java.util.Random;
import java.util.Arrays;
import flib.math.VectorFun;

public class Sample{
	
	public static int sample(final double[] x, double r){
		double[] sum = VectorFun.cumsum(x);
		sum = VectorFun.div(x,x[x.length-1]);
		int a = Arrays.binarySearch(sum,r);
		if (a<0){
			a = -(a+1);
		}
		return a;
	}
	
	public static int sample(final double[] x, final Random rng){
		double r = rng.nextDouble();
		return sample(x,r);
	}
	
	public static int sample(final double[] x){
		return sample(x,new Random());
	}
	
	public static int[] sample(final double[][] x, final Random rng){
		int l = x.length;
		int[] val = new int[l];
		for (int i=0; i<l; i++){
			val[i] = sample(x[i],rng.nextDouble());
		}
		return val;
	}
	
	public static int[] sample(final double[][] x){
		return sample(x,new Random());
	}
	
	public static int[] sample(int n, int nump, final Random rng){
		int[] x = new int[nump];
		for (int i=0; i<nump; i++){
			x[i] = rng.nextInt(n);
		}
		return x;
	}
	
	public static int[] sample(int n, int nump){
		return sample(n, nump, new Random());
	}
	
	public static double[][] addNoise(final double[][] x, double r){
		Random rng = new Random();
		double[][] y = new double[x.length][];
		for (int i=0; i<x.length; i++){
			y[i] = new double[x[i].length];
			for (int j=0; j<x[i].length; j++){
				y[i][j] = x[i][j]+r*rng.nextDouble();
			}
		}
		return y;
	}
	
	public static double[][] addNoise(final double[][] x, final double[] r){
		Random rng = new Random();
		double[][] y = new double[x.length][];
		for (int i=0; i<x.length; i++){
			y[i] = new double[x[i].length];
			for (int j=0; j<x[i].length; j++){
				y[i][j] = x[i][j]+r[j]*rng.nextDouble();
			}
		}
		return y;
	}
	
	public static double[] randomUnitVector(int n, final Random rng){
		double[] vec = new double[n];
		double a;
		double sum = 0;
		for (int i=0; i<n; i++){
			a = rng.nextGaussian();
			if (a>0){
				vec[i] = a;
			}
			else {
				vec[i] = -a;
			}
			sum+=vec[i]*vec[i];
		}
		for (int i=0; i<n; i++){
			vec[i]/=sum;
		}
		return vec;
	}
	
	public static double[] randomUnitVector(int n){
		return randomUnitVector(n, new Random());
	}
	
	public static double[] randomUnitVectorL1(int n, final Random rng){
		double[] vec = new double[n];
		double a;
		double sum = 0;
		for (int i=0; i<n; i++){
			a = rng.nextGaussian();
			if (a>0){
				vec[i] = a;
			}
			else {
				vec[i] = -a;
			}
			sum+=vec[i];
		}
		for (int i=0; i<n; i++){
			vec[i]/=sum;
		}
		return vec;
	}
	public static double[] randomUnitVectorL1(int n){
		return randomUnitVectorL1(n, new Random());
	}
}