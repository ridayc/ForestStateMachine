package flib.math;

public class UniqueSorted {
	static public int[] unique(final double[] x){
		// number of levels
		int n = 1;
		for (int i=1; i<x.length; i++){
			if (x[i]>x[i-1]){
				n++;
			}
		}
		int[] count = new int[n];
		n = 0;
		for (int i=1; i<x.length; i++){
			if (x[i]>x[i-1]){
				count[n] =i;
				n++;
			}
		}
		count[n] = x.length;
		
		return count;
	}
	
	static public int[] unique(final int[] x){
		// number of levels
		int n = 1;
		for (int i=1; i<x.length; i++){
			if (x[i]>x[i-1]){
				n++;
			}
		}
		int[] count = new int[n];
		n = 0;
		for (int i=1; i<x.length; i++){
			if (x[i]>x[i-1]){
				count[n] =i;
				n++;
			}
		}
		count[n] = x.length;
		
		return count;
	}
}