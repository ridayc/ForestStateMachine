package flib.fftfunctions;

import flib.math.VectorFun;

public class FFTWrapper {
	// Functions which can be required when applying a double valued FFT
	
	// Forward FFT shift
	// This contains a recentering and flipping of the input data
	// w: width, h: height, z: number of slices
	
	public static double[] fftshiftn(final int[] w, final double[] x){
		int l = w.length;
		int[] midw = new int[l];
		for (int i=0; i<l; i++){
			midw[i] = (w[i]-1)/2;
		}
		int[] wcm = VectorFun.cummult(w);
		double[] shift = new double[wcm[l-1]];
		nestedLoop_fftshift(l-1,x,shift,w,midw,new int[l],new int[l],wcm);
		return shift;
	}
	
	private static void nestedLoop_fftshift(int l, final double[] x, double[] shift, final int[] w, final int[] midw, int[] a, int[] b, final int[] wcm){
		// inner most for loop
		// this hopefully takes more time than the function call for the recursive 
		// for loop
		if (l<1){
			int k = 0;
			for (int i=midw[0]+1; i<w[0]; i++){
				shift[b[0]+k] = x[a[0]+i];
				k++;
			}
			for (int i=0; i<midw[0]+1; i++){
				shift[b[0]+k] = x[a[0]+i];
				k++;
			}
		}
		// apply the recursion
		else {
			a[l]+=wcm[l-1]*(midw[l]+1);
			for (int i=midw[l]+1; i<w[l]; i++){
				a[l-1] = a[l];
				b[l-1] = b[l];
				nestedLoop_fftshift(l-1, x,shift,w,midw,a,b,wcm);
				a[l]+=wcm[l-1];
				b[l]+=wcm[l-1];
			}
			a[l]-=wcm[l];
			for (int i=0; i<midw[l]+1; i++){
				a[l-1] = a[l];
				b[l-1] = b[l];
				nestedLoop_fftshift(l-1, x,shift,w,midw,a,b,wcm);
				a[l]+=wcm[l-1];
				b[l]+=wcm[l-1];
			}
		}
	}
	
	// 1D
	public static double[] fftshift1(int w, final double[] x){
		return fftshiftn(new int[]{w},x);
	}
	
	// 2D
	public static double[] fftshift2(int w, int h, final double[] x){
		return fftshiftn(new int[]{w,h},x);
	}
	
	// 3D
	public static double[] fftshift3(int w, int h, int z, final double[] x){
		return fftshiftn(new int[]{w,h, z},x);
	}
	
	public static double[] ifftshiftn(final int[] w, final double[] x){
		int l = w.length;
		int[] midw = new int[l];
		for (int i=0; i<l; i++){
			midw[i] = w[i]/2;
		}
		int[] wcm = VectorFun.cummult(w);
		double[] shift = new double[wcm[l-1]];
		nestedLoop_ifftshift(l-1,x,shift,w,midw,new int[l],new int[l],wcm);
		return shift;
	}
	
	private static void nestedLoop_ifftshift(int l, final double[] x, double[] shift, final int[] w, final int[] midw, int[] a, int[] b, final int[] wcm){
		// inner most for loop
		// this hopefully takes more time than the function call for the recursive 
		// for loop
		if (l<1){
			int k = 0;
			for (int i=midw[0]; i<w[0]; i++){
				shift[b[0]+k] = x[a[0]+i];
				k++;
			}
			for (int i=0; i<midw[0]; i++){
				shift[b[0]+k] = x[a[0]+i];
				k++;
			}
		}
		// apply the recursion
		else {
			a[l]+=wcm[l-1]*(midw[l]);
			for (int i=midw[l]; i<w[l]; i++){
				a[l-1] = a[l];
				b[l-1] = b[l];
				nestedLoop_ifftshift(l-1, x,shift,w,midw,a,b,wcm);
				a[l]+=wcm[l-1];
				b[l]+=wcm[l-1];
			}
			a[l]-=wcm[l];
			for (int i=0; i<midw[l]; i++){
				a[l-1] = a[l];
				b[l-1] = b[l];
				nestedLoop_ifftshift(l-1, x,shift,w,midw,a,b,wcm);
				a[l]+=wcm[l-1];
				b[l]+=wcm[l-1];
			}
		}
	}
	
	// Backward FFT shift
	
	// 1D
	public static double[] ifftshift1(int w, final double[] x){
		return ifftshiftn(new int[]{w},x);
	}
	
	// 2D
	public static double[] ifftshift2(int w, int h, final double[] x){
		return ifftshiftn(new int[]{w,h},x);
	}
	
	// 3D
	public static double[] ifftshift3(int w, int h, int z, final double[] x){
		return ifftshiftn(new int[]{w,h, z},x);
	}
	
	/* padding of a vector representing 2D data. 
	w: width of the image
	h: height of the image
	dim: dimension along which the image should be expanded. 0 in x direction. 1 in y direction)
	width: number of pixels to be added along the image side
	type: indicates which side the pixels should be added on. Options are left (top), right (bottom) and both.
	*/
	
	public static double[] padn(final int[] w, final double[] x, final int[] left, final int[] right, double value){
		int l = w.length;
		int[] w2 = new int[l];
		for (int i=0; i<l; i++){
			w2[i] = w[i]+left[i]+right[i];
		}
		int[] wcm = VectorFun.cummult(w);
		int[] wcm2 = VectorFun.cummult(w2);
		double[] padded = VectorFun.add(new double[wcm2[l-1]],value);
		nestedLoop_padn(l-1,x,padded,w,new int[l],new int[l],left,right,wcm,wcm2);
		return padded;
	}		

	private static void nestedLoop_padn(int l, final double[] x, double[] padded, final int[] w, int[] a, int[] b, final int[] left, final int[] right, final int[] wcm, final int[] wcm2){
		// inner most for loop
		// this hopefully takes more time than the function call for the recursive 
		// for loop
		if (l<1){
			int k = 0;
			b[0]+=left[0];
			for (int i=0; i<w[0]; i++){
				padded[b[0]+i] = x[a[0]+i];
			}
		}
		// apply the recursion
		else {
			b[l]+=wcm2[l-1]*left[l];
			for (int i=0; i<w[l]; i++){
				a[l-1] = a[l];
				b[l-1] = b[l];
				nestedLoop_padn(l-1, x,padded,w,a,b,left,right,wcm,wcm2);
				a[l]+=wcm[l-1];
				b[l]+=wcm2[l-1];
			}
		}
	}
	
	public static double[] padn(final int[] w, final double[] x, final int[] left, double value){
		return padn(w,x,left,left,value);
	}
	
	public static double[] padn(final int[] w, final double[] x, final int[] left, final int[] right){
		return padn(w,x,left,right,0);
	}
	
	public static double[] padn(final int[] w, final double[] x, final int[] left){
		return padn(w,x,left,left,0);
	}
	
	public static double[] pad2(final int[] w, final double[] x, double value){
		return padn(w,x,new int[]{1,1},new int[]{1,1},value);
	}
	
	public static double[] pad2(final int[] w, final double[] x){
		return padn(w,x,new int[]{1,1},new int[]{1,1},0);
	}
	
	public static double[] pad3(final int[] w, final double[] x, double value){
		return padn(w,x,new int[]{1,1,1},new int[]{1,1,1},value);
	}
	
	public static double[] pad3(final int[] w, final double[] x){
		return padn(w,x,new int[]{1,1,1},new int[]{1,1,1},0);
	}
	
	public static double[] flip(final double[] x){
		int l = x.length-1;
		double[] flipped = new double[l+1];
		for (int i=0; i<l+1; i++){
			flipped[i] = x[l-i];
		}
		return flipped;
	}
}

