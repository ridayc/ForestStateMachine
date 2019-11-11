package flib.algorithms;

import flib.fftfunctions.FFTWrapper;
import flib.math.VectorFun;

public class RidgeDetection {
	public static int[] ridgeDetection(final double[] x, int w, int h){
		int l = x.length;
		// the map indicating if a point belongs to a ridge or not
		int[] y = new int[l];
		int w2 = w+2;
		int h2 = h+2;
		double[] x2 = FFTWrapper.pad2(new int[]{w,h},x,Double.MAX_VALUE);
		// neighbor difference vector
		double[] d = new double[4];
		int a,b;
		int sum = 0;
		// go through all pixels and compare with their four adjacent neighbors
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				a = j*w2+i;
				// pixel neighbor difference
				d[0] = x2[a]-x2[a+1];
				d[1] = x2[a]-x2[a-1];
				d[2] = x2[a]-x2[a+w2];
				d[3] = x2[a]-x2[a-w2];
				b = (j-1)*w+(i-1);
				// check that the center is at least equal to its neighbors
				// in left-right or top-down direction
				if ((d[0]>0&&d[1]>0)||(d[2]>0&&d[3]>0)){
					sum = 4;
					if ((d[0]>0&&d[1]>0)&&(d[2]>0&&d[3]>0)){
						sum =6;
					}
					y[b] = sum;
				}
				else if (d[0]>=0&&d[1]>=0){
					sum = 1;
					if (d[0]==0&&d[1]==0){
						if (d[2]>=0&&d[3]>=0){
							sum = 2;
						}
					}
					else if (d[2]>=0||d[3]>=0){
						sum = 3;
					}
					y[b] = sum;
				}
				else if (d[2]>=0&&d[3]>=0){
					sum = 1;
					if (d[2]==0&&d[3]==0){
						if (d[0]>=0&&d[1]>=0){
							sum = 2;
						}
					}
					else if (d[0]>=0||d[1]>=0){
						sum = 3;
					}
					y[b] = sum;
				}
			}
		}
		return y;
	}
}