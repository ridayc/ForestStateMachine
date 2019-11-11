package flib.algorithms.graph;

import java.util.ArrayList;
import java.util.Arrays;
import flib.fftfunctions.FFTWrapper;

public class NeighborConnectivity {
	public static double[][] neighbors(int w, int h, final double[] x, int connectivity, double borderval){
		int w2 = w+2;
		int h2 = h+2;
		int n = x.length;
		double[] im = FFTWrapper.pad2(new int[]{w,h},x,borderval);
		double neighborval[][];
		int[] d;
		if (connectivity==4){
			neighborval = new double[n][4];
			d = new int[4];
			d[0] = -1;
			d[1] = 1;
			d[2] = -w2;
			d[3] = w2;
		}
		else {
			connectivity = 8;
			neighborval = new double[n][8];
			d = new int[8];
			d[0] = -1;
			d[1] = 1;
			d[2] = -w2;
			d[3] = w2;
			d[4] = -w2-1;
			d[5] = -w2+1;
			d[6] = w2-1;
			d[7] = w2+1;
		}
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				for (int k=0; k<connectivity; k++){
					neighborval[(j-1)*w+(i-1)][k] = im[j*w2+i+d[k]];
				}
			}
		}
		return neighborval;
	}
	
	// neighborval is expected to be of the form double[n][8]
	public static double[][] breakDiagonals(final double[][] neighborval){
		int n = neighborval.length;
		double[][] newneighbors = new double[n][8];
		for (int i=0; i<n; i++){
			for (int j=0; j<8; j++){
				newneighbors[i][j] = neighborval[i][j];
			}
		}
		for (int i=0; i<n; i++){
			if (newneighbors[i][0]==1){
				newneighbors[i][4] = 0;
				newneighbors[i][6] = 0;
			}
			if (newneighbors[i][1]==1){
				newneighbors[i][5] = 0;
				newneighbors[i][7] = 0;
			}
			if (newneighbors[i][2]==1){
				newneighbors[i][4] = 0;
				newneighbors[i][5] = 0;
			}
			if (newneighbors[i][3]==1){
				newneighbors[i][6] = 0;
				newneighbors[i][7] = 0;
			}
		}
		return newneighbors;
	}
	
	public static int[] sumNeighbors(final double[][] x){
		int n = x.length;
		int d = x[0].length;
		int[] nn = new int[n];
		for (int i=0; i<n; i++){
			for (int j=0; j<d; j++){
				if (x[i][j]>0){
					nn[i]++;
				}
			}
		}
		return nn;
	}			
	
	public static ArrayList<ArrayList<Double>> uniqueNeighbors(final double[][]neighborval){
		int n = neighborval.length;
		//int dim = neighborval[0].length;
		ArrayList<ArrayList<Double>> uniqueN = new ArrayList<ArrayList<Double>>(n);
		double[] tempvec;
		for (int i=0; i<n; i++){
			uniqueN.add(new ArrayList<Double>(8));
			tempvec = neighborval[i].clone();
			Arrays.sort(tempvec);
			uniqueN.get(i).add(tempvec[0]);
			for (int j=1; j<tempvec.length; j++){
				if (tempvec[j]>tempvec[j-1]){
					uniqueN.get(i).add(tempvec[j]);
				}
			}
		}
		return uniqueN;
	}
	
	public static ArrayList<ArrayList<Double>> uniqueNonNegNeighbors(final double[][]neighborval){
		int n = neighborval.length;
		//int dim = neighborval[0].length;
		ArrayList<ArrayList<Double>> uniqueN = new ArrayList<ArrayList<Double>>(n);
		double[] tempvec;
		for (int i=0; i<n; i++){
			uniqueN.add(new ArrayList<Double>(8));
			tempvec = neighborval[i].clone();
			Arrays.sort(tempvec);
			if (tempvec[0]>0){
				uniqueN.get(i).add(tempvec[0]);
			}
			for (int j=1; j<tempvec.length; j++){
				if (tempvec[j]>tempvec[j-1]&&tempvec[j]>0){
					uniqueN.get(i).add(tempvec[j]);
				}
			}
		}
		return uniqueN;
	}
}