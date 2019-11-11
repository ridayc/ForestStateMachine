package flib.algorithms.clustering;

import java.util.TreeSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.lang.Math;
import flib.math.RankSort;
import flib.math.SortPair;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.random.Shuffle;
import flib.algorithms.randomforest.splitfunctions.SortingFunctions;


// this is the first half of a function for dimension reduction
// the goal is to find dimensions or sub-dimensions which are rank-wise well aligned with the 
// original input dimensions
// we use a lot of dimension swapping of a set of permutation matrices to achieve local
// convergence
public class PermClustering {
	
	public static void ordering(final double[][] trainingset, int[][] orderset, final boolean[] categorical, final double[] weights, int sigma, double sigma2, int maxit){
		// trainingset: the initial training values. We only need this to extract the rank orderings of the original dimensions
		// orderset: the initial rank ordering of the output dimensions for the compression algorithm
		// categorical: are the initial dimensions categorical or not
		// weights: weights of the dimensions for the evaluation process
		// sigma: neighborhood size to be considered for performance
		// sigma2: power factor for the error type. A factor of two corresponds to standard squared error
		// maxit: maximal number of iterations if there's no convergence
		// number of points in the set
		int n = trainingset.length;
		// number of input dimensions
		int d = trainingset[0].length;
		// number of compression dimensions
		int nclust = orderset.length;
		// traingset ordering
		int[][] ir = new int[d][n];
		for (int i=0; i<d; i++){
			double[] val = new double[n];
			for (int j=0; j<n; j++){
				val[j] = trainingset[j][i];
			}
			ir[i] = (new RankSort(val)).getRank();
		}
		// copy of the output ordering to check if the ordering has changed
		int[][] orderset_old = new int[nclust][n];
		for (int i=0; i<nclust; i++){
			//orderset[i] = Shuffle.randPerm(n);
			orderset_old[i] = orderset[i].clone();
		}
		// counter for points that share a neighborhood
		int[][][] dim_count = new int[nclust][d][n];
		int[][][] dim_count2 = new int[nclust][d][n];
		// neighbors for each point for each dimension
		int[][][] neighborlist = new int[d][n][2*sigma];
		ArrayList<ArrayList<TreeSet<SortPair>>> dm_out = new ArrayList<ArrayList<TreeSet<SortPair>>>();
		// we use another sigma to get the neighbors...
		neighbors(neighborlist,ir,trainingset,categorical,sigma);
		counter(neighborlist,dim_count,dim_count2,orderset,dm_out,sigma);
		// now comes the main algorithm with swapping fun!
		for (int j=0; j<maxit; j++){
			if(!update(weights,orderset,orderset_old,neighborlist,dim_count,dim_count2,dm_out,sigma,sigma2)){
				double gap = 0;
				for (int k=0; k<d; k++){
					double t = 0;
					for (int l=0; l<n; l++){
						t+=Math.pow(2*sigma-dm_out.get(k).get(l).last().getValue(),sigma2);
					}
					gap+=t*weights[k];
				}
				gap = Math.pow(gap/(VectorFun.sum(weights)*n),1/sigma2);
				System.out.println("Iteration: "+Integer.toString(j));
				System.out.println("Gap Sum: "+Double.toString(gap));
				System.out.println("Finished the permutation clustering after "+Integer.toString(j)+" iterations");
				break;
			}
			double gap = 0;
			for (int k=0; k<d; k++){
				double t = 0;
				for (int l=0; l<n; l++){
					t+=Math.pow(2*sigma-dm_out.get(k).get(l).last().getValue(),sigma2);
				}
				gap+=t*weights[k];
			}
			gap = Math.pow(gap/(VectorFun.sum(weights)*n),1/sigma2);
			System.out.println("Iteration: "+Integer.toString(j));
			System.out.println("Gap Sum: "+Double.toString(gap));
		}
	}
	
	// the final isn't valid for orderset and dim_count2
	private static boolean update(final double[] weights, final int[][] orderset, int[][] orderset_old, final int[][][] neighborlist, int[][][] dim_count, final int[][][] dim_count2,ArrayList<ArrayList<TreeSet<SortPair>>> dm_out, final int sigma, final double sigma2){
		final int n = orderset[0].length;
		final int nclust = orderset.length;
		int d = neighborlist.length;
		final double[][] max_count = new double[d][n];
		final double[][] max_count2 = new double[d][n];
		for (int i=0; i<d; i++){
			for (int j=0; j<n; j++){
				SortPair sp = dm_out.get(i).get(j).last();
				max_count[i][j] = Math.pow(2*sigma-sp.getValue(),sigma2);
				max_count2[i][j] = Math.pow(2*sigma-dm_out.get(i).get(j).lower(sp).getValue(),sigma2);
			}
		}
		int counter = 0;
		final int n2 = n*n;
		// hopefully we can shuffle... the list is quite large...
		int[] rp = Shuffle.randPerm(n*nclust);
		for (int i=0; i<n*nclust; i++){
			final int x = rp[i]%n;
			final int z = rp[i]/n;
			int ox = orderset[z][x];
			final double[] err = new double[n];
			final int NUM_CORES = Runtime.getRuntime().availableProcessors();
			ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
			try {
				for (int j=0; j<NUM_CORES; j++){
					final int[][] dim_copy = new int[d][n];
					for (int k=0; k<d; k++){
						dim_copy[k] = dim_count[z][k].clone();
					}
					final int j2 = j;
					exec.submit(new Runnable() {
						@Override
						public void run(){
							for (int k=j2; k<n; k+=NUM_CORES){
								if (x!=k){
									err[k] = swap_test(x,k,z,orderset[z],neighborlist,dim_copy,dim_count2[z],max_count,max_count2,sigma,sigma2,weights);
								}
							}
						}
					});
				}
			}
			finally {
				exec.shutdown();
			}
			while(!exec.isTerminated()){
				// wait
			}
			double[] t = VectorFun.max(err);
			if (t[0]>0){
				swap(x,(int)t[1],z,orderset[z],neighborlist,dim_count[z],dim_count2[z],max_count,max_count2,dm_out,sigma,sigma2,weights);
				counter++;
			}
		}
		System.out.println(Integer.toString(counter)+" swaps of "+Integer.toString(n*nclust)+" were useful");
		// check if the ordering has changed in any of the coordinates
		boolean change = true;
		for (int i=0; i<nclust; i++){
			if (Arrays.equals(orderset[i],orderset_old[i])){
				change = false;
			}
			orderset_old[i] = orderset[i].clone();
		}
		return change;
	}
	
	public static double swap_test(int x, int y, int z, final int[] orderset, final int[][][] neighborlist, int[][] dim_count, final int[][] dim_count2, final double[][] max_count, final double[][] max_count2, int sigma, double sigma2, final double[] weights){
		ArrayList<Integer> points = new ArrayList<Integer>();
		int ox = orderset[x];
		int oy = orderset[y];
		int d = dim_count.length;
		int n = orderset.length;
		for (int j=0; j<d; j++){
			for (int k=0; k<sigma; k++){
				// subtraction before swapping
				int a = x-k-1;
				if (a>=0&&a!=y){
					// subtract the current swap point from the neighborhood of all its neighbors
					// if it was contained in this dimensions neighborhood
					// we also need to make check how the neighborhood of the center point changes
					int oa = orderset[a];
					if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
						dim_count[j][oa]--;
						points.add(j*n+oa);
						dim_count[j][ox]--;
						points.add(j*n+ox);
					}
					if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
						dim_count[j][oa]++;
						points.add(j*n+oa);
						dim_count[j][oy]++;
						points.add(j*n+oy);
					}
				}
				a = x+k+1;
				if (a<n&&a!=y){
					int oa = orderset[a];
					if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
						dim_count[j][oa]--;
						points.add(j*n+orderset[a]);
						dim_count[j][ox]--;
						points.add(j*n+ox);
					}
					if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
						dim_count[j][oa]++;
						points.add(j*n+oa);
						dim_count[j][oy]++;
						points.add(j*n+oy);
					}
				}
				a = y-k-1;
				if (a>=0&&a!=x){
					int oa = orderset[a];
					if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
						dim_count[j][oa]--;
						points.add(j*n+oa);
						dim_count[j][oy]--;
						points.add(j*n+oy);
					}
					if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
						dim_count[j][oa]++;
						points.add(j*n+oa);
						dim_count[j][ox]++;
						points.add(j*n+ox);
					}
				}
				a = y+k+1;
				if (a<n&&a!=x){
					int oa = orderset[a];
					if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
						dim_count[j][oa]--;
						points.add(j*n+oa);
						dim_count[j][oy]--;
						points.add(j*n+oy);
					}
					if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
						dim_count[j][oa]++;
						points.add(j*n+oa);
						dim_count[j][ox]++;
						points.add(j*n+ox);
					}
				}
			}
		}
		int[] points2 = new int[points.size()];
		for (int j=0; j<points.size();j++){
			points2[j] = points.get(j);
		}
		Arrays.sort(points2);
		int count = 1;
		for (int j=1; j<points2.length; j++){
			if(points2[j]>points2[j-1]){
				count++;
			}
		}
		int[] p = new int[count];
		count = 0;
		if (points.size()>0){
			p[count] = points2[0];
			for (int j=1; j<points2.length; j++){
				if(points2[j]>points2[j-1]){
					count++;
					p[count] = points2[j];
				}
			}
		}
		// go through all these points and check if the maximum value changes
		double perr = 0;
		for (int j=0; j<p.length; j++){
			int loc = p[j]%n;
			int dim = p[j]/n;
			double a = Math.pow(2*sigma-dim_count[dim][loc],sigma2);
			double b = Math.pow(2*sigma-dim_count2[dim][loc],sigma2);
			// if we have increased the count value for the current point
			if (a<max_count[dim][loc]){
				perr-=(a-max_count[dim][loc])*weights[dim];
			}
			// otherwise check if the point had the highest value
			else if (b==max_count[dim][loc]){
				// will the second largest value take over?
				if (max_count2[dim][loc]<=a){
					perr-=(max_count2[dim][loc]-max_count[dim][loc])*weights[dim];
				}
				else {
					perr-=(a-max_count[dim][loc])*weights[dim];
				}
			}
		}
		for (int j=0; j<p.length; j++){
			int loc = p[j]%n;
			int dim = p[j]/n;
			dim_count[dim][loc] = dim_count2[dim][loc];
		}
		return perr;
	}
	
	public static void swap(int x, int y, int z, int[] orderset, final int[][][] neighborlist, int[][] dim_count, int[][] dim_count2, double[][] max_count, double[][] max_count2, ArrayList<ArrayList<TreeSet<SortPair>>> dm_out, int sigma, double sigma2, final double[] weights){
		ArrayList<Integer> points = new ArrayList<Integer>();
		int ox = orderset[x];
		int oy = orderset[y];
		int d = dim_count.length;
		int n = orderset.length;
		for (int j=0; j<d; j++){
			for (int k=0; k<sigma; k++){
				// subtraction before swapping
				int a = x-k-1;
				if (a>=0&&a!=y){
					// subtract the current swap point from the neighborhood of all its neighbors
					// if it was contained in this dimensions neighborhood
					// we also need to make check how the neighborhood of the center point changes
					int oa = orderset[a];
					if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
						dim_count[j][oa]--;
						points.add(j*n+oa);
						dim_count[j][ox]--;
						points.add(j*n+ox);
					}
					if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
						dim_count[j][oa]++;
						points.add(j*n+oa);
						dim_count[j][oy]++;
						points.add(j*n+oy);
					}
				}
				a = x+k+1;
				if (a<n&&a!=y){
					int oa = orderset[a];
					if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
						dim_count[j][oa]--;
						points.add(j*n+orderset[a]);
						dim_count[j][ox]--;
						points.add(j*n+ox);
					}
					if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
						dim_count[j][oa]++;
						points.add(j*n+oa);
						dim_count[j][oy]++;
						points.add(j*n+oy);
					}
				}
				a = y-k-1;
				if (a>=0&&a!=x){
					int oa = orderset[a];
					if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
						dim_count[j][oa]--;
						points.add(j*n+oa);
						dim_count[j][oy]--;
						points.add(j*n+oy);
					}
					if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
						dim_count[j][oa]++;
						points.add(j*n+oa);
						dim_count[j][ox]++;
						points.add(j*n+ox);
					}
				}
				a = y+k+1;
				if (a<n&&a!=x){
					int oa = orderset[a];
					if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
						dim_count[j][oa]--;
						points.add(j*n+oa);
						dim_count[j][oy]--;
						points.add(j*n+oy);
					}
					if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
						dim_count[j][oa]++;
						points.add(j*n+oa);
						dim_count[j][ox]++;
						points.add(j*n+ox);
					}
				}
			}
		}
		int[] points2 = new int[points.size()];
		for (int j=0; j<points.size();j++){
			points2[j] = points.get(j);
		}
		Arrays.sort(points2);
		int count = 1;
		for (int j=1; j<points2.length; j++){
			if(points2[j]>points2[j-1]){
				count++;
			}
		}
		int[] p = new int[count];
		count = 0;
		if (points.size()>0){
			p[count] = points2[0];
			for (int j=1; j<points2.length; j++){
				if(points2[j]>points2[j-1]){
					count++;
					p[count] = points2[j];
				}
			}
		}
		for (int j=0; j<p.length; j++){
			int loc = p[j]%n;
			int dim = p[j]/n;
			// update the max values and such
			SortPair sp = new SortPair(dim_count2[dim][loc],z);
			dm_out.get(dim).get(loc).remove(sp);
			dm_out.get(dim).get(loc).add(new SortPair(dim_count[dim][loc],z));
			sp = dm_out.get(dim).get(loc).last();
			max_count[dim][loc] = Math.pow(2*sigma-sp.getValue(),sigma2);
			max_count2[dim][loc] = Math.pow(2*sigma-dm_out.get(dim).get(loc).lower(sp).getValue(),sigma2);
			// copy over the old dim count values
			dim_count2[dim][loc] = dim_count[dim][loc];
		}
		// change the ordering
		orderset[x] = oy;
		orderset[y] = ox;
	}
	
	public static void neighbors(int[][][] neighborlist, final int[][] ir, final double[][] trainingset, final boolean[] categorical, int sigma){
		int d = neighborlist.length;
		int n = neighborlist[0].length;
		for (int j=0; j<d; j++){
			// for ordered dimensions
			if (!categorical[j]){
				for (int k=0; k<n; k++){
					for (int l=0; l<sigma; l++){
						if(k-l-1>=0){
							neighborlist[j][ir[j][k]][2*l] = ir[j][k-l-1];
						}
						else {
							neighborlist[j][ir[j][k]][2*l] = -1;
						}
						if(k+l+1<n){
							neighborlist[j][ir[j][k]][2*l+1] = ir[j][k+l+1];
						}
						else {
							neighborlist[j][ir[j][k]][2*l+1] = -1;
						}
					}
					Arrays.sort(neighborlist[j][ir[j][k]]);
				}
			}
			// otherwise referencing gets convoluted
			else {
				// get the number of labels
				double[] val = new double[n];
				for (int k=0; k<n; k++){
					val[k] = trainingset[k][j];
				}
				val = VectorConv.int2double(SortingFunctions.uniqueLabels(val));
				int l = (int)VectorFun.max(val)[0]+1;
				int[] counter = new int[l];
				int[] loc = new int[l];
				for (int k=0; k<n; k++){
					if (counter[(int)val[k]]==0){
						loc[(int)val[k]] = k;
					}
					counter[(int)val[k]]++;
				}
				for (int k=0; k<l; k++){
					neighborlist[j][loc[k]] = new int[counter[k]];
				}
				counter = new int[l];
				for (int k=0; k<n; k++){
					// good old cyclic referencing...
					// we're letting the array reference other array entries
					if (counter[(int)val[k]]>0){
						neighborlist[j][k] = neighborlist[j][loc[(int)val[k]]];
					}
					// it should matter if we inset in [j][k] or in [j][loc[k]]
					neighborlist[j][k][counter[(int)val[k]]] = k;
					counter[(int)val[k]]++;
				}
				for (int k=0; k<l; k++){
					Arrays.sort(neighborlist[j][loc[k]]);
				}
			}
		}
	}
		
	public static void counter(final int[][][] neighborlist, int[][][] dim_count, int[][][] dim_count2, final int[][] orderset, ArrayList<ArrayList<TreeSet<SortPair>>> dm_out, int sigma){
		int d = neighborlist.length;
		int n = neighborlist[0].length;
		int nclust = orderset.length;
		// count the overlapping neighbors for all points and dimensions and clusters
		for (int j=0; j<d; j++){
			for (int k=0; k<n; k++){
				for (int l=0; l<nclust; l++){
					// check if the neighbor based on the order set contains the same neighbor points
					for (int m=0; m<sigma; m++){
						// che
						if(k-m-1>=0){
							if (Arrays.binarySearch(neighborlist[j][orderset[l][k]],orderset[l][k-m-1])>=0){
								dim_count[l][j][orderset[l][k]]++;
							}
						}
						if(k+m+1<n){
							if (Arrays.binarySearch(neighborlist[j][orderset[l][k]],orderset[l][k+m+1])>=0){
								dim_count[l][j][orderset[l][k]]++;
							}
						}
					}
				}
			}
		}
		for (int i=0; i<nclust; i++){
			for (int j=0; j<d; j++){
				dim_count2[i][j] = dim_count[i][j].clone();
			}
		}
		for (int j=0; j<d; j++){
			dm_out.add(new ArrayList<TreeSet<SortPair>>());
			for (int k=0; k<n; k++){
				dm_out.get(j).add(new TreeSet<SortPair>());
				for (int l=0; l<nclust; l++){
					dm_out.get(j).get(k).add(new SortPair(dim_count[l][j][k],l));
				}
			}
		}
	}
}
