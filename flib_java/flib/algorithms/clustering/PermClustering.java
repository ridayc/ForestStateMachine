package flib.algorithms.clustering;

import java.util.TreeSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
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
	
	public static double ordering(final double[][] trainingset, int[][] orderset, final boolean[] categorical, final double[] weights, final double[] sigma, double scrdim, int maxit, double comp, double err){
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
		// counter for points that share a neighborhood
		int[][][] dim_count = new int[nclust][d][n];
		int[][][] dim_count2 = new int[nclust][d][n];
		int[][] upper = new int[nclust][n];
		int[][] lower = new int[nclust][n];
		// neighbors for each point for each dimension
		double[][] neighborlist = new double[d][n];
		ArrayList<ArrayList<TreeSet<SortPair>>> dm_out = new ArrayList<ArrayList<TreeSet<SortPair>>>();
		// we use another sigma to get the neighbors...
		neighbors(upper,lower,neighborlist,ir,trainingset,orderset,categorical);
		counter(categorical,neighborlist,dim_count,dim_count2,orderset,dm_out,sigma);
		double temp = err;
		// now comes the main algorithm with swapping fun!
		for (int i=0; i<1; i++){
			//int[] temp = Shuffle.randPerm(nclust);
			/*
			int[] z = new int[scrdim];
			for (int j=0; j<scrdim; j++){
				z[j] = temp[j];
			}
			if (i>0){
				scrambleDim(categorical,neighborlist,upper,lower,dim_count,dim_count2,orderset,dm_out,sigma,z);
			}
			*/
			double gap = 0;
			for (int k=0; k<d; k++){
				double t = 0;
				for (int l=0; l<n; l++){
					t+=Math.pow(2*sigma[0]-dm_out.get(k).get(l).last().getValue(),sigma[2]);
				}
				gap+=t*weights[k];
			}
			gap = Math.pow(gap/(VectorFun.sum(weights)*n),1/sigma[2]);
			System.out.println("Initial Gap Sum: "+Double.toString(gap));
			for (int j=0; j<maxit; j++){
				if(!(update(weights,categorical,orderset,neighborlist,upper,lower,dim_count,dim_count2,dm_out,sigma,temp)>comp)){
					gap = 0;
					for (int k=0; k<d; k++){
						double t = 0;
						for (int l=0; l<n; l++){
							t+=Math.pow(2*sigma[0]-dm_out.get(k).get(l).last().getValue(),sigma[2]);
						}
						gap+=t*weights[k];
					}
					gap = Math.pow(gap/(VectorFun.sum(weights)*n),1/sigma[2]);
					System.out.println("Iteration: "+Integer.toString(j));
					System.out.println("Gap Sum: "+Double.toString(gap));
					System.out.println("Finished the permutation clustering after "+Integer.toString(j)+" iterations");
					break;
				}
				gap = 0;
				for (int k=0; k<d; k++){
					double t = 0;
					for (int l=0; l<n; l++){
						t+=Math.pow(2*sigma[0]-dm_out.get(k).get(l).last().getValue(),sigma[2]);
					}
					gap+=t*weights[k];
				}
				gap = Math.pow(gap/(VectorFun.sum(weights)*n),1/sigma[2]);
				System.out.println("Iteration: "+Integer.toString(j));
				System.out.println("Gap Sum: "+Double.toString(gap));
				temp/=scrdim;
			}
		}
		return temp;
	}
	
	private static int update(final double[] weights, final boolean[] categorical, int[][] orderset, final double[][] neighborlist, int[][] upper, int[][] lower, int[][][] dim_count, int[][][] dim_count2,ArrayList<ArrayList<TreeSet<SortPair>>> dm_out, final double[] sigma, double err){
		int n = orderset[0].length;
		int nclust = orderset.length;
		int d = neighborlist.length;
		double[][] max_count = new double[d][n];
		double[][] max_count2 = new double[d][n];
		double gap = 0;
		for (int i=0; i<d; i++){
			for (int j=0; j<n; j++){
				SortPair sp = dm_out.get(i).get(j).last();
				max_count[i][j] = Math.pow(2*sigma[0]-sp.getValue(),sigma[2]);
				gap+=max_count[i][j]*weights[i];
				if(nclust>1){
					max_count2[i][j] = Math.pow(2*sigma[0]-dm_out.get(i).get(j).lower(sp).getValue(),sigma[2]);
				}
			}
		}
		int counter = 0;
		final int n2 = n*n;
		// hopefully we can shuffle... the list is quite large...
		int[] rp = Shuffle.randPerm(2*n2*nclust);
		Random rng = new Random();
		for (int i=0; i<rp.length; i++){
			boolean u = false;
			if (rp[i]%2==1){
				u = true;
			}
			rp[i] = rp[i]/2;
			final int x = rp[i]%n;
			final int y = (rp[i]/n)%n;
			final int z = rp[i]/n2;
			if (x!=y&&((!u&(x!=lower[z][y]))||(u&(x!=upper[z][y])))){
			//if (x!=y&&((!u&(x!=lower[z][y]))||(u&(x!=upper[z][y])))&&((upper[z][x]==-1||upper[z][y]==-1)&&(lower[z][x]==-1||lower[z][y]==-1))&&x!=lower[z][y]&&x!=upper[z][y]&&y!=lower[z][x]&&y!=upper[z][x]){
			//if (x!=y&&((!u&(x!=lower[z][y]))||(u&(x!=upper[z][y])))&&((lower[z][y]==-1||upper[z][y]==-1))&&((lower[z][x]==-1||upper[z][x]==-1))&&y!=lower[z][x]&&y!=upper[z][x]){
				// to remember points that were changed
				ArrayList<Integer> points = new ArrayList<Integer>();
				int lx = lower[z][x];
				int ux = upper[z][x];
				// all losses
				// losses of neighborhood x
				for (int j=0; j<sigma[0]; j++){
					for (int k=0; k<d; k++){
						double a = neighborlist[k][x];
						if (!categorical[k]){
							if (lx>-1){
								if (Math.abs(a-neighborlist[k][lx])<=sigma[1]){
									dim_count[z][k][x]--;
									points.add(k*n+x);
									dim_count[z][k][lx]--;
									points.add(k*n+lx);
								}
							}
							if (ux>-1){
								if (Math.abs(a-neighborlist[k][ux])<=sigma[1]){
									dim_count[z][k][x]--;
									points.add(k*n+x);
									dim_count[z][k][ux]--;
									points.add(k*n+ux);
								}
							}
						}
						else {
							if (lx>-1){
								if (a==neighborlist[k][lx]){
									dim_count[z][k][x]--;
									points.add(k*n+x);
									dim_count[z][k][lx]--;
									points.add(k*n+lx);
								}
							}
							if (ux>-1){
								if (a==neighborlist[k][ux]){
									dim_count[z][k][x]--;
									points.add(k*n+x);
									dim_count[z][k][ux]--;
									points.add(k*n+ux);
								}
							}
						}
					}
					// shift through neighbors by 1
					if (lx>-1){
						lx = lower[z][lx];
					}
					if (ux>-1){
						ux = upper[z][ux];
					}
				}
				// losses of neighborhood y
				// preparation of variables
				int ly = y;
				int uy = y;
				int c = 0;
				if (u){
					for (int j=0; j<sigma[0]; j++){
						if (upper[z][uy]>-1){
							uy = upper[z][uy];
						}
						else {
							c = (int)sigma[0]-j;
							j = (int)sigma[0];
						}
					}
					for (int j=0; j<sigma[0]-1; j++){
						if (lower[z][ly]>-1){
							ly = lower[z][ly];
							if (j>=c){
								uy = lower[z][uy];
							}
						}
						else {
							j = (int)sigma[0]-1;
						}
					}
					if (c==sigma[0]){
						uy = -1;
					}
				}
				else {
					for (int j=0; j<sigma[0]-1; j++){
						if (upper[z][uy]>-1){
							uy = upper[z][uy];
						}
						else {
							c = (int)sigma[0]-1-j;
							j = (int)sigma[0]-1;
						}
					}
					if (lower[z][ly]>-1){
						ly = lower[z][ly];
						for (int j=0; j<sigma[0]-1; j++){
							if (lower[z][ly]>-1){
								ly = lower[z][ly];
								if (j>=c){
									uy = lower[z][uy];
								}
							}
							else {
								j = (int)sigma[0]-1;
							}
						}
					}
					else {
						uy = -1;
					}
				}
				while(ly!=y){
					//System.out.println("i: "+Integer.toString(i));
					//System.out.println("ly: "+Integer.toString(ly));
					if (uy>-1&&ly!=x&&uy!=x){
						for (int k=0; k<d; k++){
							if (!categorical[k]){
								if (Math.abs(neighborlist[k][ly]-neighborlist[k][uy])<=sigma[1]){
									dim_count[z][k][ly]--;
									points.add(k*n+ly);
									dim_count[z][k][uy]--;
									points.add(k*n+uy);
								}
							}
							else {
								if (neighborlist[k][ly]==neighborlist[k][uy]){
									dim_count[z][k][ly]--;
									points.add(k*n+ly);
									dim_count[z][k][uy]--;
									points.add(k*n+uy);
								}
							}
						}
					}
					ly = upper[z][ly];
					if (uy>-1){
						uy = upper[z][uy];
					}
				}
				if (u){
					if (uy>-1&&ly!=x&&uy!=x){
						for (int k=0; k<d; k++){
							if (!categorical[k]){
								if (Math.abs(neighborlist[k][ly]-neighborlist[k][uy])<=sigma[1]){
									dim_count[z][k][ly]--;
									points.add(k*n+ly);
									dim_count[z][k][uy]--;
									points.add(k*n+uy);
								}
							}
							else {
								if (neighborlist[k][ly]==neighborlist[k][uy]){
									dim_count[z][k][ly]--;
									points.add(k*n+ly);
									dim_count[z][k][uy]--;
									points.add(k*n+uy);
								}
							}
						}
					}
				}
				// store the current neighborhoods
				int[] old = new int[6];
				old[0] = x;
				old[1] = lower[z][x];
				old[2] = upper[z][x];
				old[3] = y;
				old[4] = lower[z][y];
				old[5] = upper[z][y];
				// swap x neighborhood
				if (old[1]>-1){
					upper[z][old[1]] = old[2];
				}
				if (old[2]>-1){
					lower[z][old[2]] = old[1];
				}
				// insert x
				if (u){
					upper[z][y] = x;
					lower[z][x] = y;
					upper[z][x] = old[5];
					if (old[5]>-1){
						lower[z][old[5]] = x;
					}
				}
				else {
					lower[z][y] = x;
					upper[z][x] = y;
					lower[z][x] = old[4];
					if (old[4]>-1){
						upper[z][old[4]] = x;
					}
				}
				lx = lower[z][x];
				ux = upper[z][x];
				// gains due to new neighbors x
				for (int j=0; j<sigma[0]; j++){
					for (int k=0; k<d; k++){
						double a = neighborlist[k][x];
						if (!categorical[k]){
							if (lx>-1){
								if (Math.abs(a-neighborlist[k][lx])<=sigma[1]){
									dim_count[z][k][x]++;
									points.add(k*n+x);
									dim_count[z][k][lx]++;
									points.add(k*n+lx);
								}
							}
							if (ux>-1){
								if (Math.abs(a-neighborlist[k][ux])<=sigma[1]){
									dim_count[z][k][x]++;
									points.add(k*n+x);
									dim_count[z][k][ux]++;
									points.add(k*n+ux);
								}
							}
						}
						else {
							if (lx>-1){
								if (a==neighborlist[k][lx]){
									dim_count[z][k][x]++;
									points.add(k*n+x);
									dim_count[z][k][lx]++;
									points.add(k*n+lx);
								}
							}
							if (ux>-1){
								if (a==neighborlist[k][ux]){
									dim_count[z][k][x]++;
									points.add(k*n+x);
									dim_count[z][k][ux]++;
									points.add(k*n+ux);
								}
							}
						}
					}
					// shift through neighbors by 1
					if (lx>-1){
						lx = lower[z][lx];
					}
					if (ux>-1){
						ux = upper[z][ux];
					}
				}
				// gains in previous neighborhood of x
				lx = old[1];
				ux = old[2];
				c = 0;
				if (ux>-1&&lx>-1){
					for (int j=0; j<sigma[0]-1; j++){
						if (upper[z][ux]>-1){
							ux = upper[z][ux];
						}
						else {
							c = (int)sigma[0]-1-j;
							j = (int)sigma[0]-1;
						}
					}
					for (int j=0; j<sigma[0]-1; j++){
						if (lower[z][lx]>-1){
							lx = lower[z][lx];
							if (j>=c){
								ux = lower[z][ux];
							}
						}
						else {
							j = (int)sigma[0]-1;
						}
					}
					while(lx!=old[2]){
						if (ux>-1&&lx!=x&&ux!=x){
							for (int k=0; k<d; k++){
								if (!categorical[k]){
									if (Math.abs(neighborlist[k][lx]-neighborlist[k][ux])<=sigma[1]){
										dim_count[z][k][lx]++;
										points.add(k*n+lx);
										dim_count[z][k][ux]++;
										points.add(k*n+ux);
									}
								}
								else {
									if (neighborlist[k][lx]==neighborlist[k][ux]){
										dim_count[z][k][lx]++;
										points.add(k*n+lx);
										dim_count[z][k][ux]++;
										points.add(k*n+ux);
									}
								}
							}
						}
						lx = upper[z][lx];
						if (ux>-1){
							ux = upper[z][ux];
						}
					}
				}				
				int count = 0;
				for (int j=0; j<points.size();j++){
					int loc = points.get(j)%n;
					int dim = points.get(j)/n;
					if (dim_count[z][dim][loc]!=dim_count2[z][dim][loc]){
						count++;
					}
				}
				int[] points2 = new int[count];
				count = 0;
				for (int j=0; j<points.size();j++){
					int loc = points.get(j)%n;
					int dim = points.get(j)/n;
					if (dim_count[z][dim][loc]!=dim_count2[z][dim][loc]){
						points2[count] = points.get(j);
						count++;
					}
				}
				Arrays.sort(points2);
				if (points2.length>0){
					count = 1;
				}
				for (int j=1; j<points2.length; j++){
					if(points2[j]>points2[j-1]){
						count++;
					}
				}
				int[] p = new int[count];
				count = 0;
				if (p.length>0){
					p[count] = points2[0];
					for (int j=1; j<points2.length; j++){
						if(points2[j]>points2[j-1]){
							count++;
							p[count] = points2[j];
						}
					}
				}
				// go through all these points and check if the maximum value changes
				double perr_p = 0;
				double perr_n = 0;
				for (int j=0; j<p.length; j++){
					int loc = p[j]%n;
					int dim = p[j]/n;
					double a = Math.pow(2*sigma[0]-dim_count[z][dim][loc],sigma[2]);
					double b = Math.pow(2*sigma[0]-dim_count2[z][dim][loc],sigma[2]);
					// if we have increased the count value for the current point
					if (a<max_count[dim][loc]){
						perr_p-=(a-max_count[dim][loc])*weights[dim];
					}
					// otherwise check if the point had the highest value
					else if (b==max_count[dim][loc]){
						// will the second largest value take over?
						if (max_count2[dim][loc]<=a&&nclust>1){
							perr_n-=(max_count2[dim][loc]-max_count[dim][loc])*weights[dim];
						}
						else {
							perr_n-=(a-max_count[dim][loc])*weights[dim];
						}
					}
				}
				// check if we have improved
				//if (perr>0||0.5*Math.exp(perr/sigma)>rng.nextDouble()){
				double r = 1;
				if (perr_n<0){
					r = -perr_p/perr_n;
				}
				if ((perr_p+perr_n>=0)||Math.exp((perr_p+perr_n)/err)>rng.nextDouble()){
				//if ((perr_p+perr_n>0)||(Math.exp(-1./(err*r*perr_p))>rng.nextDouble())){
					// adjust all points
					for (int j=0; j<p.length; j++){
						int loc = p[j]%n;
						int dim = p[j]/n;
						// update the max values and such
						SortPair sp = new SortPair(dim_count2[z][dim][loc],z);
						dm_out.get(dim).get(loc).remove(sp);
						dm_out.get(dim).get(loc).add(new SortPair(dim_count[z][dim][loc],z));
						sp = dm_out.get(dim).get(loc).last();
						max_count[dim][loc] = Math.pow(2*sigma[0]-sp.getValue(),sigma[2]);
						if (nclust>1){
							max_count2[dim][loc] = Math.pow(2*sigma[0]-dm_out.get(dim).get(loc).lower(sp).getValue(),sigma[2]);
						}
						// copy over the old dim count values
						dim_count2[z][dim][loc] = dim_count[z][dim][loc];
					}
					// change the ordering
					counter++;
					gap-=perr_p+perr_n;
				}
				// otherwise we have some resetting to do
				else {
					for (int j=0; j<p.length; j++){
						int loc = p[j]%n;
						int dim = p[j]/n;
						dim_count[z][dim][loc] = dim_count2[z][dim][loc];
					}
					// swap x neighborhood
					lower[z][x] = old[1];
					upper[z][x] = old[2];
					if (old[1]>-1){
						upper[z][old[1]] = x;
					}
					if (old[2]>-1){
						lower[z][old[2]] = x;
					}
					// swap y neighborhood
					if (u){
						upper[z][y] = old[5];
						if (old[5]>-1){
							lower[z][old[5]] = y;
						}
					}
					else {
						lower[z][y] = old[4];
						if (old[4]>-1){
							upper[z][old[4]] = y;
						}
					}
				}
			}
		}
		// reorder all values
		for (int i=0; i<nclust; i++){
			int a = 0;
			for (int j=0; j<n; j++){
				if (lower[i][j]==-1){
					a = j;
					break;
				}
			}
			for (int j=0; j<n; j++){
				orderset[i][j] = a;
				a = upper[i][a];
			}
		}
		System.out.println(Integer.toString(counter)+" swaps of "+Integer.toString(2*n2*nclust)+" were useful");
		for (int i=0; i<nclust; i++){
			//System.out.println("lower order "+Arrays.toString(lower[i]));
			//System.out.println("upper order "+Arrays.toString(upper[i]));
		}
		// check if the ordering has changed in any of the coordinates
		return counter;
	}
	
	public static void neighbors(int[][] upper, int[][] lower, double[][] neighborlist, final int[][] ir, final double[][] trainingset, final int[][] orderset, final boolean[] categorical){
		int d = neighborlist.length;
		int n = neighborlist[0].length;
		int nclust = orderset.length;
		for (int j=0; j<d; j++){
			// for ordered dimensions
			if (!categorical[j]){
				for (int k=0; k<n; k++){
					neighborlist[j][ir[j][k]] = k;
				}
			}
			// otherwise referencing gets convoluted... not anymore
			else {
				for (int k=0; k<n; k++){
					neighborlist[j][k] = trainingset[k][j];
				}
			}
		}
		for (int j=0; j<nclust; j++){
			for (int k=1; k<n-1; k++){
				upper[j][orderset[j][k]] = orderset[j][k+1];
				lower[j][orderset[j][k]] = orderset[j][k-1];
			}
			upper[j][orderset[j][0]] = orderset[j][1];
			lower[j][orderset[j][0]] = -1;
			upper[j][orderset[j][n-1]] = -1;
			lower[j][orderset[j][n-1]] = orderset[j][n-2];
		}
	}
		
	public static void counter(final boolean[] categorical, final double[][] neighborlist, int[][][] dim_count, int[][][] dim_count2, final int[][] orderset, ArrayList<ArrayList<TreeSet<SortPair>>> dm_out, final double[] sigma){
		int d = neighborlist.length;
		int n = neighborlist[0].length;
		int nclust = orderset.length;
		// count the overlapping neighbors for all points and dimensions and clusters
		for (int j=0; j<d; j++){
			for (int k=0; k<n; k++){
				for (int l=0; l<nclust; l++){
					// check if the neighbor based on the order set contains the same neighbor points
					if (!categorical[j]){
						for (int m=0; m<sigma[0]; m++){
							// che
							if(k-m-1>=0){
								if (Math.abs(neighborlist[j][orderset[l][k]]-neighborlist[j][orderset[l][k-m-1]])<=sigma[1]){
									dim_count[l][j][orderset[l][k]]++;
								}
							}
							if(k+m+1<n){
								if (Math.abs(neighborlist[j][orderset[l][k]]-neighborlist[j][orderset[l][k+m+1]])<=sigma[1]){
									dim_count[l][j][orderset[l][k]]++;
								}
							}
						}
					}
					else {
						for (int m=0; m<sigma[0]; m++){
							// che
							if(k-m-1>=0){
								if (neighborlist[j][orderset[l][k]]==neighborlist[j][orderset[l][k-m-1]]){
									dim_count[l][j][orderset[l][k]]++;
								}
							}
							if(k+m+1<n){
								if (neighborlist[j][orderset[l][k]]==neighborlist[j][orderset[l][k+m+1]]){
									dim_count[l][j][orderset[l][k]]++;
								}
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
	
	public static void scrambleDim(final boolean[] categorical, final double[][] neighborlist, int[][] upper, int[][] lower, int[][][] dim_count, int[][][] dim_count2, int[][] orderset, ArrayList<ArrayList<TreeSet<SortPair>>> dm_out, final double[] sigma, int[] z){
		// variables
		int d = neighborlist.length;
		int n = neighborlist[0].length;
		int nclust = orderset.length;
		// remove the the neighbor counts for this dimension from the treeset
		// reset the dim_count value
		for (int i=0; i<d; i++){
			for (int j=0; j<n; j++){
				for (int k=0; k<z.length; k++){
					dm_out.get(i).get(j).remove(new SortPair(dim_count[z[k]][i][j],z[k]));
					dim_count[z[k]][i][j] = 0;
				}
			}
		}
		// create a new ordering for the current dimension
		for (int k=0; k<z.length; k++){
			orderset[z[k]] = Shuffle.randPerm(n);
		}
		// get the new neighbor points
		for (int k=0; k<z.length; k++){
			for (int i=1; i<n-1; i++){
				upper[z[k]][orderset[z[k]][i]] = orderset[z[k]][i+1];
				lower[z[k]][orderset[z[k]][i]] = orderset[z[k]][i-1];
			}
			upper[z[k]][orderset[z[k]][0]] = orderset[z[k]][1];
			lower[z[k]][orderset[z[k]][0]] = -1;
			upper[z[k]][orderset[z[k]][n-1]] = -1;
			lower[z[k]][orderset[z[k]][n-1]] = orderset[z[k]][n-2];
		}
		// count the number of neighbors according to the original dimensions
		for (int j=0; j<d; j++){
			for (int k=0; k<n; k++){
				// check if the neighbor based on the order set contains the same neighbor points
				if (!categorical[j]){
					for (int l=0; l<z.length; l++){
						for (int m=0; m<sigma[0]; m++){
							// che
							if(k-m-1>=0){
								if (Math.abs(neighborlist[j][orderset[z[l]][k]]-neighborlist[j][orderset[z[l]][k-m-1]])<=sigma[1]){
									dim_count[z[l]][j][orderset[z[l]][k]]++;
								}
							}
							if(k+m+1<n){
								if (Math.abs(neighborlist[j][orderset[z[l]][k]]-neighborlist[j][orderset[z[l]][k+m+1]])<=sigma[1]){
									dim_count[z[l]][j][orderset[z[l]][k]]++;
								}
							}
						}
					}
				}
				else {
					for (int l=0; l<z.length; l++){
						for (int m=0; m<sigma[0]; m++){
							// che
							if(k-m-1>=0){
								if (neighborlist[j][orderset[z[l]][k]]==neighborlist[j][orderset[z[l]][k-m-1]]){
									dim_count[z[l]][j][orderset[z[l]][k]]++;
								}
							}
							if(k+m+1<n){
								if (neighborlist[j][orderset[z[l]][k]]==neighborlist[j][orderset[z[l]][k+m+1]]){
									dim_count[z[l]][j][orderset[z[l]][k]]++;
								}
							}
						}
					}
				}
			}
		}
		for (int j=0; j<d; j++){
			for (int k=0; k<z.length; k++){
				dim_count2[z[k]][j] = dim_count[z[k]][j].clone();
			}
		}
		for (int j=0; j<d; j++){
			for (int k=0; k<n; k++){
				for (int l=0; l<z.length; l++){
					dm_out.get(j).get(k).add(new SortPair(dim_count[z[l]][j][k],z[l]));
				}
			}
		}
	}
		
}
