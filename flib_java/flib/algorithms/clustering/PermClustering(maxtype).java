package flib.algorithms.clustering;

import java.util.TreeSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;
import flib.math.RankSort;
import flib.math.SortPair;
import flib.math.VectorFun;
import flib.math.random.Shuffle;

public class PermClustering {
	
	public static void ordering(final double[][] trainingset, int[][] orderset, final int[] sigma, final int[] sigma3, int maxit, double sigma2){
		int n = trainingset.length;
		int d = trainingset[0].length;
		int nclust = orderset.length;
		Random rng = new Random();
		// traingset ordering
		int[][] ir = new int[d][n];
		for (int i=0; i<d; i++){
			double[] val = new double[n];
			for (int j=0; j<n; j++){
				val[j] = trainingset[j][i];
			}
			ir[i] = (new RankSort(val)).getRank();
		}
		int[][] orderset_old = new int[nclust][n];
		for (int i=0; i<nclust; i++){
			//orderset[i] = Shuffle.randPerm(n);
			orderset_old[i] = orderset[i].clone();
		}
		for (int i=0; i<sigma.length; i++){
			// counter for points that share a neighborhood
			int[][][] dim_count = new int[nclust][d][n];
			int[][][] dim_count2 = new int[nclust][d][n];
			// neighbors for each point for each dimension
			int[][][] neighborlist = new int[d][n][2*sigma3[i]];
			ArrayList<ArrayList<TreeSet<SortPair>>> dm_out = new ArrayList<ArrayList<TreeSet<SortPair>>>();
			// we use another sigma to get the neighbors...
			neighbors(neighborlist,ir,sigma3[i]);
			counter(neighborlist,dim_count,dim_count2,orderset,dm_out,sigma[i]);
			// now comes the main algorithm with swapping fun!
			for (int j=0; j<maxit; j++){
				if(!update(orderset,orderset_old,neighborlist,dim_count,dim_count2,dm_out,sigma[i],sigma2)){
					double gap = 0;
					for (int k=0; k<d; k++){
						for (int l=0; l<n; l++){
							gap+=Math.pow(dm_out.get(k).get(l).last().getValue(),sigma2);
						}
					}
					System.out.println("Sigma: "+Double.toString(sigma[i])+", Iteration: "+Integer.toString(j));
					System.out.println("Gap Sum: "+Double.toString(gap));
					System.out.println("Finished the permutation clustering after "+Integer.toString(j)+" iterations");
					break;
				}
				double gap = 0;
				for (int k=0; k<d; k++){
					for (int l=0; l<n; l++){
						gap+=Math.pow(dm_out.get(k).get(l).last().getValue(),sigma2);
					}
				}
				System.out.println("Sigma: "+Double.toString(sigma[i])+", Iteration: "+Integer.toString(j));
				System.out.println("Gap Sum: "+Double.toString(gap));
			}
		}
	}
	
	private static boolean update(int[][] orderset, int[][] orderset_old, final int[][][] neighborlist, int[][][] dim_count, int[][][] dim_count2,ArrayList<ArrayList<TreeSet<SortPair>>> dm_out, double sigma, double sigma2){
		int n = orderset[0].length;
		int nclust = orderset.length;
		int d = neighborlist.length;
		double[][] max_count = new double[d][n];
		double[][] max_count2 = new double[d][n];
		for (int i=0; i<d; i++){
			for (int j=0; j<n; j++){
				SortPair sp = dm_out.get(i).get(j).last();
				max_count[i][j] = Math.pow(sp.getValue(),sigma2);
				max_count2[i][j] = Math.pow(dm_out.get(i).get(j).lower(sp).getValue(),sigma2);
			}
		}
		Random rng = new Random();
		final int n2 = n*n;
		// hopefully we can shuffle... the list is quite large...
		int[] rp = Shuffle.randPerm(n2*nclust);
		for (int i=0; i<n2*nclust; i++){
			final int x = rp[i]%n;
			final int y = (rp[i]/n)%n;
			final int z = rp[i]/n2;
			int ox = orderset[z][x];
			int oy = orderset[z][y];
			if (x!=y){
				// to remember points that were changed
				ArrayList<Integer> points = new ArrayList<Integer>();
				for (int j=0; j<d; j++){
					for (int k=0; k<sigma; k++){
						// subtraction before swapping
						int a = x-k-1;
						if (a>=0&&a!=y){
							// subtract the current swap point from the neighborhood of all its neighbors
							// if it was contained in this dimensions neighborhood
							// we also need to make check how the neighborhood of the center point changes
							int oa = orderset[z][a];
							if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
								dim_count[z][j][oa]--;
								points.add(j*n+oa);
								dim_count[z][j][ox]--;
								points.add(j*n+ox);
							}
							if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
								dim_count[z][j][oa]++;
								points.add(j*n+oa);
								dim_count[z][j][oy]++;
								points.add(j*n+oy);
							}
						}
						a = x+k+1;
						if (a<n&&a!=y){
							int oa = orderset[z][a];
							if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
								dim_count[z][j][oa]--;
								points.add(j*n+orderset[z][a]);
								dim_count[z][j][ox]--;
								points.add(j*n+ox);
							}
							if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
								dim_count[z][j][oa]++;
								points.add(j*n+oa);
								dim_count[z][j][oy]++;
								points.add(j*n+oy);
							}
						}
						a = y-k-1;
						if (a>=0&&a!=x){
							int oa = orderset[z][a];
							if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
								dim_count[z][j][oa]--;
								points.add(j*n+oa);
								dim_count[z][j][oy]--;
								points.add(j*n+oy);
							}
							if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
								dim_count[z][j][oa]++;
								points.add(j*n+oa);
								dim_count[z][j][ox]++;
								points.add(j*n+ox);
							}
						}
						a = y+k+1;
						if (a<n&&a!=x){
							int oa = orderset[z][a];
							if (Arrays.binarySearch(neighborlist[j][oa],oy)>=0){
								dim_count[z][j][oa]--;
								points.add(j*n+oa);
								dim_count[z][j][oy]--;
								points.add(j*n+oy);
							}
							if (Arrays.binarySearch(neighborlist[j][oa],ox)>=0){
								dim_count[z][j][oa]++;
								points.add(j*n+oa);
								dim_count[z][j][ox]++;
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
					double a = Math.pow(dim_count[z][dim][loc],sigma2);
					double b = Math.pow(dim_count2[z][dim][loc],sigma2);
					// if we have increased the count value for the current point
					if (a>max_count[dim][loc]){
						perr+=a-max_count[dim][loc];
					}
					// otherwise check if the point had the highest value
					else if (b==max_count[dim][loc]){
						// will the second largest value take over?
						if (max_count2[dim][loc]>=a){
							perr+=max_count2[dim][loc]-max_count[dim][loc];
						}
						else {
							perr+=a-max_count[dim][loc];
						}
					}
				}
				// check if we have improved
				//if (perr>0||0.5*Math.exp(perr/sigma)>rng.nextDouble()){
				if (perr>0){
					// adjust all points
					for (int j=0; j<p.length; j++){
						int loc = p[j]%n;
						int dim = p[j]/n;
						// update the max values and such
						SortPair sp = new SortPair(dim_count2[z][dim][loc],z);
						dm_out.get(dim).get(loc).remove(sp);
						dm_out.get(dim).get(loc).add(new SortPair(dim_count[z][dim][loc],z));
						sp = dm_out.get(dim).get(loc).last();
						max_count[dim][loc] = Math.pow(sp.getValue(),sigma2);
						max_count2[dim][loc] = Math.pow(dm_out.get(dim).get(loc).lower(sp).getValue(),sigma2);
						// copy over the old dim count values
						dim_count2[z][dim][loc] = dim_count[z][dim][loc];
					}
					// change the ordering
					orderset[z][x] = oy;
					orderset[z][y] = ox;
				}
				// otherwise we have a minimal amount of resetting to do
				else {
					for (int j=0; j<p.length; j++){
						int loc = p[j]%n;
						int dim = p[j]/n;
						dim_count[z][dim][loc] = dim_count2[z][dim][loc];
					}
				}
			}
		}
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
	
	public static void neighbors(int[][][] neighborlist, final int[][] ir, int sigma){
		int d = neighborlist.length;
		int n = neighborlist[0].length;
		for (int j=0; j<d; j++){
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
