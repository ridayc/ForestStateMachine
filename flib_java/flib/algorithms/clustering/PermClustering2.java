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
// distances given by random forest proximities
// we use a lot of dimension swapping of a set of permutation matrices to achieve local
// convergence
public class PermClustering2 {
	
	public static void ordering(final int[][] proximities, int[][] orderset, double[] sigma, int maxit, double comp){
		// proximities: the initial distances between all the points. We find the nearest neighbors for each point based on these
		// orderset: the initial rank ordering of the output dimensions for the compression algorithm
		// sigma: neighborhood size to be considered for performance
		// sigma2: power factor for the error type. A factor of two corresponds to standard squared error
		// maxit: maximal number of iterations if there's no convergence
		// number of points in the set
		int n = proximities.length;
		// number of compression dimensions
		int nclust = orderset.length;
		// nieghborhood distance power
		double s = 0;
		for (int i=0; i<(int)sigma[0]; i++){
			s+=Math.pow((int)sigma[0]-(double)i,sigma[4]);
		}
		// proximity neighbors
		int[][] ir = new int[n][n];
		for (int i=0; i<n; i++){
			double[] val = new double[n];
			for (int j=0; j<n; j++){
				val[j] = proximities[i][j];
			}
			ir[i] = (new RankSort(val)).getRank();
		}
		// counter for points that are in the neighborhood of each point
		double[] dim_count = new double[n];
		double[] dim_count2 = new double[n];
		// neighbor location for each point for each dimension
		int[][] neighborinlist = new int[n][(int)sigma[0]];
		// points which consider this point as a neighbor
		// the neighbor rank matrix is not symmetric
		int[][][] neighboroutlist = new int[n][][];
		// sigma[2]-norms for neighbordistances
		double[][] distlist = new double[n][(int)sigma[0]];
		double[][] distlist2 = new double[n][(int)sigma[0]];
		// check if points are within the neighborhood
		boolean[][] distcount = new boolean[n][(int)sigma[0]];
		boolean[][] distcount2 = new boolean[n][(int)sigma[0]];
		// calculate neighborhood distances
		neighbors(orderset,neighborinlist,neighboroutlist,distlist,distlist2,distcount,distcount2,dim_count,dim_count2,ir,sigma);
		double gap = 0;
		for (int k=0; k<n; k++){
			gap+=Math.pow(s-dim_count[k],sigma[3]);
		}
		gap = Math.pow(gap/n,1./sigma[3]);
		System.out.println("Initial Gap Sum: "+Double.toString(gap));
		// now comes the main algorithm with swapping fun!
		for (int j=0; j<maxit; j++){
			if(!(update(orderset,neighborinlist,neighboroutlist,distlist,distlist2,distcount,distcount2,dim_count,dim_count2,sigma)>comp)){
				gap = 0;
				for (int k=0; k<n; k++){
					gap+=Math.pow(s-dim_count[k],sigma[3]);
				}
				gap = Math.pow(gap/n,1./sigma[3]);
				System.out.println("Iteration: "+Integer.toString(j));
				System.out.println("Gap Sum: "+Double.toString(gap));
				System.out.println("Finished the permutation clustering after "+Integer.toString(j)+" iterations");
				break;
			}
			gap = 0;
			for (int k=0; k<n; k++){
				gap+=Math.pow(s-dim_count[k],sigma[3]);
			}
			gap = Math.pow(gap/n,1./sigma[3]);
			System.out.println("Iteration: "+Integer.toString(j));
			System.out.println("Gap Sum: "+Double.toString(gap));
		}
		// counter for points that are in the neighborhood of each point
		dim_count = new double[n];
		dim_count2 = new double[n];
		// neighbor location for each point for each dimension
		neighborinlist = new int[n][(int)sigma[0]];
		// points which consider this point as a neighbor
		// the neighbor rank matrix is not symmetric
		neighboroutlist = new int[n][][];
		// sigma[2]-norms for neighbordistances
		distlist = new double[n][(int)sigma[0]];
		distlist2 = new double[n][(int)sigma[0]];
		// check if points are within the neighborhood
		distcount = new boolean[n][(int)sigma[0]];
		distcount2 = new boolean[n][(int)sigma[0]];
		neighbors(orderset,neighborinlist,neighboroutlist,distlist,distlist2,distcount,distcount2,dim_count,dim_count2,ir,sigma);
		gap = 0;
		for (int k=0; k<n; k++){
			gap+=Math.pow(s-dim_count[k],sigma[3]);
		}
		gap = Math.pow(gap/n,1./sigma[3]);
		System.out.println("Final Gap Sum: "+Double.toString(gap));
	}
	
	private static int update(int[][] orderset, final int[][] neighborinlist, final int[][][] neighboroutlist, double[][] distlist, double[][] distlist2, boolean[][] distcount, boolean[][] distcount2, double[] dim_count, double[] dim_count2, final double[] sigma){
		int n = orderset[0].length;
		int nclust = orderset.length;
		int sg = (int)sigma[0];
		int count = 0;
		double count2 = 0;
		int counter = 0;
		final int n2 = n*n;
		// sum of neighborhood distance powers
		double s = 0;
		for (int i=0; i<sg; i++){
			s+=Math.pow(sg-(double)i,sigma[4]);
		}
		Random rng = new Random();
		// all possible point swappings for all dimensions
		// hopefully we can shuffle... the list is quite large...
		int[] rp = Shuffle.randPerm(n2*nclust);
		for (int i=0; i<n2*nclust; i++){
			final int x = rp[i]%n;
			final int y = (rp[i]/n)%n;
			final int z = rp[i]/n2;
			int ox = orderset[z][x];
			int oy = orderset[z][y];
			// no swapping of the same point
			if (x!=y){
				// points that might see a change
				ArrayList<Integer> points = new ArrayList<Integer>();
				// go through and adjust the neighbors of both points
				for (int j=0; j<sg; j++){
					int oa = orderset[z][neighborinlist[x][j]];
					int ob = orderset[z][neighborinlist[y][j]];
					points.add(x*sg+j);
					points.add(y*sg+j);
					// losses in the old neighborhood
					// additions in the new neighborhood
					// only if the points weren't neighbors
					if (oy!=oa){
						distlist[x][j]-=Math.pow(Math.abs(ox-oa),sigma[2]);
						distlist[x][j]+=Math.pow(Math.abs(oy-oa),sigma[2]);
					}
					if (ox!=ob){
						distlist[y][j]-=Math.pow(Math.abs(oy-ob),sigma[2]);
						distlist[y][j]+=Math.pow(Math.abs(ox-ob),sigma[2]);
					}
				}
				// neighbors from the other perspective
				for (int j=0; j<neighboroutlist[x].length; j++){
					int a = neighboroutlist[x][j][0];
					int b = neighboroutlist[x][j][1];
					points.add(a*sg+b);
					int oa = orderset[z][a];
					if (oy!=oa){
						distlist[a][b]-=Math.pow(Math.abs(ox-oa),sigma[2]);
						distlist[a][b]+=Math.pow(Math.abs(oy-oa),sigma[2]);
					}
				}
				for (int j=0; j<neighboroutlist[y].length; j++){
					int a = neighboroutlist[y][j][0];
					int b = neighboroutlist[y][j][1];
					points.add(a*sg+b);
					int oa = orderset[z][a];
					if (ox!=oa){
						distlist[a][b]-=Math.pow(Math.abs(oy-oa),sigma[2]);
						distlist[a][b]+=Math.pow(Math.abs(ox-oa),sigma[2]);
					}
				}
				// create a list of all visited points
				int[] points2 = new int[points.size()];
				count = 0;
				for (int j=0; j<points.size();j++){
					points2[j] = points.get(j);
				}
				// make the list unique
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
				count2 = 0;
				// check how the neighborhoods have changed
				int c = p[0]/sg;
				for (int j=0; j<p.length; j++){
					int a = p[j]/sg;
					int b = p[j]%sg;
					// adjust the neighborhoods
					// if the point is a neighbor now
					if (Math.pow(distlist[a][b]/nclust,1./sigma[2])<=sigma[1]){
						distcount[a][b] = true;
						if (distcount[a][b]!=distcount2[a][b]){
							dim_count[a]+=Math.pow(sg-(double)b,sigma[4]);
						}
					}
					else {
						distcount[a][b] = false;
						if (distcount[a][b]!=distcount2[a][b]){
							dim_count[a]-=Math.pow(sg-(double)b,sigma[4]);
						}
					}
					// whenever the main observed point changes
					if (a!=c){
						count2+=Math.pow(s-dim_count2[c],sigma[3])-Math.pow(s-dim_count[c],sigma[3]);
					}
					c = a;
				}
				count2+=Math.pow(s-dim_count2[c],sigma[3])-Math.pow(s-dim_count[c],sigma[3]);
				// if the swapping improved the neighborhood closeness
				if (count2>=0||rng.nextDouble()<sigma[6]){
					// successful swap
					counter++;
					// overwrite all old values
					for (int j=0; j<p.length; j++){
						int a = p[j]/sg;
						int b = p[j]%sg;
						distlist2[a][b] = distlist[a][b];
						distcount2[a][b] = distcount[a][b];
						dim_count2[a] = dim_count[a];
					}
					// swap the points
					orderset[z][x] = oy;
					orderset[z][y] = ox;
				}
				// otherwise
				else {
					// revert to old values
					for (int j=0; j<p.length; j++){
						int a = p[j]/sg;
						int b = p[j]%sg;
						distlist[a][b] = distlist2[a][b];
						distcount[a][b] = distcount2[a][b];
						dim_count[a] = dim_count2[a];
					}
				}
			}
		}
		System.out.println(Integer.toString(counter)+" swaps of "+Integer.toString(n2*nclust)+" were useful");
		// check if the ordering has changed in any of the coordinates
		return counter;
	}
	
	public static void neighbors(final int[][] orderset, int[][] neighborinlist, int[][][] neighboroutlist, double[][] distlist, double[][] distlist2, boolean[][] distcount, boolean[][] distcount2, double[] dim_count, double[] dim_count2, final int[][] ir, final double[] sigma){
		int nclust = orderset.length;
		int n = orderset[0].length;
		// temporary storage for the neighborinlist and distinlist
		ArrayList<ArrayList<Integer>> temp = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> temp2 = new ArrayList<ArrayList<Integer>>();
		for (int i=0; i<n; i++){
			temp.add(new ArrayList<Integer>());
			temp2.add(new ArrayList<Integer>());
		}
		// go through all points and find the nearest neighbors
		for (int i=0; i<n; i++){
			int off = 0;
			for (int j=0; j<(int)sigma[0]; j++){
				if(ir[i][n-j-1]==i){
					off = 1;
				}
				neighborinlist[i][j] = ir[i][n-j-1-off];
				// calculate the distance combined from all dimensions
				for (int k=0; k<nclust; k++){
					distlist[i][j]+=Math.pow(Math.abs(orderset[k][i]-orderset[k][neighborinlist[i][j]]),sigma[2]);
					distlist2[i][j]+=Math.pow(Math.abs(orderset[k][i]-orderset[k][neighborinlist[i][j]]),sigma[2]);
				}
				// count the number of neighbors (points within the neighborhood threshold
				if (Math.pow(distlist[i][j]/nclust,1./sigma[2])<=sigma[1]){
					dim_count[i]+=Math.pow((int)sigma[0]-(double)j,sigma[4]);
					dim_count2[i]+=Math.pow((int)sigma[0]-(double)j,sigma[4]);
					distcount[i][j] = true;
					distcount2[i][j] = true;
				}
				// put this point in the receiver group of the target point
				temp.get(neighborinlist[i][j]).add(i);
				temp2.get(neighborinlist[i][j]).add(j);
			}
		}
		for (int i=0; i<n; i++){
			neighboroutlist[i] = new int[temp.get(i).size()][2];
			for (int j=0; j<temp.get(i).size(); j++){
				neighboroutlist[i][j][0] = temp.get(i).get(j);
				neighboroutlist[i][j][1] = temp2.get(i).get(j);
			}
		}
		//System.out.println(Arrays.toString(sigma));
	}
	
}
