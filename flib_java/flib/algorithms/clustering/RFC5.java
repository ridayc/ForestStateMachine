package flib.algorithms.clustering;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.lang.Math;
import flib.math.VectorFun;
import flib.math.RankSort;
import flib.math.random.Shuffle;

public class RFC5 implements
java.io.Serializable {
	private int[][] leafindices, neighbors;
	private double[][] proximities;
	private int[] indices, sizes;
	private int nn, nclust;
	private double balance, flip;
	
	//input variables:
	// leafIndices: an array of random forest leafindices for an array of points
	// nclust: number of clusters to form
	// treeSizes: the number of leaves in each tree
	// maxit: maximum number of iterations for the algorithm
	// centerind: the inital center points. Random initialization 
	// if the length of the vector is unequal to nclust
	public RFC5(final int[][] leafIndices, int nclust, int nn, double balance, double flip, int maxit, final int[] centerind){
		// store all necessary variables
		this.n = leafindices.length;
		this.d = leafindices[0].length;
		this.leafindices = new int[n][d];
		for (int i=0; i<n; i++){
			this.leafindices[i] = leafindices[i].clone();
		}
		this.nclust = nclust;
		this.nn = nn;
		this.balance = balance;
		this.flip = flip;
		this.proximities = new double[n][n];
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				for (int k=0; k<d; k++){
					if (leafindices[i][k]==leafindices[j][k]){
						this.proximities[i][j]++;
					}
				}
			}
		}
		this.neighbors = new int[n][nn];
		for (int i=0; i<n; i++){
			int[] temp = RankSort(proximities[i]).getRank();
			int off = 0;
			for (int j=0; j<nn; j++){
				if (temp[n-1-j-off]==i){
					off = 1;
				}
				neighbors[i][j] = temp[n-1-j-off];
			}
		}
		int[] indices = new int[n];
		int[] sizes = new int[nclust];
		int[] rp = Shuffle.randPerm(n);
		// randomly initialize all clusters
		for (int i=0; i<n; i++){
			indices[i] = rp[i]%nclust;
			sizes[indices[i] ]++;
		}
		// start iterating
		for (int it=0; it<maxit; it++){
			
		