package flib.algorithms.clustering;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.lang.Math;
import flib.math.VectorFun;
import flib.math.random.Shuffle;

public class RFC implements
java.io.Serializable {
	private double[][][] centers;
	private int[] sizes, centerind, treeSizes, indices, indices_old;
	private double[] dimweight;
	private int nclust, n, ntree, it;
	private double balance, dimsum;
	
	//input variables:
	// leafIndices: an array of random forest leafindices for an array of points
	// nclust: number of clusters to form
	// treeSizes: the number of leaves in each tree
	// maxit: maximum number of iterations for the algorithm
	// centerind: the inital center points. Random initialization 
	// if the length of the vector is unequal to nclust
	public RFC(final int[][] leafIndices, int nclust, final int[] treeSizes, double balance, int maxit, final int[] centerind, final double[] dimweight){
		// variable initialization
		this.nclust = nclust;
		this.n = leafIndices.length;
		this.ntree = leafIndices[0].length;
		this.balance = balance;
		this.treeSizes = treeSizes.clone();
		this.centers = new double[nclust][this.ntree][];
		this.dimweight = dimweight.clone();
		this.dimsum = VectorFun.sum(dimweight);
		this.indices = VectorFun.add(new int[this.n],-1);
		for (int i=0; i<this.nclust; i++){
			for (int j=0; j<this.ntree; j++){
				this.centers[i][j] = new double[this.treeSizes[j]];
			}
		}
		this.sizes = new int[this.nclust];
		// the initial cluster size is one
		for (int i=0; i<this.nclust; i++){
			this.sizes[i] = 1;
		}
		// incase we need to still sample random initial centers
		if (centerind.length!=this.nclust){
			this.centerind = new int[nclust];
			this.initialize(leafIndices);
		}
		else {
			this.centerind = centerind.clone();
			for (int i=0; i<this.nclust; i++){
				this.indices[this.centerind[i]] = i;
				for (int j=0; j<this.ntree; j++){
					this.centers[i][j][leafIndices[this.centerind[i]][j]]++;
				}
			}
		}
		// initialize all cluster values
		this.firstIter(leafIndices);
		this.it = maxit;
		// iterate until number of iterations is reaches or until convergence
		for (int i=0; i<maxit; i++){
			if (this.iter(leafIndices)){
				it = i;
				break;
			}
		}
	}
	
	//alternative constructors
	public RFC(final int[][] leafIndices, int nclust, final int[] treeSizes, double balance, int maxit, final double[] dimweight){
		this(leafIndices,nclust,treeSizes,balance,maxit,new int[0],dimweight);
	}
	
	public RFC(final int[][] leafIndices, int nclust, final int[] treeSizes, double balance, int maxit){
		this(leafIndices,nclust,treeSizes,balance,maxit,new int[0],VectorFun.add(new double[treeSizes.length],1));
	}
	
	public RFC(final int[][] leafIndices, int nclust, final int[] treeSizes, double balance){
		this(leafIndices,nclust,treeSizes,balance,200);
	}
	
	public RFC(final int[][] leafIndices, int nclust, final int[] treeSizes){
		this(leafIndices,nclust,treeSizes,2);
	}
	
	// the initialization is related to the kmeans++ initialization
	private void initialize(final int[][] leafIndices){
		Random random = new Random();
		// randomly choose the first cluster center point
		this.centerind[0] = random.nextInt(this.n);
		for (int i=0; i<this.ntree; i++){
			this.centers[0][i][leafIndices[this.centerind[0]][i]]++;
		}
		this.indices[centerind[0]] = 0;
		// vector storing the minimal distance of each point to the current cluster centers
		double[] maxdist = new double[this.n];
		double[] b = new double[this.n];
		double a;
		for (int i=0; i<this.n; i++){
			// a is the number of overlapping leaves with the cluster center
			a = 0;
			for (int j=0; j<this.ntree; j++){
				a+=centers[0][j][leafIndices[i][j]];
			}
			// the subtraction guarantees that the same center can't be
			// chosen twice
			maxdist[i] = 1.0/(a+1.0)-1.0/(this.ntree+1.0);
		}
		// choose further cluster center based on their distance to all current centers
		for (int i=1; i<this.nclust; i++){
			b = VectorFun.cumsum(maxdist);
			int loc = Arrays.binarySearch(b,random.nextDouble()*b[this.n-1]);
			if (loc<0){
				loc = -(loc+1);
			}
			centerind[i] = loc;
			for (int j=0; j<this.ntree; j++){
				this.centers[i][j][leafIndices[this.centerind[i]][j]]++;
			}
			this.indices[loc] = i;
			for (int j=0; j<this.n; j++){
				a = 0;
				for (int k=0; k<this.ntree; k++){
					a+=centers[i][k][leafIndices[j][k]];
				}
				
				a = 1.0/(a+1.0)-1.0/(this.ntree+1.0);
				if (maxdist[j]<a){
					maxdist[j] = a;
				}
			}
		}
	}
	
	public int assignCluster(final int[] point){
		double dist,a;
		int c = 0;
		a = 0;
		for (int i=0; i<this.ntree; i++){
			a+=this.centers[0][i][point[i]];
		}
		// find the cluster which has the greatest overlap with this cluster
		a/=Math.pow((double)this.sizes[0],balance);
		dist = a;
		for (int i=1; i<this.nclust; i++){
			a = 0;
			for (int j=0; j<this.ntree; j++){
				a+=this.centers[i][j][point[j]];
			}
			a/=Math.pow((double)this.sizes[i],balance);
			if (a>dist){
				dist = a;
				c = i;
			}
		}
		return c;
	}
	
	public int[] assignCluster(final int[][] points){
		int l = points.length;
		int[] cl = new int[l];
		for (int i=0; i<l; i++){
			cl[i] = assignCluster(points[i]);
		}
		return cl;
	}
	
	public double[] getDist(final int[] point){
		double[] dist = new double[this.centers.length];
		for (int i=0; i<this.nclust; i++){
			double a = 0;
			for (int j=0; j<this.ntree; j++){
				a+=this.centers[i][j][point[j]];
			}
			// distance isn't used for cluster evaluation
			// but this is the distance measure the algorithm
			// bases its clustering on
			a/=(Math.pow((double)this.sizes[i],balance)*this.dimsum*this.nclust);
			a*=Math.pow(this.n,this.balance-1);
			//a/=(double)this.sizes[i];
			dist[i] = a;
		}
		//VectorFun.multi(dist,1./VectorFun.sum(dist));
		return dist;
	}
	
	public double[][] getDist(final int[][] points){
		final int l = points.length;
		final double[][] cl = new double[l][this.centers.length];
		final int NUM_CORES = Runtime.getRuntime().availableProcessors();
		ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
		try {
			for (int j=0; j<NUM_CORES; j++){
				final int j2 = j;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						try{
							for (int i=j2; i<l; i+=NUM_CORES){
								cl[i] = getDist(points[i]);
							}
						}
						catch (Throwable t){
							System.out.println("Don't want to be here");
							//t.printStackTrace();
						}
					}
				});
			}
		}
		finally {
			exec.shutdown();
		}
		//exec.awaitTermination(time, time_unit);
		while(!exec.isTerminated()){
			// wait
		}
		return cl;
	}
	
	private void firstIter(final int[][] leafIndices){
		int[] rp = Shuffle.randPerm(this.n);
		for (int i=0; i<this.n; i++){
			if (this.indices[rp[i]]==-1){
				this.indices[rp[i]] = assignCluster(leafIndices[rp[i]]);
				this.sizes[indices[rp[i]]]++;
				for (int j=0; j<this.ntree; j++){
					this.centers[indices[rp[i]]][j][leafIndices[rp[i]][j]]++;
				}
			}
		}
	}
	
	// the iteration used for the clustering algorithm once everything has been initialized
	private boolean iter(final int[][] leafIndices){
		// store the old cluster indices
		this.indices_old = this.indices.clone();
		// first assign points to the current clusters
		//this.indices = assignCluster(leafIndices);
		int[] r = Shuffle.randPerm(this.n);
		for (int i=0; i<this.n; i++){
			// remove this point from its current cluster
			this.sizes[indices[r[i]]]--;
			for (int j=0; j<this.ntree; j++){
				this.centers[indices[r[i]]][j][leafIndices[r[i]][j]]--;
			}
			this.indices[r[i]] = assignCluster(leafIndices[r[i]]);
			this.sizes[indices[r[i]]]++;
			for (int j=0; j<this.ntree; j++){
				this.centers[indices[r[i]]][j][leafIndices[r[i]][j]]++;
			}
		}
		return Arrays.equals(indices,indices_old);
	}
	
	public int getIter(){
		return this.it;
	}
	
	public int[] getSizes(){
		return this.sizes.clone();
	}
	
	public int[] getCenterInd(){
		return this.centerind.clone();
	}
	
	public double[][][] getCenters(){
		return this.centers;
	}
}