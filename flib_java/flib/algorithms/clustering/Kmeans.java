package flib.algorithms.clustering;

import java.util.Random;
import java.util.Arrays;
import flib.math.VectorFun;
import flib.math.random.Shuffle;

public class Kmeans {
	// number of clusters for the kmeans
	private int nclust;
	// the dimensionality of the cluster points
	private int d;
	// number of points investigated
	private int n;
	// distance type. 0: cityblock distance, else euclidean
	private int type;
	// the coordinates of the cluster centers
	private double[][] centers;
	// array containing a points used
	private double[][] points;
	// current cluster indices of all points
	private int[] clind;
	// old cluster indices
	private int[] clind_old;
	// number of iterations needed for the last kmeans run
	private int it;
	
	public Kmeans(final double[][] x, int nclust, int maxit, int type, int start){
		this.nclust = nclust;
		this.d = x[0].length;
		this.n = x.length;
		this.type = type;
		this.points = new double[this.n][this.d];
		for (int i=0; i<this.n; i++){
			this.points[i] = x[i].clone();
		}
		if (start==0){
			kmeanspp();
		}
		else {
			kmeans();
		}
		this.it = run(maxit);
	}
	
	public Kmeans(final double[][] x, int nclust, int maxit, int type){
		this(x, nclust, maxit, type, 0);
	}
	
	public Kmeans(final double[][] x, int nclust, int maxit){
		this(x, nclust, maxit, 1, 0);
	}
	
	public Kmeans(final double[][] x, int nclust){
		this(x, nclust, 200, 1, 0);
	}
	
	// initialization function for kmeans++
	public void kmeanspp(){
		this.centers = new double[this.nclust][this.d];
		Random random = new Random();
		this.centers[0] = this.points[random.nextInt(this.n)].clone();
		// placeholder variables
		double[] a = new double[this.n];
		double[] b = new double[this.n];
		double [] c = new double[this.n];
		double[] t = new double[this.d];
		if (type==0){
				for (int i=0; i<this.n; i++){
					a[i] = VectorFun.sum(VectorFun.abs(VectorFun.sub(this.points[i],this.centers[0])));
				}
		}
		else {
			
			for (int i=0; i<this.n; i++){
				t = VectorFun.sub(this.points[i],this.centers[0]);
				a[i] = VectorFun.sum(VectorFun.mult(t,t));
			}
		}
		for (int i=1; i<this.nclust; i++){
			b = VectorFun.cumsum(a);
			b = VectorFun.div(b,b[this.n-1]);
			int loc = Arrays.binarySearch(b,random.nextDouble());
			if (loc<0){
				loc = -(loc+1);
			}
			centers[i] = this.points[loc].clone();
			if (type==0){
				for (int j=0; j<this.n; j++){
					c[j] = VectorFun.sum(VectorFun.abs(VectorFun.sub(this.points[j],this.centers[i])));
				}
			}
			else {
				for (int j=0; j<this.n; j++){
					t = VectorFun.sub(this.points[j],this.centers[i]);
					c[j] = VectorFun.sum(VectorFun.mult(t,t));
				}
			}
			a = VectorFun.min(a,c);
		}
	}
	
	// initialization function for kmeans... random initial points
	public void kmeans(){
		this.centers = new double[this.nclust][this.d];
		int[] r = Shuffle.randPerm(this.n);
		for (int i=0; i<this.nclust; i++){
			this.centers[i] = this.points[r[i]].clone();
		}
	}
	
	// find which cluster a given point belongs to
	public int assignCluster(double[] point){
		double dist,a;
		int c = 0;
		double[] t = new double[this.d];
		if (type==0){
			dist = VectorFun.sum(VectorFun.abs(VectorFun.sub(point,this.centers[0])));
		}
		else {
			t = VectorFun.sub(point,this.centers[0]);
			dist = VectorFun.sum(VectorFun.mult(t,t));
		}
		for (int i=1; i<this.nclust; i++){
			if (type==0){
				a = VectorFun.sum(VectorFun.abs(VectorFun.sub(point,this.centers[i])));
			}
			else {
				t = VectorFun.sub(point,this.centers[i]);
				a = VectorFun.sum(VectorFun.mult(t,t));
			}
			if (a<dist){
				dist = a;
				c = i;
			}
		}
		return c;
	}
	
	public int[] assignCluster(double[][] points){
		int l = points.length;
		int[] cl = new int[l];
		for (int i=0; i<l; i++){
			cl[i] = assignCluster(points[i]);
		}
		return cl;
	}
	
	// update the centers based on the current cluster assignments
	public void updateCenters(){
		double[][] cent = new double[this.nclust][this.d];
		double[] sum = new double[this.nclust];
		for (int i=0; i<this.n; i++){
			sum[this.clind[i]]++;
			cent[this.clind[i]] = VectorFun.add(cent[clind[i]],this.points[i]);
		}
		for (int i=0; i<this.nclust; i++){
			if (sum[i]>0){
				this.centers[i] = VectorFun.div(cent[i],sum[i]);
			}
		}
	}
	
	// a kmeans iteration step
	public boolean iter(){
		this.clind_old = this.clind.clone();
		for (int i=0; i<this.n; i++){
			clind[i] = assignCluster(this.points[i]);
		}
		updateCenters();		
		return Arrays.equals(clind,clind_old);
	}
	
	public int run(int maxit){
		int count;
		clind = new int[this.n];
		for (count=0; count<maxit; count++){
			if (iter()) break;
		}
		return count;
	}
	
	public void sortCenters(int[] indices){
		double[][] tempcenters = new double[this.nclust][this.d];
		for (int i=0; i<this.nclust; i++){
			tempcenters[i] = this.centers[indices[i]].clone();
		}
		for (int i=0; i<this.nclust; i++){
			this.centers[i] = tempcenters[i];
		}
		for (int i=0; i<this.n; i++){
			this.clind[i] = assignCluster(this.points[i]);
		}
	}
	
	//getter functions
	public double[][] getCenters(){
		double[][] cent = new double[this.nclust][this.d];
		for (int i=0; i<this.nclust; i++){
			cent[i] = this.centers[i].clone();
		}
		return cent;
	}
	
	public int[] getIndices(){
		return this.clind.clone();
	}
	
	public int getIter(){
		return this.it;
	}
	
	public double[][] getDist(){
		double[][] dist = new double[this.nclust][this.d];
		double t;
		for (int i=0; i<this.n; i++){
			for (int j=0; j<this.d; j++){
				if (this.type==0){
					dist[this.clind[i]][j]+=Math.abs(this.points[i][j]-this.centers[this.clind[i]][j]);
				}
				else {
					t = this.points[i][j]-this.centers[this.clind[i]][j];
					dist[this.clind[i]][j]+=t*t;
				}
			}
		}
		return dist;
	}

	public int[] getSize(){
		int[] size = new int[this.nclust];
		for (int i=0; i<this.n; i++){
			size[this.clind[i]]++;
		}
		return size;
	}
}