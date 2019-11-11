package flib.algorithms.clustering;

import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.lang.Math;
import flib.math.RankSort;
import flib.math.VectorFun;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import sparse.eigenvolvers.java.Lobpcg;
import sparse.eigenvolvers.java.OperatorPrecCG;

public class SpectralRFC {
	
	public static double[][] distanceVectors(final int[][] leafindices, int ndim, final double[] treeweights, double balance, int maxIt, int preIt){
		int n = leafindices.length;
		int ntree = treeweights.length;
		if (ndim>n/5){
			System.out.println("To many dimensions required...");
			return new double[1][0];
		}
		// create the proximity matrix
		CompRowMatrix mat = forestProximity(leafindices,treeweights,balance);
		return eigenVectors(mat,ndim,maxIt,preIt);
	}
	
	public static CompRowMatrix forestProximity(final int[][] leafindices, final double[] treeweights, final double balance){
		final int n = leafindices.length;
		final int ntree = treeweights.length;
		final double[][] fullmat = new double[n][n];
		// list for all points, which other points they overlap with at least once
		final int[][] nz = new int[n][n];
		final int NUM_CORES = Runtime.getRuntime().availableProcessors();
		ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
		try {
			for (int i=0; i<NUM_CORES; i++){
				final int i2 = i;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						try{
							for (int j=i2; j<n; j+=NUM_CORES){
								for (int k=0; k<n; k++){
									nz[j][k] = k;
									for (int l=0; l<ntree; l++){
										if (leafindices[j][l]==leafindices[k][l]){
											fullmat[j][k]+=treeweights[l];
										}
									}
								}
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
		exec = Executors.newFixedThreadPool(NUM_CORES);
		final CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		try {
			for (int i=0; i<NUM_CORES; i++){
				final int i2 = i;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						try{
							for (int j=i2; j<n; j+=NUM_CORES){
								for (int k=0; k<n; k++){
									mat.set(j,k,Math.pow(fullmat[j][k],balance));
								}
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
		return mat;
	}
	
	public static double[][] eigenVectors(final CompRowMatrix mat, int ndim, int maxIt, int preIt){
		int n = mat.numRows();
		// mat should have writable values along its diagonal
		// otherwise diagonal entries will be missing in the final matrix
		CompRowMatrix L = new CompRowMatrix(mat);
		double[] di = new double[n];
		double[] di2 = new double[n];
		// calculate the entries of the diagonal weight sum matrix
		// D
		Iterator<MatrixEntry> it = L.iterator();
		// iterate over all elements of the sparse matrix
		while (it.hasNext()){
			MatrixEntry m = it.next();
			int i = m.row();
			di[i]+=m.get();
		}
		// D^(-1/2)
		for (int i=0; i<n; i++){
			di2[i] = 1/Math.sqrt(di[i]);
		}
		// most calculations can be done efficiently without having to generate 
		// new object classes
		// D-W
		it = L.iterator();
		while (it.hasNext()){
			MatrixEntry m = it.next();
			int i = m.row();
			int j = m.column();
			m.set(-m.get());
			if (i==j){
				m.set(m.get()+di[i]);
			}
		}
		// D^(-1/2)*(D-W)*D^(-1/2)
		it = L.iterator();
		while (it.hasNext()){
			MatrixEntry m = it.next();
			int i = m.row();
			int j = m.column();
			m.set(di2[i]*m.get()*di2[j]);
		}
		// calculate the eigenvalues and eigenvectors
		Lobpcg leig = new Lobpcg();
		OperatorPrecCG operT = new OperatorPrecCG(L);
		// still need to find the effect of this number...
		operT.setCGNumberIterations(preIt);
		leig.setMaxIterations(maxIt);
		leig.setVerbosityLevel(0);
		double[][] block = new double[n][ndim+1];
		Random rng = new Random();
		for (int i=0; i<n; i++){
			for (int j=0; j<ndim+1; j++){
				block[i][j] = rng.nextDouble();
			}
		}
		leig.runLobpcg(new DenseMatrix(block),L,null,operT);
		//leig.runLobpcg(2,L,null,operT);
		//leig.runLobpcg(new DenseMatrix(block),L);
		DenseMatrix dm = leig.getEigenvectors();
		double[][] v = new double[n][ndim];
		for (int i=0; i<n; i++){
			for (int j=0; j<ndim; j++){
				v[i][j] = dm.get(i,j+1);
			}
		}
		return v;
	}
	
	public static CompRowMatrix pointOverlap(final double[] leaves){
		final int n = leaves.length;
		int[][] nz = new int[n][];
		// list of overlapping points (worst case this will be a full square matrix
		ArrayList<TreeSet<Integer>> loc = new ArrayList<TreeSet<Integer>>();
		for (int i=0; i<n; i++){
			loc.add(new TreeSet<Integer>());
		}
		// sort all the leaf indices of the current tree
		RankSort rs = new RankSort(leaves);
		double[] s = rs.getSorted();
		int[] o = rs.getRank();
		int a = 0;
		int b = 0;
		for (int j=1; j<n; j++){
			// whenever subsequent leaf numbers are the same
			// points lie within the same leaf node
			// add the neighboring points to each others
			// neighbor lists
			if (s[j]==s[j-1]){
				b = j+1;
			}
			else {
				for (int k=a; k<b; k++){
					for (int l=k+1; l<b; l++){
						loc.get(o[k]).add(o[l]);
						loc.get(o[l]).add(o[k]);
					}
				}
				a = j;
			}
		}
		for (int k=a; k<b; k++){
			for (int l=k+1; l<b; l++){
				loc.get(o[k]).add(o[l]);
				loc.get(o[l]).add(o[k]);
			}
		}
		// for all points find where they have a leaf overlap
		for (int j=0; j<n; j++){
			// each point definitely shares a leaf with itself
			loc.get(j).add(j);
			// create arrays of all points which the current point overlaps with
			// in any tree
			nz[j] = new int[loc.get(j).size()];
			Iterator<Integer> it = loc.get(j).iterator();
			int counter = 0;
			while (it.hasNext()){
				nz[j][counter] = it.next();
				counter++;
			}
		}
		// create the sparse proximity matrix framework
		CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		a = 0;
		b = 0;
		for (int j=1; j<n; j++){
			// whenever subsequent leaf numbers are the same
			// points lie within the same leaf node
			// add the neighboring points to each others
			// neighbor lists
			if (s[j]==s[j-1]){
				b = j+1;
			}
			else {
				for (int k=a; k<b; k++){
					for (int l=k+1; l<b; l++){
						mat.set(o[k],o[l],1);
						mat.set(o[l],o[k],1);
					}
				}
				a = j;
			}
		}
		for (int k=a; k<b; k++){
			for (int l=k+1; l<b; l++){
				mat.set(o[k],o[l],1);
				mat.set(o[l],o[k],1);
			}
		}
		for (int j=0; j<n; j++){
			mat.set(j,j,1);
		}
		return mat;
	}
	
	public static CompRowMatrix forestMat(final int[][] leafindices, final double balance){
		final int n = leafindices.length;
		final int ntree = leafindices[0].length;
		final CompRowMatrix[] pointset = new CompRowMatrix[ntree];
		final int[][] nz = new int[ntree][ntree];
		final int NUM_CORES = Runtime.getRuntime().availableProcessors();
		final double[][] leaves = new double[ntree][n];
		for (int i=0; i<n; i++){
			for (int j=0; j<ntree; j++){
				leaves[j][i] = leafindices[i][j];
			}
		}
		ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
		try {
			for (int i=0; i<NUM_CORES; i++){
				final int i2 = i;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						try{
							for (int j=i2; j<ntree; j+=NUM_CORES){
								pointset[j] = pointOverlap(leaves[j]);
								for (int k=0; k<ntree; k++){
									nz[j][k] = k;
								}
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
		final double[][] forestmat = new double[ntree][ntree];
		exec = Executors.newFixedThreadPool(NUM_CORES);
		try {
			for (int i=0; i<NUM_CORES; i++){
				final int i2 = i;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						try{
							for (int j=i2; j<ntree; j+=NUM_CORES){
								for (int k=0; k<ntree; k++){
									// go through all points in the smaller matrix
									Iterator<MatrixEntry> it = pointset[j].iterator();
									while (it.hasNext()){
										MatrixEntry m = it.next();
										int x = m.row();
										int y = m.column();
										if(pointset[k].get(x,y)>0){
											forestmat[j][k]++;
										}
									}
								}
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
		exec = Executors.newFixedThreadPool(NUM_CORES);
		final CompRowMatrix mat = new CompRowMatrix(ntree,ntree,nz);
		try {
			for (int i=0; i<NUM_CORES; i++){
				final int i2 = i;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						try{
							for (int j=i2; j<ntree; j+=NUM_CORES){
								for (int k=0; k<ntree; k++){
									mat.set(j,k,Math.pow(forestmat[j][k],balance));
								}
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
		return mat;
	}

	public static CompRowMatrix forestProximity2(final int[][] leafindices, final double[] treeweights, double balance){
		int n = leafindices.length;
		int ntree = treeweights.length;
		double w = VectorFun.sum(treeweights);
		// list for all points, which other points they overlap with at least once
		int[][] nz = new int[n][];
		// list of overlapping points (worst case this will be a full square matrix
		ArrayList<TreeSet<Integer>> loc = new ArrayList<TreeSet<Integer>>();
		for (int i=0; i<n; i++){
			loc.add(new TreeSet<Integer>());
		}
		// for all trees
		for (int i=0; i<ntree; i++){
			if (treeweights[i]>0){
				// get the collection of leaf indices
				double[] leaves = new double[n];
				for (int j=0; j<n; j++){
					leaves[j] = leafindices[j][i];
				}
				// sort all the leaf indices of the current tree
				RankSort rs = new RankSort(leaves);
				double[] s = rs.getSorted();
				int[] o = rs.getRank();
				int a = 0;
				int b = 0;
				for (int j=1; j<n; j++){
					// whenever subsequent leaf numbers are the same
					// points lie within the same leaf node
					// add the neighboring points to each others
					// neighbor lists
					if (s[j]==s[j-1]){
						b = j+1;
					}
					else {
						for (int k=a; k<b; k++){
							for (int l=k+1; l<b; l++){
								loc.get(o[k]).add(o[l]);
								loc.get(o[l]).add(o[k]);
							}
						}
						a = j;
					}
				}
				for (int k=a; k<b; k++){
					for (int l=k+1; l<b; l++){
						loc.get(o[k]).add(o[l]);
						loc.get(o[l]).add(o[k]);
					}
				}
			}
		}
		// for all points find where they have a leaf overlap
		for (int i=0; i<n; i++){
			// each point definitely shares a leaf with itself
			loc.get(i).add(i);
			// create arrays of all points which the current point overlaps with
			// in any tree
			nz[i] = new int[loc.get(i).size()];
			Iterator<Integer> it = loc.get(i).iterator();
			int counter = 0;
			while (it.hasNext()){
				nz[i][counter] = it.next();
				counter++;
			}
		}
		// create the sparse proximity matrix framework
		CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		// go through all leaves and treees again
		for (int i=0; i<ntree; i++){
			double[] leaves = new double[n];
			for (int j=0; j<n; j++){
				leaves[j] = leafindices[j][i];
			}
			RankSort rs = new RankSort(leaves);
			double[] s = rs.getSorted();
			int[] o = rs.getRank();
			int a = 0;
			int b = 0;
			for (int j=1; j<n; j++){
				// each time there is an overlap, increase the counter
				// at the given proximity matrix location
				if (s[j]==s[j-1]){
					b = j+1;
				}
				else {
					for (int k=a; k<b; k++){
						for (int l=k+1; l<b; l++){
							// still a bit uncertain at the last time checked
							// we should be adding +1 to both symmetric locations, yes?
							mat.set(o[k],o[l],mat.get(o[k],o[l])+treeweights[i]);
							mat.set(o[l],o[k],mat.get(o[l],o[k])+treeweights[i]);
						}
					}
					a = j;
				}
			}
			for (int k=a; k<b; k++){
				for (int l=k+1; l<b; l++){
					// still a bit uncertain at the last time checked
					// we should be adding +1 to both symmetric locations, yes?
					mat.set(o[k],o[l],mat.get(o[k],o[l])+treeweights[i]);
					mat.set(o[l],o[k],mat.get(o[l],o[k])+treeweights[i]);
				}
			}
		}
		for (int i=0; i<n; i++){
			mat.set(i,i,w);
		}
		// d is a reference to the proximity matrix data
		// we want to see if we have a full matrix or not
		double[] d = mat.getData();
		// adjust the weight matrix
		for (int i=0; i<d.length; i++){
			d[i] = Math.pow(d[i],balance);
		}
		System.out.print("Number of RF proximity components: "+Integer.toString(d.length)+"\n");
		return mat;
	}
}
		
		
		