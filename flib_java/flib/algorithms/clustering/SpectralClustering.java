package flib.algorithms.clustering;

import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;
import flib.math.RankSort;
import flib.math.VectorFun;
import flib.math.SortPair;
import flib.math.SortPair2;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import sparse.eigenvolvers.java.Lobpcg;
import sparse.eigenvolvers.java.OperatorPrecCG;


public class SpectralClustering {
	
	public static double cluster(final double[][] S, int[] labels){
		if (S.length<2){
			return 0;
		}
		// we assume that S is symmetric
		// start creating the Laplacian
		// size of the square matrix
		int n = S.length;
		double[] di = new double[n];
		double[] di2 = new double[n];
		// calculate the entries of the diagonal weight sum matrix
		// D, D^(-1/2)
		for (int i=0; i<n; i++){
			di[i] = VectorFun.sum(S[i]);
			di2[i] = 1/Math.sqrt(di[i]);
		}
		// most calculations can be done efficiently without having to generate 
		// new object classes
		// D-W
		double[][] L = new double[n][n];
		for (int i = 0; i<n; i++){
			for (int j=0; j<n; j++){
				L[i][j] = -S[i][j];
			}
			L[i][i]+=di[i];
		}
		double[][] diff = new double[n][n];
		// D^(-1/2)*(D-W)
		for (int i = 0; i<n; i++){
			for (int j=0; j<n; j++){
				diff[i][j] = di2[i]*L[i][j];
			}
		}
		// D^(-1/2)*(D-W)*D^(-1/2)
		for (int i = 0; i<n; i++){
			for (int j=0; j<n; j++){
				L[i][j] = diff[i][j]*di2[j];
			}
		}
		// calculate the eigenvalues and eigenvectors
		EigenvalueDecomposition eig = new EigenvalueDecomposition(DoubleFactory2D.dense.make(L));
		// the eigenvector of the second smallest eigenvalue
		double[] v = eig.getV().viewColumn(1).toArray();
		// order of the classes according to the eigenvector
		int[] rv = (new RankSort(v)).getRank();
		// find the minimal cut or maximal nassoc
		double[] min = new double[2];
		double NAA = 0, NBB = 0, NAV = 0, NBV = 0, Nassoc = 0;
		// initilize the whole setup
		// A contains a single entry
		int a, b;
		a = rv[0];
		NAA = S[a][a];
		NAV = VectorFun.sum(S[a]);
		// B contains the rest
		for (int i=1; i<n; i++){
			b = rv[i];
			NBV+=VectorFun.sum(S[b]);
		}
		NBB = NBV-VectorFun.sum(S[a])+S[a][a];
		Nassoc = NAA/NAV+NBB/NBV;
		min[0] = Nassoc;
		// go through possible splits given the eigenvector sorting
		for (int i=1; i<n-1; i++){
			a = rv[i];
			double c = VectorFun.sum(S[a]);
			NAV+=c;
			NBV-=c;
			for (int j=0; j<=i; j++){
				b = rv[j];
				NAA+=2*S[a][b];
			}
			NAA-=S[a][a];
			for (int j=i; j<n; j++){
				b = rv[j];
				NBB-=2*S[a][b];
			}
			NBB+=S[a][a];
			Nassoc = NAA/NAV+NBB/NBV;
			if (i==labels.length/2){
			//if (Nassoc>min[0]){
				min[0] = Nassoc*i;
				min[1] = i;
			}
		}
		// start assigning all labels to the two new classes
		for (int i=0; i<n; i++){
			if (i>min[1]){
			/*if (i>labels.length/2){*/
				labels[rv[i]] = 1;
			}
		}
		return min[0];
		//return labels.length;
	}
	
	// for !three! or more clusters
	public static int[] cluster(final double[][] S, int nc, double sigma){
		// if the cluster is a final cluster
		ArrayList<Boolean> finclust = new ArrayList<Boolean>();
		// list of assoc values
		TreeSet<SortPair> assoc = new TreeSet<SortPair>();
		// points contained in this cluster
		ArrayList<int[]> points = new ArrayList<int[]>();
		// labels of the contained points
		ArrayList<int[]> ind = new ArrayList<int[]>();
		finclust.add(true);
		int[] p = new int[S.length];
		for (int i=0; i<p.length; i++){
			p[i] = i;
		}
		points.add(p.clone());
		int[] lab = new int[p.length];
		if (sigma==0){
			assoc.add(new SortPair(cluster(S,lab),0));
		}
		else {
			assoc.add(new SortPair(rankCluster(S,lab,sigma),0));
		}
		ind.add(lab.clone());
		int counter = 2;
		while (counter<nc){
			// find the current split with the highest assoc value
			int index = (int)assoc.last().getOriginalIndex();
			int l1 = VectorFun.sum(ind.get(index));
			int l0 = ind.get(index).length-l1;
			// point indices giving the best split
			int[] p0 = new int[l0];
			int[] p1 = new int[l1];
			int count = 0;
			for (int i=0; i<ind.get(index).length; i++){
				if (ind.get(index)[i]==0){
					p0[count] = points.get(index)[i];
					count++;
				}
			}
			count = 0;
			for (int i=0; i<ind.get(index).length; i++){
				if (ind.get(index)[i]==1){
					p1[count] = points.get(index)[i];
					count++;
				}
			}
			// prepare the new assoc list
			assoc.remove(assoc.last());
			assoc.add(new SortPair(0,index));
			// the cluster being split isn't a final cluster any more
			finclust.set(index,false);
			// caculate the assoc values of both children
			double[][] S2 = new double[p0.length][p0.length];
			for (int i=0; i<S2.length; i++){
				for (int j=0; j<S2.length; j++){
					S2[i][j] = S[p0[i]][p0[j]];
				}
			}
			finclust.add(true);
			lab = new int[l0];
			assoc.add(new SortPair(cluster(S2,lab),finclust.size()-1));
			points.add(p0.clone());
			ind.add(lab.clone());
			S2 = new double[p1.length][p1.length];
			for (int i=0; i<S2.length; i++){
				for (int j=0; j<S2.length; j++){
					S2[i][j] = S[p1[i]][p1[j]];
				}
			}
			finclust.add(true);
			lab = new int[l1];
			if (sigma==0){
				assoc.add(new SortPair(cluster(S2,lab),finclust.size()-1));
			}
			else {
				assoc.add(new SortPair(rankCluster(S2,lab,sigma),finclust.size()-1));
			}
			points.add(p1.clone());
			ind.add(lab.clone());
			counter++;
		}
		int[] indices = new int[S.length];
		counter = 0;
		for (int i=0; i<finclust.size()-2; i++){
			if (finclust.get(i)){
				for (int j=0; j<points.get(i).length; j++){
					indices[points.get(i)[j]] = counter;
				}
				counter++;
			}
		}
		// the last two additions to the point list are the final clusters
		int a = (int)assoc.last().getOriginalIndex(), b = 0;
		if (a!=finclust.size()-1){
			b = finclust.size()-1;
		}
		else {
			b = finclust.size()-2;
		}
		for (int i=0; i<points.get(b).length; i++){
			indices[points.get(b)[i]] = counter;
		}
		counter++;
		for (int i=0; i<points.get(a).length; i++){
			indices[points.get(a)[i]] = counter+ind.get(a)[i];
		}
		return indices;
	}
	
	public static double rankCluster(final double[][] S, int[] labels, double sigma){
		int n = S.length;
		double[][] S2 = new double[n][n];
		for (int i=0; i<n; i++){
			int[] o = (new RankSort(S[i])).getRank();
			for (int j=0; j<n; j++){
				S2[i][o[j]] = n-j-1;
			}
		}
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				if (S2[i][j]>S2[j][i]){
					S2[i][j] = S2[j][i];
				}
			}
		}
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				if (S2[i][j]>sigma){
					S2[i][j] = 0;
				}
				else {
					S2[i][j] = 1;
				}
			}
		}
		return cluster(S,labels);
	}
	
	public static double cluster(final CompRowMatrix mat, int[] labels, int maxIt, int preIt){
		// we assume that mat is symmetric
		// start creating the Laplacian
		// size of the square matrix
		int n = mat.numRows();
		// to prevent lobpcg from crashing the whole environment
		// because blocksize would be too large for input matrix
		if (n<11){
			return cluster(convert(mat),labels);
		}
		else {
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
			double[][] block = new double[n][2];
			Random rng = new Random();
			for (int i=0; i<n; i++){
				block[i][0] = 1;
				//block[i][0] = rng.nextDouble();
				block[i][1] = rng.nextDouble();
			}
			leig.runLobpcg(new DenseMatrix(block),L,null,operT);
			//leig.runLobpcg(2,L,null,operT);
			//leig.runLobpcg(new DenseMatrix(block),L);
			DenseMatrix dm = leig.getEigenvectors();
			double[] v = new double[n];
			// the eigenvector of the second smallest eigenvalue
			for (int i=0; i<n; i++){
				v[i] = dm.get(i,1);
			}
			// order of the classes according to the eigenvector
			int[] rv = (new RankSort(v)).getRank();
			// find the minimal cut or maximal nassoc
			double[] min = new double[2];
			double NAA = 0, NBB = 0, NAV = 0, NBV = 0, Nassoc = 0;
			// initilize the whole setup
			// A contains a single entry
			int a, b;
			TreeSet<Integer> ga = new TreeSet<Integer>();
			int[] c = mat.getColumnIndices();
			int[] r = mat.getRowPointers();
			double[] d = mat.getData();
			a = rv[0];
			ga.add(a);
			NAA = mat.get(a,a);
			NAV+=di[a];
			// B contains the rest
			for (int i=0; i<n; i++){
				NBV+=di[i];
			}
			NBV-=NAV;
			NBB = NBV-NAV+mat.get(a,a);
			Nassoc = NAA/NAV+NBB/NBV;
			min[0] = Nassoc;
			// go through possible splits given the eigenvector sorting
			for (int i=1; i<n-1; i++){
				a = rv[i];
				ga.add(a);
				for (int j=r[a]; j<r[a+1]; j++){
					if (ga.contains(c[j])){
						NAA+=2*d[j];
					}
					else {
						NBB-=2*d[j];
					}
				}
				NAA-=mat.get(a,a);
				NBB-=mat.get(a,a);
				NAV+=di[a];
				NBV-=di[a];
				Nassoc = NAA/NAV+NBB/NBV;
				//if (Nassoc>min[0]){
				if (i==labels.length/2){
					min[0] = Nassoc*i;
					min[1] = i;
				}
			}
			// start assigning all labels to the two new classes
			for (int i=0; i<n; i++){
				if (i>min[1]){
				/*if (i>labels.length/2){*/
					labels[rv[i]] = 1;
				}
			}
			return min[0];
			//return labels.length;
		}
	}
	
	public static double[][] projections(final double[][] S, int nd, double nn, double sigma, int maxIt, int preIt){
		int n = S.length;
		double[][] S2 = new double[n][n];
		if (sigma>0){
			for (int i=0; i<n; i++){
				int[] o = (new RankSort(S[i])).getRank();
				for (int j=0; j<n; j++){
					S2[i][o[j]] = n-j-1;
				}
			}
			for (int i=0; i<n; i++){
				for (int j=0; j<n; j++){
					if (S2[i][j]>S2[j][i]){
						S2[i][j] = S2[j][i];
					}
				}
			}
			for (int i=0; i<n; i++){
				for (int j=0; j<n; j++){
					if (S2[i][j]>sigma){
						S2[i][j] = 0;
					}
					else {
						S2[i][j] = 1;
					}
				}
			}
		}
		else {
			for (int i=0; i<n; i++){
				S2[i] = S[i].clone();
			}
		}
		if (n<5*nd+1){
			// we assume that S is symmetric
			// start creating the Laplacian
			// size of the square matrix
			double[] di = new double[n];
			double[] di2 = new double[n];
			// calculate the entries of the diagonal weight sum matrix
			// D, D^(-1/2)
			for (int i=0; i<n; i++){
				di[i] = VectorFun.sum(S2[i]);
				di2[i] = 1/Math.sqrt(di[i]);
			}
			// most calculations can be done efficiently without having to generate 
			// new object classes
			// D-W
			double[][] L = new double[n][n];
			for (int i = 0; i<n; i++){
				for (int j=0; j<n; j++){
					L[i][j] = -S2[i][j];
				}
				L[i][i]+=di[i];
			}
			double[][] diff = new double[n][n];
			// D^(-1/2)*(D-W)
			for (int i = 0; i<n; i++){
				for (int j=0; j<n; j++){
					diff[i][j] = di2[i]*L[i][j];
				}
			}
			// D^(-1/2)*(D-W)*D^(-1/2)
			for (int i = 0; i<n; i++){
				for (int j=0; j<n; j++){
					L[i][j] = diff[i][j]*di2[j];
				}
			}
			// calculate the eigenvalues and eigenvectors
			EigenvalueDecomposition eig = new EigenvalueDecomposition(DoubleFactory2D.dense.make(L));
			// the eigenvector of the second smallest eigenvalue
			double[][] v = new double[nd][n];
			for (int i=0; i<nd; i++){
				v[i] = eig.getV().viewColumn(i+1).toArray();
			}
			return v;
		}
		else {
			// mat should have writable values along its diagonal
			// otherwise diagonal entries will be missing in the final matrix
			CompRowMatrix L = convert(S2);
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
			double[][] block = new double[n][nd+1];
			Random rng = new Random();
			for (int i=0; i<n; i++){
				block[i][0] = 1;
				//block[i][0] = rng.nextDouble();
				for (int j=0; j<nd; j++){
					block[i][j+1] = rng.nextDouble();
				}
			}
			leig.runLobpcg(new DenseMatrix(block),L,null,operT);
			//leig.runLobpcg(2,L,null,operT);
			//leig.runLobpcg(new DenseMatrix(block),L);
			DenseMatrix dm = leig.getEigenvectors();
			double[][] v = new double[nd][n];
			// the eigenvector of the second smallest eigenvalue
			for (int i=0; i<n; i++){
				for (int j=0; j<nd; j++){
					v[j][i] = dm.get(i,j+1);
				}
			}
			return v;
		}
	}
	
	// for !three! or more clusters
	public static int[] cluster(final CompRowMatrix mat, int nc, double sigma, int maxIt, int preIt){
		// if the cluster is a final cluster
		ArrayList<Boolean> finclust = new ArrayList<Boolean>();
		// list of assoc values
		TreeSet<SortPair> assoc = new TreeSet<SortPair>();
		// points contained in this cluster
		ArrayList<int[]> points = new ArrayList<int[]>();
		// labels of the contained points
		ArrayList<int[]> ind = new ArrayList<int[]>();
		finclust.add(true);
		int n = mat.numRows();
		int[] p = new int[n];
		for (int i=0; i<n; i++){
			p[i] = i;
		}
		points.add(p.clone());
		int[] lab = new int[n];
		if (sigma==0){
			assoc.add(new SortPair(cluster(mat,lab,maxIt,preIt),0));
		}
		else {
			assoc.add(new SortPair(rankCluster(mat,lab,sigma,maxIt,preIt),0));
		}
		ind.add(lab.clone());
		int counter = 2;
		while (counter<nc){
			// find the current split with the highest assoc value
			int index = (int)assoc.last().getOriginalIndex();
			int l1 = VectorFun.sum(ind.get(index));
			int l0 = ind.get(index).length-l1;
			// point indices giving the best split
			int[] p0 = new int[l0];
			int[] p1 = new int[l1];
			int count = 0;
			for (int i=0; i<ind.get(index).length; i++){
				if (ind.get(index)[i]==0){
					p0[count] = points.get(index)[i];
					count++;
				}
			}
			count = 0;
			for (int i=0; i<ind.get(index).length; i++){
				if (ind.get(index)[i]==1){
					p1[count] = points.get(index)[i];
					count++;
				}
			}
			// prepare the new assoc list
			assoc.remove(assoc.last());
			assoc.add(new SortPair(0,index));
			// the cluster being split isn't a final cluster any more
			finclust.set(index,false);
			// caculate the assoc values of both children
			// prepare the sparse storage matrices
			int[][] nz0 = new int[l0][];
			int[][] nz1 = new int[l1][];
			TreeSet<SortPair2> g0 = new TreeSet<SortPair2>();
			TreeSet<SortPair2> g1 = new TreeSet<SortPair2>();
			int[] c = mat.getColumnIndices();
			int[] r = mat.getRowPointers();
			double[] d = mat.getData();
			int counter2;
			for (int i=0; i<l0; i++){
				g0.add(new SortPair2(p0[i],i));
			}
			for (int i=0; i<l1; i++){
				g1.add(new SortPair2(p1[i],i));
			}
			for (int i=0; i<l0; i++){
				counter2 = 0;
				for (int j=r[p0[i]]; j<r[p0[i]+1]; j++){
					if (g0.contains(new SortPair2(c[j],0))){
						counter2++;
					}
				}
				nz0[i] = new int[counter2];
				counter2 = 0;
				for (int j=r[p0[i]]; j<r[p0[i]+1]; j++){
					if (g0.contains(new SortPair2(c[j],0))){
						nz0[i][counter2] = (int)g0.floor(new SortPair2(c[j],0)).getOriginalIndex();
						counter2++;
					}
				}
			}
			CompRowMatrix mat0 = new CompRowMatrix(l0,l0,nz0);
			double[] d0 = mat0.getData();
			counter2 = 0;
			for (int i=0; i<p0.length; i++){
				for (int j=r[p0[i]]; j<r[p0[i]+1]; j++){
					if (g0.contains(new SortPair2(c[j],0))){
						d0[counter2] = d[j];
						counter2++;
					}
				}
			}
			lab = new int[l0];
			finclust.add(true);
			assoc.add(new SortPair(cluster(mat0,lab,maxIt,preIt),finclust.size()-1));
			points.add(p0.clone());
			ind.add(lab.clone());
			
			for (int i=0; i<l1; i++){
				counter2 = 0;
				for (int j=r[p1[i]]; j<r[p1[i]+1]; j++){
					if (g1.contains(new SortPair2(c[j],0))){
						counter2++;
					}
				}
				nz1[i] = new int[counter2];
				counter2 = 0;
				for (int j=r[p1[i]]; j<r[p1[i]+1]; j++){
					if (g1.contains(new SortPair2(c[j],0))){
						nz1[i][counter2] = (int)g1.floor(new SortPair2(c[j],0)).getOriginalIndex();
						counter2++;
					}
				}
			}
			CompRowMatrix mat1 = new CompRowMatrix(l1,l1,nz1);
			double[] d1 = mat1.getData();
			counter2 = 0;
			for (int i=0; i<p1.length; i++){
				for (int j=r[p1[i]]; j<r[p1[i]+1]; j++){
					if (g1.contains(new SortPair2(c[j],0))){
						d1[counter2] = d[j];
						counter2++;
					}
				}
			}
			lab = new int[l1];
			finclust.add(true);
			if (sigma==0){
				assoc.add(new SortPair(cluster(mat1,lab,maxIt,preIt),finclust.size()-1));
			}
			else {
				assoc.add(new SortPair(rankCluster(mat1,lab,sigma,maxIt,preIt),finclust.size()-1));
			}
			points.add(p1.clone());
			ind.add(lab.clone());
			counter++;
		}
		int[] indices = new int[n];
		counter = 0;
		for (int i=0; i<finclust.size()-2; i++){
			if (finclust.get(i)){
				for (int j=0; j<points.get(i).length; j++){
					indices[points.get(i)[j]] = counter;
				}
				counter++;
			}
		}
		// the last two additions to the point list are the final clusters
		int a = (int)assoc.last().getOriginalIndex(), b = 0;
		if (a!=finclust.size()-1){
			b = finclust.size()-1;
		}
		else {
			b = finclust.size()-2;
		}
		for (int i=0; i<points.get(b).length; i++){
			indices[points.get(b)[i]] = counter;
		}
		counter++;
		for (int i=0; i<points.get(a).length; i++){
			indices[points.get(a)[i]] = counter+ind.get(a)[i];
		}
		return indices;
	}
	
	public static double cluster(final CompRowMatrix mat, int[] labels){
		return cluster(mat,labels,20,0);
	}
	
	public static int[] cluster(final CompRowMatrix mat, int nc){
		return cluster(mat,nc,0,20,0);
	}
	
	public static double rankCluster(final CompRowMatrix mat, int[] labels, double sigma, int maxIt, int preIt){
		double[] d = mat.getData();
		CompRowMatrix mat2 = new CompRowMatrix(mat);
		double[] d2 = mat2.getData();
		int[] r = mat.getRowPointers();
		int[] c = mat.getColumnIndices();
		for (int i=1; i<r.length; i++){
			double[] a = new double[r[i]-r[i-1]];
			for (int j=r[i-1]; j<r[i]; j++){
				a[j-r[i-1]] = d[j];
			}
			// label the entries of the current row according to their
			// rank
			int[] o = (new RankSort(a)).getRank();
			for (int j=0; j<o.length; j++){
				d2[r[i-1]+o[j]] = o.length-1-j;
			}
		}
		int n = mat.numRows();
		Iterator<MatrixEntry> it = mat2.iterator();
		// iterate over all elements of the sparse matrix
		// and set all values to the maximum value of the matrix and
		// it's transpose
		while (it.hasNext()){
			MatrixEntry m = it.next();
			int i = m.row();
			int j = m.column();
			if(m.get()>mat2.get(j,i)){
				mat2.set(i,j,mat2.get(j,i));
			}
		}
		// make a geometric row in each row by exponentiating the matrix values
		for (int i=0; i<d2.length; i++){
			if (d2[i]>sigma){
				d2[i] = 0;
			}
			else {
				d2[i] = 1;
			}
		}
		int[][] nz = new int[n][];
		for (int i=1; i<r.length; i++){
			int counter = 0;
			for (int j=r[i-1]; j<r[i]; j++){
				if (d2[j]>0){
					counter++;
				}
			}
			nz[i-1] = new int[counter];
			counter = 0;
			for (int j=r[i-1]; j<r[i]; j++){
				if (d2[j]>0){
					nz[i-1][counter] = c[j];
					counter++;
				}
			}
		}
		CompRowMatrix mat3 = new CompRowMatrix(n,n,nz);
		int[] r3 = mat3.getRowPointers();
		double[] d3 = mat3.getData();
		for (int i=1; i<r.length; i++){
			int counter = 0;
			for (int j=r[i-1]; j<r[i]; j++){
				if (d2[j]>0){
					d3[r3[i-1]+counter] = d2[j];
					counter++;
				}
			}
		}
		System.out.print("Number of NN components: "+Integer.toString(d3.length)+"\n");
		return cluster(mat3,labels,maxIt,preIt);
	}
	
	public static CompRowMatrix convert(final double[][] S, double thr){
		int n = S.length;
		int[][] nz = new int[n][];
		for (int i=0; i<n; i++){
			int counter = 0;
			for (int j=0; j<n; j++){
				if (S[i][j]>thr){
					counter++;
				}
			}
			nz[i] = new int[counter];
			counter = 0;
			for (int j=0; j<n; j++){
				if (S[i][j]>thr){
					nz[i][counter] = j;
					counter++;
				}
			}
		}
		CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		double[] d = mat.getData();
		int counter = 0;
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				if (S[i][j]>thr){
					d[counter] = S[i][j];
					counter++;
				}
			}
		}
		return mat;
	}
	
	public static CompRowMatrix convert(final double[][] S){
		int n = S.length;
		int[][] nz = new int[n][];
		for (int i=0; i<n; i++){
			int counter = 0;
			for (int j=0; j<n; j++){
				if (S[i][j]!=0){
					counter++;
				}
			}
			nz[i] = new int[counter];
			counter = 0;
			for (int j=0; j<n; j++){
				if (S[i][j]!=0){
					nz[i][counter] = j;
					counter++;
				}
			}
		}
		CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		double[] d = mat.getData();
		int counter = 0;
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				if (S[i][j]!=0){
					d[counter] = S[i][j];
					counter++;
				}
			}
		}
		return mat;
	}
	
	public static double[][] convert(final CompRowMatrix mat){
		int n = mat.numRows();
		double[][] S = new double[n][n];
		double[] d = mat.getData();
		int[] r = mat.getRowPointers();
		int[] c = mat.getColumnIndices();
		for (int i=0; i<n; i++){
			for (int j=r[i]; j<r[i+1]; j++){
				S[i][c[j]] = d[j];
			}
		}
		return S;
	}
}