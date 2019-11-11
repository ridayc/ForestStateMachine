package flib.algorithms.clustering;

import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import flib.math.RankSort;
import flib.math.SortPair;
import flib.math.SortPair2;
import flib.math.VectorConv;
import flib.math.VectorAccess;
import flib.math.random.Shuffle;

import flib.algorithms.clustering.SpectralClustering;

public class SpectralMatrices {
	public static int[] forestProximity(final int[][] leafindices, int nc, double nn, double sigma, int maxIt, int preIt){
		// number of points in the matrix
		int n = leafindices.length;
		// number of trees for which we have to make leaf comparisons between points
		int ntree = leafindices[0].length;
		// list for all points, which other points they overlap with at least once
		int[][] nz = new int[n][];
		// list 
		ArrayList<TreeSet<Integer>> loc = new ArrayList<TreeSet<Integer>>();
		for (int i=0; i<n; i++){
			loc.add(new TreeSet<Integer>());
		}
		// for all trees
		for (int i=0; i<ntree; i++){
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
							mat.set(o[k],o[l],mat.get(o[k],o[l])+1);
							mat.set(o[l],o[k],mat.get(o[l],o[k])+1);
						}
					}
					a = j;
				}
			}
			for (int k=a; k<b; k++){
				for (int l=k+1; l<b; l++){
					// still a bit uncertain at the last time checked
					// we should be adding +1 to both symmetric locations, yes?
					mat.set(o[k],o[l],mat.get(o[k],o[l])+1);
					mat.set(o[l],o[k],mat.get(o[l],o[k])+1);
				}
			}
		}
		for (int i=0; i<n; i++){
			mat.set(i,i,ntree);
		}
		// d is a reference to the proximity matrix data
		double[] d = mat.getData();
		for (int i=0; i<d.length; i++){
			d[i] = Math.pow(d[i],nn);
		}
		//System.out.print(Arrays.toString(mat.getRowPointers()));
		//System.out.print(Arrays.toString(mat.getColumnIndices()));
		System.out.print("Number of RF proximity components: "+Integer.toString(d.length)+"\n");
		if (nc<=2){
			int[] ind = new int[n];
			if (sigma==0){
				SpectralClustering.cluster(mat,ind,maxIt,preIt);
			}
			else {
				SpectralClustering.rankCluster(mat,ind,sigma,maxIt,preIt);
			}
			return ind;
		}
		else {
			return SpectralClustering.cluster(mat,nc,sigma,maxIt,preIt);
		}
	}
	
	public static int[] minRankDist(final double[][] set, int nc, int dist, double sigma){
		int n = set.length;
		int dim = set[0].length;
		int[][] nz = new int[n][];
		ArrayList<TreeSet<SortPair2>> loc = new ArrayList<TreeSet<SortPair2>>();
		for (int i=0; i<n; i++){
			loc.add(new TreeSet<SortPair2>());
		}
		for (int i=0; i<dim; i++){
			double[] slice = new double[n];
			for (int j=0; j<n; j++){
				slice[j] = set[j][i];
			}
			RankSort rs = new RankSort(slice);
			double[] s = rs.getSorted();
			int[] o = rs.getRank();
			for (int j=0; j<n; j++){
				int a = j-dist;
				if (a<0){
					a = 0;
				}
				// don't forget to +1. Otherwise the distance isn't symmetric on both sides
				// of the current point
				int b = j+dist+1;
				if (b>n){
					b = n;
				}
				for (int k=a; k<b; k++){
					if (!loc.get(o[j]).add(new SortPair2(o[k],Math.abs(j-k)))){
						// if this location is already contained, we compare
						// to the current distance and reinsert the minimum value
						double m = loc.get(o[j]).floor(new SortPair2(o[k],0)).getOriginalIndex();
						double m2 = Math.abs(j-k);
						if (m2<m){
							loc.get(o[j]).remove(new SortPair2(o[k],0));
							loc.get(o[j]).add(new SortPair2(o[k],m2));
						}
					}
				}
			}
		}
		for (int i=0; i<n; i++){
			nz[i] = new int[loc.get(i).size()];
			Iterator<SortPair2> it = loc.get(i).iterator();
			int counter = 0;
			while (it.hasNext()){
				nz[i][counter] = (int)it.next().getValue();
				counter++;
			}
			nz[i] = VectorConv.double2int((new RankSort(VectorConv.int2double(nz[i]))).getSorted());
		}
		CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		double[] d = mat.getData();
		int[] r = mat.getRowPointers();
		for (int i=0; i<n; i++){
			Iterator<SortPair2> it = loc.get(i).iterator();
			double[] temp = new double[loc.get(i).size()];
			double[] ord = new double [loc.get(i).size()];
			int counter = 0;
			while (it.hasNext()){
				SortPair2 sp = it.next();
				temp[counter] = Math.exp(-sp.getOriginalIndex()/sigma);
				ord[counter] = sp.getValue();
				counter++;
			}
			int[] o = (new RankSort(ord)).getRank();
			for (int j=0; j<o.length; j++){
				d[r[i]+j] = temp[o[j]];
			}
		}
		if (nc<=2){
			int[] ind = new int[n];
			SpectralClustering.cluster(mat,ind,100,0);
			return ind;
		}
		else {
			return SpectralClustering.cluster(mat,nc,0,100,0);
		}
	}
	
	public static int[] minmaxRankDist(final double[][] set, int nc, int dist, double sigma){
		int n = set.length;
		int dim = set[0].length;
		int[][] nz = new int[n][];
		ArrayList<TreeSet<SortPair>> loc = new ArrayList<TreeSet<SortPair>>();
		for (int i=0; i<n; i++){
			loc.add(new TreeSet<SortPair>());
		}
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				double v = Math.abs(set[i][0]-set[j][0]);
				for (int k=1; k<dim; k++){
					double v2 = Math.abs(set[i][k]-set[j][k]);
					if (v2>v){
						v = v2;
					}
				}
				if (loc.get(i).size()<dist||v<loc.get(i).last().getValue()){
					loc.get(i).add(new SortPair(v,j));
					if (loc.get(i).size()>dist){
						loc.get(i).remove(loc.get(i).last());
					}
				}
			}
		}
		for (int i=0; i<n; i++){
			// duplicate all nearest neighbors, so the sparse distance matrix
			// becomes symmetric
			Iterator<SortPair> it = loc.get(i).iterator();
			while (it.hasNext()){
				SortPair temp = it.next();
				loc.get((int)temp.getOriginalIndex()).add(new SortPair(temp.getValue(),i));
			}
		}
		for (int i=0; i<n; i++){
			nz[i] = new int[loc.get(i).size()];
			Iterator<SortPair> it = loc.get(i).iterator();
			int counter = 0;
			while (it.hasNext()){
				nz[i][counter] = (int)it.next().getOriginalIndex();
				counter++;
			}
			nz[i] = VectorConv.double2int((new RankSort(VectorConv.int2double(nz[i]))).getSorted());
		}
		CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		double[] d = mat.getData();
		int[] r = mat.getRowPointers();
		for (int i=0; i<n; i++){
			Iterator<SortPair> it = loc.get(i).iterator();
			double[] temp = new double[loc.get(i).size()];
			double[] ord = new double [loc.get(i).size()];
			int counter = 0;
			while (it.hasNext()){
				SortPair sp = it.next();
				temp[counter] = Math.exp(-sp.getValue()/sigma);
				ord[counter] = sp.getOriginalIndex();
				counter++;
			}
			int[] o = (new RankSort(ord)).getRank();
			for (int j=0; j<o.length; j++){
				d[r[i]+j] = temp[o[j]];
			}
		}
		if (nc<=2){
			int[] ind = new int[n];
			SpectralClustering.cluster(mat,ind,100,0);
			return ind;
		}
		else {
			return SpectralClustering.cluster(mat,nc,0,100,0);
		}
	}
	
	public static int[] maxRankDist(final double[][] set, int nc, int dist, double sigma){
		int n = set.length;
		int dim = set[0].length;
		int[][] nz = new int[n][];
		ArrayList<TreeSet<SortPair>> loc = new ArrayList<TreeSet<SortPair>>();
		for (int i=0; i<n; i++){
			loc.add(new TreeSet<SortPair>());
		}
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				double v = Math.abs(set[i][0]-set[j][0]);
				for (int k=1; k<dim; k++){
					double v2 = Math.abs(set[i][k]-set[j][k]);
					if (v2>v){
						v = v2;
					}
				}
				if (loc.get(i).size()<dist||v>loc.get(i).last().getValue()){
					loc.get(i).add(new SortPair(v,j));
					if (loc.get(i).size()>dist){
						loc.get(i).remove(loc.get(i).first());
					}
				}
			}
		}
		for (int i=0; i<n; i++){
			loc.get(i).add(new SortPair(0,i));
			// duplicate all nearest neighbors, so the sparse distance matrix
			// becomes symmetric
			Iterator<SortPair> it = loc.get(i).iterator();
			while (it.hasNext()){
				SortPair temp = it.next();
				loc.get((int)temp.getOriginalIndex()).add(new SortPair(temp.getValue(),i));
			}
		}
		for (int i=0; i<n; i++){
			nz[i] = new int[loc.get(i).size()];
			Iterator<SortPair> it = loc.get(i).iterator();
			int counter = 0;
			while (it.hasNext()){
				nz[i][counter] = (int)it.next().getOriginalIndex();
				counter++;
			}
			nz[i] = VectorConv.double2int((new RankSort(VectorConv.int2double(nz[i]))).getSorted());
		}
		CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		double[] d = mat.getData();
		int[] r = mat.getRowPointers();
		for (int i=0; i<n; i++){
			Iterator<SortPair> it = loc.get(i).iterator();
			double[] temp = new double[loc.get(i).size()];
			double[] ord = new double [loc.get(i).size()];
			int counter = 0;
			while (it.hasNext()){
				SortPair sp = it.next();
				temp[counter] = Math.pow(sp.getValue(),sigma);
				ord[counter] = sp.getOriginalIndex();
				counter++;
			}
			int[] o = (new RankSort(ord)).getRank();
			for (int j=0; j<o.length; j++){
				d[r[i]+j] = temp[o[j]];
			}
		}
		if (nc<=2){
			int[] ind = new int[n];
			SpectralClustering.cluster(mat,ind,100,0);
			return ind;
		}
		else {
			return SpectralClustering.cluster(mat,nc,0,100,0);
		}
	}
	
	public static int[] medRankDist(final double[][] set, int nc, int dist, double sigma){
		int n = set.length;
		int dim = set[0].length;
		int[][] nz = new int[n][];
		ArrayList<TreeSet<SortPair>> loc = new ArrayList<TreeSet<SortPair>>();
		for (int i=0; i<n; i++){
			loc.add(new TreeSet<SortPair>());
		}
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				double v = Math.abs(set[i][0]-set[j][0]);
				double w = Math.abs(set[i][0]-set[j][0]);
				for (int k=1; k<dim; k++){
					double v2 = Math.abs(set[i][k]-set[j][k]);
					if (v2>v){
						v = v2;
					}
					if (v2<w){
						w = v2;
					}
				}
				v = (v+w)*0.5;
				if (loc.get(i).size()<dist||v<loc.get(i).last().getValue()){
					loc.get(i).add(new SortPair(v,j));
					if (loc.get(i).size()>dist){
						loc.get(i).remove(loc.get(i).last());
					}
				}
			}
		}
		for (int i=0; i<n; i++){
			// duplicate all nearest neighbors, so the sparse distance matrix
			// becomes symmetric
			Iterator<SortPair> it = loc.get(i).iterator();
			while (it.hasNext()){
				SortPair temp = it.next();
				loc.get((int)temp.getOriginalIndex()).add(new SortPair(temp.getValue(),i));
			}
		}
		for (int i=0; i<n; i++){
			nz[i] = new int[loc.get(i).size()];
			Iterator<SortPair> it = loc.get(i).iterator();
			int counter = 0;
			while (it.hasNext()){
				nz[i][counter] = (int)it.next().getOriginalIndex();
				counter++;
			}
			nz[i] = VectorConv.double2int((new RankSort(VectorConv.int2double(nz[i]))).getSorted());
		}
		CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		double[] d = mat.getData();
		int[] r = mat.getRowPointers();
		for (int i=0; i<n; i++){
			Iterator<SortPair> it = loc.get(i).iterator();
			double[] temp = new double[loc.get(i).size()];
			double[] ord = new double [loc.get(i).size()];
			int counter = 0;
			while (it.hasNext()){
				SortPair sp = it.next();
				temp[counter] = Math.exp(-sp.getValue()/sigma);
				ord[counter] = sp.getOriginalIndex();
				counter++;
			}
			int[] o = (new RankSort(ord)).getRank();
			for (int j=0; j<o.length; j++){
				d[r[i]+j] = temp[o[j]];
			}
		}
		if (nc<=2){
			int[] ind = new int[n];
			SpectralClustering.cluster(mat,ind,100,0);
			return ind;
		}
		else {
			return SpectralClustering.cluster(mat,nc,0,100,0);
		}
	}
	
	public static int[] diffRankDist(final double[][] set, int nc, int dist, double sigma){
		int n = set.length;
		int dim = set[0].length;
		int[][] nz = new int[n][];
		ArrayList<TreeSet<SortPair>> loc = new ArrayList<TreeSet<SortPair>>();
		for (int i=0; i<n; i++){
			loc.add(new TreeSet<SortPair>());
		}
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				double v = Math.abs(set[i][0]-set[j][0]);
				double w = Math.abs(set[i][0]-set[j][0]);
				for (int k=1; k<dim; k++){
					double v2 = Math.abs(set[i][k]-set[j][k]);
					if (v2>v){
						v = v2;
					}
					if (v2<w){
						w = v2;
					}
				}
				v = v-w;
				if (loc.get(i).size()<dist||v<loc.get(i).last().getValue()){
					loc.get(i).add(new SortPair(v,j));
					if (loc.get(i).size()>dist){
						loc.get(i).remove(loc.get(i).last());
					}
				}
			}
		}
		for (int i=0; i<n; i++){
			// duplicate all nearest neighbors, so the sparse distance matrix
			// becomes symmetric
			Iterator<SortPair> it = loc.get(i).iterator();
			while (it.hasNext()){
				SortPair temp = it.next();
				loc.get((int)temp.getOriginalIndex()).add(new SortPair(temp.getValue(),i));
			}
		}
		for (int i=0; i<n; i++){
			nz[i] = new int[loc.get(i).size()];
			Iterator<SortPair> it = loc.get(i).iterator();
			int counter = 0;
			while (it.hasNext()){
				nz[i][counter] = (int)it.next().getOriginalIndex();
				counter++;
			}
			nz[i] = VectorConv.double2int((new RankSort(VectorConv.int2double(nz[i]))).getSorted());
		}
		CompRowMatrix mat = new CompRowMatrix(n,n,nz);
		double[] d = mat.getData();
		int[] r = mat.getRowPointers();
		for (int i=0; i<n; i++){
			Iterator<SortPair> it = loc.get(i).iterator();
			double[] temp = new double[loc.get(i).size()];
			double[] ord = new double [loc.get(i).size()];
			int counter = 0;
			while (it.hasNext()){
				SortPair sp = it.next();
				temp[counter] = Math.exp(-sp.getValue()/sigma);
				ord[counter] = sp.getOriginalIndex();
				counter++;
			}
			int[] o = (new RankSort(ord)).getRank();
			for (int j=0; j<o.length; j++){
				d[r[i]+j] = temp[o[j]];
			}
		}
		if (nc<=2){
			int[] ind = new int[n];
			SpectralClustering.cluster(mat,ind,100,0);
			return ind;
		}
		else {
			return SpectralClustering.cluster(mat,nc,0,100,0);
		}
	}
}