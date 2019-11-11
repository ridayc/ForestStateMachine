package flib.algorithms.clustering;

import java.util.ArrayList;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;
import java.lang.Math;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.SeqBlas;
import cern.jet.math.Functions;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import flib.math.RankSort;
import flib.math.VectorFun;

public class SpectralClustering {
	private Algebra Alg = new Algebra();
	private int[] currentsplit1;
	private int[] currentsplit2;
	private DoubleMatrix2D currentmat;
	private ArrayList<int[]> clusters = new ArrayList<int[]>();
	private ArrayList<DoubleMatrix2D> clustermat = new ArrayList<DoubleMatrix2D>();
	private ArrayList<Double> assoc = new ArrayList<Double>();
	private int[] indices;
	private DoubleMatrix2D mat;
	
	public void cluster(final double[][] S, int nc){
		this.clusters = new ArrayList<int[]>();
		this.clustermat = new ArrayList<DoubleMatrix2D>();
		this.assoc = new ArrayList<Double>();
		this.currentmat = DoubleFactory2D.dense.make(S);
		this.mat = DoubleFactory2D.dense.make(S);
		this.spec(this.currentmat); 
		int[] temp1, temp2;
		this.clusters.add(this.currentsplit1.clone());
		this.clusters.add(this.currentsplit2.clone());
		this.clustermat.add(DoubleFactory2D.dense.make(this.currentmat.viewSelection(this.currentsplit1,this.currentsplit1).toArray()));
		this.clustermat.add(DoubleFactory2D.dense.make(this.currentmat.viewSelection(this.currentsplit2,this.currentsplit2).toArray()));
		this.assoc.add(this.assocval(this.clustermat.get(0),this.currentmat,0));
		double max;
		int ind;
		for (int i=2; i<nc; i++){
			this.assoc.add(this.assocval(this.clustermat.get(i-1),this.currentmat,i-1));
			max = 0;
			ind = 0;
			for (int j=0; j<i; j++){
				if (max<this.assoc.get(j)){
					max = this.assoc.get(j);
					ind = j;
				}
			}
			this.spec(this.clustermat.get(ind));
			temp1 = new int[this.currentsplit1.length];
			for (int j=0; j<this.currentsplit1.length; j++){
				temp1[j] = this.clusters.get(ind)[this.currentsplit1[j]];
			}
			temp2 = new int[this.currentsplit2.length];
			for (int j=0; j<this.currentsplit2.length; j++){
				temp2[j] = this.clusters.get(ind)[this.currentsplit2[j]];
			}
			this.clusters.add(temp2.clone());
			this.clusters.set(ind, temp1.clone());
			this.clustermat.add(DoubleFactory2D.dense.make(clustermat.get(ind).viewSelection(this.currentsplit2,this.currentsplit2).toArray()));
			this.clustermat.set(ind, DoubleFactory2D.dense.make(clustermat.get(ind).viewSelection(this.currentsplit1,this.currentsplit1).toArray()));
			this.assoc.set(ind, this.assocval(this.clustermat.get(ind),this.currentmat,ind));
		}
		indices = new int[this.mat.rows()];
		for (int i=0; i<clusters.size(); i++){
			for (int j=0; j<clusters.get(i).length; j++){
				indices[clusters.get(i)[j]] = i;
			}
		}
	}
	
	private void spec(DoubleMatrix2D A){
		int n = A.rows();
		DoubleMatrix1D B = DoubleFactory1D.dense.make(n);
		for (int i=0; i<n; i++){
			B.set(i,1/Math.sqrt(A.viewRow(i).zSum()));
		}
		DoubleMatrix2D D = DoubleFactory2D.dense.diagonal(B);
		DoubleMatrix2D L = DoubleFactory2D.dense.identity(n);
		SeqBlas.seqBlas.assign(L,this.Alg.mult(this.Alg.mult(D,A),D),Functions.minus);
		EigenvalueDecomposition eig = new EigenvalueDecomposition(L);
		// need to still check if the eigenvalues are sorted in descending order
		int[] eigord = (new RankSort(VectorFun.abs(eig.getRealEigenvalues().toArray()))).getRank();
		double[] v = eig.getV().viewColumn(eigord[1]).toArray();
		//double[] v = eig.getV().viewColumn(1).toArray();
		int[] rv = (new RankSort(v)).getRank();
		double[] min = new double[2];
		min[0] = Double.MAX_VALUE;
		double w = 0;
		double a = 0;
		double b = A.zSum();
		int[] i0 = new int[1];
		double ncut;
		for (int i=0; i<n-1; i++){
			i0[0] = rv[i];
			int[] i1 = new int[n-i-1];
			for (int j=0; j<n-i-1; j++){
				i1[j] = rv[i+1+j];
			}
			int[] i2 = new int[i];
			for (int j=0; j<i; j++){
				i2[j] = rv[j];
			}
			w+=A.viewSelection(i0,i1).zSum()-A.viewSelection(i0,i2).zSum();
			a+=A.viewSelection(i0,null).zSum();
			ncut = w*(1/a+1/(b-a));
			if (ncut<min[0]){
				min[0] = ncut;
				min[1] = i;
			}
		}
		this.currentsplit1 = new int[(int)min[1]+1];
		for (int i=0; i<min[1]+1; i++){
			this.currentsplit1[i] = rv[i];
		}
		this.currentsplit2 = new int[n-1-(int)min[1]];
		for (int i=0; i<n-1-(int)min[1]; i++){
			this.currentsplit2[i] = rv[i+(int)min[1]+1];
		}
	}
	
	private double assocval(DoubleMatrix2D A, DoubleMatrix2D T, int count){
		if (A.rows()<=1) {
			return 0;
		}
		this.spec(A);
		int[] temp1,temp2;
		temp1 = new int[this.currentsplit1.length];
		for (int i=0; i<this.currentsplit1.length; i++){
			temp1[i] = this.clusters.get(count)[this.currentsplit1[i]];
		}
		temp2 = new int[this.currentsplit2.length];
		for (int i=0; i<this.currentsplit2.length; i++){
			temp2[i] = this.clusters.get(count)[this.currentsplit2[i]];
		}
		double val = A.viewSelection(this.currentsplit1,this.currentsplit1).zSum()/T.viewSelection(temp1,null).zSum()+A.viewSelection(this.currentsplit2,this.currentsplit2).zSum()/T.viewSelection(temp2,null).zSum();
		return val;
	}
	
	public int[] getClusterNumbers(){
		return this.indices.clone();
	}
	
	public double totalAssoc(){
		double val = 0;
		int[] temp;
		for (int i=0; i<clusters.size(); i++){
			val+=this.mat.viewSelection(this.clusters.get(i),this.clusters.get(i)).zSum()/this.mat.viewSelection(this.clusters.get(i),null).zSum();
		}
		return val;
	}
}