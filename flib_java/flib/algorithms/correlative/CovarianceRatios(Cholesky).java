package flib.algorithms.correlative;


import java.util.ArrayList;
import java.util.Random;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.RankSort;
import flib.algorithms.correlative.CovarianceMatrix;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleSingularValueDecomposition;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleEigenvalueDecomposition;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleCholeskyDecomposition;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;

public class CovarianceRatios {
	private DenseDoubleEigenvalueDecomposition[] eig;
	private DoubleMatrix2D[] L;
	private DoubleMatrix2D[] Linv;
	private int nclass, type;
	private double tol;
	private double[] std;
	
	public CovarianceRatios(final double[][] set, final double[] labels, int np, int type, double tol){
		int n = set.length;
		int dim = set[0].length;
		this.type = type;
		this.tol = tol;
		// covariance ratios
		if (type>0){
			// setting this up for later matrix calculations
			DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
			// we expect the labels to be between 0 and nclass
			this.nclass = (int)(VectorFun.max(labels)[0]+1);
			this.eig = new DenseDoubleEigenvalueDecomposition[nclass];
			this.L = new DenseDoubleMatrix2D[nclass];
			this.Linv = new DenseDoubleMatrix2D[nclass];
			ArrayList<ArrayList<Integer>> loc = new ArrayList<ArrayList<Integer>>();
			for (int i=0; i<nclass; i++){
				loc.add(new ArrayList<Integer>());
			}
			for (int i=0; i<n; i++){
				loc.get((int)labels[i]).add(i);
			}
			double[][] set_tot = new double[np*nclass][dim];
			double[][][] sets = new double[nclass][np][dim];
			Random rng = new Random();
			// sample the points for all classes
			for (int i=0; i<nclass; i++){
				for (int j=0; j<np; j++){
					int a = rng.nextInt(loc.get(i).size());
					set_tot[i*np+j] = set[loc.get(i).get(a)].clone();
					sets[i][j] = set[loc.get(i).get(a)].clone();
				}
			}
			this.std = new double[dim];
			//z-score case
			if (type%2==0){
				std = CovarianceMatrix.std(VectorAccess.flip(set_tot));
				for (int j=0; j<dim; j++){
					if (std[j]>0){
						for (int k=0; k<set_tot.length; k++){
							set_tot[k][j]/=std[j];
						}
					}
				}
				for (int i=0; i<nclass; i++){
					for (int j=0; j<dim; j++){
						if (std[j]>0){
							for (int k=0; k<sets[i].length; k++){
								sets[i][k][j]/=std[j];
							}
						}
					}
				}
			}
			// prepare the covariance matrices
			for (int i=0; i<nclass; i++){
				double[][] temp_set = new double[np*(2*nclass-2)][dim];
				for (int j=0; j<np*nclass; j++){
					temp_set[j] = set_tot[j].clone();
				}
				for (int k=0; k<nclass-2; k++){
					for (int j=0; j<np; j++){
						temp_set[(nclass+k)*np+j] = sets[i][j].clone();
					}
				}
				DenseDoubleMatrix2D covZ = new DenseDoubleMatrix2D(CovarianceMatrix.covariance(VectorAccess.flip(temp_set)));
				this.L[i] = (new DenseDoubleCholeskyDecomposition(new DenseDoubleMatrix2D(CovarianceMatrix.covariance(VectorAccess.flip(sets[i]))))).getL();
				this.Linv[i] = alg.inverse(this.L[i]);
				/*DenseDoubleSingularValueDecomposition svd = new DenseDoubleSingularValueDecomposition(new DenseDoubleMatrix2D(CovarianceMatrix.covariance(VectorAccess.flip(sets[i]))),true,true);
				DoubleMatrix2D S = svd.getS();
				for (int j=0; j<dim; j++){
					if (S.get(j,j)>tol){
						S.set(j,j,1./S.get(j,j));
					}
					else {
						S.set(j,j,0);
					}
				}
				DoubleMatrix2D icov = alg.mult(svd.getV(),alg.mult(S,alg.transpose(svd.getU())));
				
				if (type<3){
					eig[i] = new DenseDoubleEigenvalueDecomposition(alg.mult(icov,covZ));
				}
				else {
					eig[i] = new DenseDoubleEigenvalueDecomposition(alg.mult(alg.mult(alg.mult(covZ,icov),icov),covZ));
				}*/
				eig[i] = new DenseDoubleEigenvalueDecomposition(alg.mult(Linv[i],alg.mult(covZ,alg.transpose(Linv[i]))));
			}
		}
		// else we do normal pca on the set
		else {
			this.nclass = 1;
			double[][] set_tot = VectorAccess.flip(set);
			if (type%2==0){
				this.std = CovarianceMatrix.std(set_tot);
				for (int j=0; j<dim; j++){
					if (std[j]>0){
						VectorFun.multi(set_tot[j],1./std[j]);
					}
				}
			}
			this.eig = new DenseDoubleEigenvalueDecomposition[1];
			this.eig[0] = new DenseDoubleEigenvalueDecomposition(new DenseDoubleMatrix2D(CovarianceMatrix.covariance(set_tot)));
			this.L = new DenseDoubleMatrix2D[1];
			this.L[0] = DoubleFactory2D.dense.identity(dim);
			this.Linv= new DenseDoubleMatrix2D[1];
			this.Linv[0] = DoubleFactory2D.dense.identity(dim);
		}
	}
	
	public double[] getEigenvalues(){
		int n = eig[0].getV().rows();
		double[] eigs = new double[nclass*n];
		for (int i=0; i<nclass; i++){
			double[] eigt = eig[i].getRealEigenvalues().toArray();
			int[] o = (new RankSort(eigt)).getRank();
			for (int j=0; j<n; j++){
				//eigs[i*n+j] = eigt[n-j-1];
				eigs[i*n+j] = eigt[o[n-j-1]];
			}
		}
		return eigs;
	}
	
	public double[][] getEigenvectors(int nv){
		int n = eig[0].getV().rows();
		double[][] eigv = new double[nclass*nv][n];
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		for (int i=0; i<nclass; i++){
			DoubleMatrix2D V = alg.mult(alg.transpose(Linv[i]),eig[i].getV());
			double[] eigt = eig[i].getRealEigenvalues().toArray();
			int[] o = (new RankSort(eigt)).getRank();
			for (int j=0; j<nv; j++){
				//eigv[i*nv+j] = eig[i].getV().viewColumn(n-j-1).toArray();
				eigv[i*nv+j] = (V.viewColumn(o[n-j-1])).toArray();
			}
		}
		if (type%2==0||type==-2){
			for (int i=0; i<eigv.length; i++){
				for (int j=0; j<eigv[i].length; j++){
					if (std[j]>0){
						eigv[i][j]/=std[j];
					}
				}
			}
		}
		return eigv;
	}
	
	public static double[][] project(final double[][] set, final double[][] eigv){
		DenseDoubleAlgebra alg = new DenseDoubleAlgebra();
		return VectorAccess.flip(alg.mult(new DenseDoubleMatrix2D(eigv),new DenseDoubleMatrix2D(VectorAccess.flip(set))).toArray());
	}
}	