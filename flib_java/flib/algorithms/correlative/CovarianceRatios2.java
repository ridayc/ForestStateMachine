package flib.algorithms.correlative;


import java.util.Arrays;
import java.util.ArrayList;
import java.util.Random;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.RankSort;
import flib.algorithms.correlative.CovarianceMatrix;
//import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleSingularValueDecomposition;
//import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleEigenvalueDecomposition;
//import cern.colt.matrix.tdouble.DoubleMatrix2D;
//import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
//import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import cern.colt.matrix.linalg.SingularValueDecomposition;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;


public class CovarianceRatios{
	private double[][][] eigv;
	private double[][] eigs;
	private int nclass, type;
	private double tol;
	private double[] std;
	
	public CovarianceRatios(final double[][] set, final double[] labels, int np, int type, double tol){
		int n = set.length;
		int dim = set[0].length;
		this.type = type;
		this.tol = tol;
		this.std = VectorFun.add(new double[dim],1);
		// covariance ratios
		if (type>0){
			// setting this up for later matrix calculations
			Algebra alg = new Algebra();
			// we expect the labels to be between 0 and nclass-1
			this.nclass = (int)(VectorFun.max(labels)[0]+1);
			this.eigv = new double[nclass][][];
			this.eigs = new double[nclass][];
			ArrayList<ArrayList<Integer>> loc = new ArrayList<ArrayList<Integer>>();
			for (int i=0; i<nclass; i++){
				loc.add(new ArrayList<Integer>());
			}
			for (int i=0; i<n; i++){
				loc.get((int)labels[i]).add(i);
			}
			Random rng = new Random();
			double[][] m = new double[nclass][dim];
			double[][][] m2 = new double[nclass][dim][dim];
			double[] mt = new double[dim];
			double[][] cvt = new double[dim][dim];
			double[][] val = new double[dim][np*nclass];
			for (int i=0; i<nclass; i++){
				for (int j=0; j<np; j++){
					int a = rng.nextInt(loc.get(i).size());
					double[] v = set[loc.get(i).get(a)];
					for (int k=0; k<dim; k++){
						m[i][k]+=v[k];
						mt[k]+=v[k];
						val[k][i*np+j] = v[k];
						for (int l=0; l<dim; l++){
							m2[i][k][l]+=v[k]*v[l];
							cvt[k][l]+=v[k]*v[l];
						}
					}
				}
			}				
			//z-score case
			if (type%2==0){
				for (int i=0; i<dim; i++){
					std[i] = Math.sqrt((cvt[i][i]-(mt[i]*mt[i])/(np*nclass))/(np*nclass));
					if (std[i]>0){
						std[i] = 1./std[i];
					}
				}
				for (int i=0; i<nclass; i++){
					for (int k=0; k<dim; k++){
						m[i][k]*=std[k];
						for (int l=0; l<dim; l++){
							m2[i][k][l]*=std[k]*std[l];
						}
					}
				}
				for (int k=0; k<dim; k++){
					mt[k]*=std[k];
					for (int l=0; l<dim; l++){
						cvt[k][l]*=std[k]*std[l];
					}
				}
			}
			if (type<=2&&type>=-2){
				for (int i=0; i<nclass; i++){
					// prepare the covariance matrices
					double[][] cv = new double[dim][dim];
					double[][] cv_i = new double[dim][dim];
					for (int k=0; k<dim; k++){
						double a = (m[i][k]+(mt[k]-m[i][k])/(nclass-1))/np;
						for (int l=0; l<dim; l++){
							cv[k][l] = (m2[i][k][l]+(cvt[k][l]-m2[i][k][l])/(nclass-1))/np;
							double b = (m[i][l]+(mt[l]-m[i][l])/(nclass-1))/np;
							cv[k][l]-=a*b;
							cv_i[k][l] = (m2[i][k][l]-(m[i][k]*m[i][l])/np)/np;
						}
					}
					DoubleMatrix2D covZ = new DenseDoubleMatrix2D(cv);
					SingularValueDecomposition svd = new SingularValueDecomposition(new DenseDoubleMatrix2D(cv_i));
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
					EigenvalueDecomposition eig = new EigenvalueDecomposition(alg.mult(icov,covZ));
					this.eigv[i] = extractEigenvectors(eig);
					this.eigs[i] = extractEigenvalues(eig);
				}
			}
			else {
				//System.out.println("made it this far");
				for (int i=0; i<nclass; i++){
					// prepare the covariance matrices
					//double[][] cv = new double[dim][dim];
					double[][] cv_i = new double[dim][dim];
					for (int k=0; k<dim; k++){
						for (int l=0; l<dim; l++){
							cv_i[k][l] = m2[i][k][l]/np/np+(cvt[k][l]-m2[i][k][l])/(nclass-1)/np/(nclass-1)/np-2*m[i][k]/np*(mt[l]-m[i][l])/(nclass-1)/np;
						}
					}
					DenseDoubleMatrix2D covZ = new DenseDoubleMatrix2D(cv_i);
					//System.out.println("created matrix");
					EigenvalueDecomposition eig = new EigenvalueDecomposition(covZ);
					this.eigv[i] = extractEigenvectors(eig);
					this.eigs[i] = extractEigenvalues(eig);
					//System.out.println("got eigenvalues");
				}
			}
		}
		// else we do normal pca on the set
		else {
			this.nclass = 1;
			double[][] set_tot = VectorAccess.flip(set);
			if (type%2==0){
				this.std = VectorFun.sqrt(CovarianceMatrix.std(set_tot));
				for (int j=0; j<dim; j++){
					if (std[j]>0){
						std[j] = 1./std[j];
					}
					VectorFun.multi(set_tot[j],std[j]);
				}
			}
			eigv = new double[1][][];
			eigs = new double[1][];
			EigenvalueDecomposition eig = new EigenvalueDecomposition(new DenseDoubleMatrix2D(CovarianceMatrix.covariance(set_tot)));
			this.eigv[0] = extractEigenvectors(eig);
			this.eigs[0] = extractEigenvalues(eig);
		}
	}
	
	public double[] getEigenvalues(int nv){
		int n = eigs[0].length;
		double[] eigval = new double[nclass*n];
		for (int i=0; i<nclass; i++){
			for (int j=0; j<nv; j++){
				eigval[i*nv+j] = eigs[i][j];
			}
		}
		return eigval;
	}
	
	private double[][] extractEigenvectors(EigenvalueDecomposition eig){
		int n = eig.getV().rows();
		double[][] eigvec = new double[n][n];
		double[] eigt = eig.getRealEigenvalues().toArray();
		int[] o = (new RankSort(eigt)).getRank();
		for (int i=0; i<n; i++){
			eigvec[i] = eig.getV().viewColumn(o[n-i-1]).toArray();
		}
		return eigvec;
	}
	
	private double[] extractEigenvalues(EigenvalueDecomposition eig){
		double[] eigt = eig.getRealEigenvalues().toArray();
		return (new RankSort(eigt)).getSorted();
	}
	
	public double[][] getEigenvectors(int nv){
		int n = eigv[0].length;
		double[][] eigvec = new double[nclass*nv][n];
		for (int i=0; i<nclass; i++){
			for (int j=0; j<nv; j++){
				eigvec[i*nv+j] = eigv[i][j].clone();
			}
		}
		if (type%2==0||type==-2){
			for (int i=0; i<eigv.length; i++){
				for (int j=0; j<eigvec[i].length; j++){
					eigvec[i][j]*=std[j];
				}
				VectorFun.multi(eigvec[i],1./Math.sqrt(VectorFun.sum(VectorFun.mult(eigvec[i],eigvec[i]))));
			}
		}
		return eigvec;
	}
	
	public static double[][] project(final double[][] set, final double[][] eigv){
		Algebra alg = new Algebra();
		return VectorAccess.flip(alg.mult(new DenseDoubleMatrix2D(eigv),new DenseDoubleMatrix2D(VectorAccess.flip(set))).toArray());
	}
}	