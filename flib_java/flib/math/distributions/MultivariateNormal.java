package flib.math.distributions;

import java.util.Random;
import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleFactory1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import java.lang.Math;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.algo.SmpDoubleBlas;
import cern.jet.math.tdouble.DoubleFunctions;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleCholeskyDecomposition;
import flib.math.VectorFun;

public class MultivariateNormal {
	private DoubleMatrix2D mu;
	private DoubleMatrix2D sigma;
	private DoubleMatrix2D sigma_inv;
	// prefactor constant of the distribution
	private double p;
	private int d;
	private DenseDoubleAlgebra Alg = new DenseDoubleAlgebra();
	private boolean init = false;
	private DoubleMatrix2D L;
	private Random random;
	
	public MultivariateNormal(final double[] mu, final double[][] sigma, int type){
		this.d = mu.length;
		this.mu = DoubleFactory2D.dense.make(mu,this.d);
		if (type==0){
			this.sigma = DoubleFactory2D.dense.make(sigma);
			this.invertSigma();
		}
		else {
			this.sigma_inv = DoubleFactory2D.dense.make(sigma);
			this.invertSigma_inv();
		}
		this.prefactor();
	}
	
	public MultivariateNormal(final double[] mu, final double[][] sigma){
		this(mu,sigma,0);
	}
	
	public void invertSigma(){
		this.sigma_inv = Alg.inverse(this.sigma);
	}
	
	public void invertSigma_inv(){
		this.sigma = Alg.inverse(this.sigma_inv);
	}
	
	public void prefactor(){
		this.p = 1/Math.sqrt(Math.pow((2*Math.PI),this.d)*Alg.det(this.sigma));
	}
	
	public double density(final double[] x){
		DoubleMatrix2D xm = DoubleFactory2D.dense.make(x,this.d);
		(new SmpDoubleBlas()).assign(xm,this.mu,DoubleFunctions.minus);
		double[][] a = (Alg.mult(Alg.mult(Alg.transpose(xm),this.sigma_inv),xm)).toArray();
		return this.p*Math.exp(-0.5*a[0][0]);
	}
	
	public double[] density(final double[][] x){
		int l = x.length;
		double[] den = new double[l];
		for (int i=0; i<l; i++){
			den[i] = this.density(x[i]);
		}
		return den;
	}
	
	public double[] generate(){
		if (!this.init){
			this.random = new Random();
			this.L = (new DenseDoubleCholeskyDecomposition(this.sigma)).getL();
		}
		double[] z = new double[this.d];
		for (int i=0; i<this.d; i++){
			z[i] = random.nextGaussian();
		}
		DoubleMatrix2D r = DoubleFactory2D.dense.make(z,this.d);
		r = Alg.mult(this.L,r);
		double[][] a = r.toArray();
		double[][] b = this.mu.toArray();
		for (int i=0; i<this.d; i++){
			z[i] = a[i][0]+b[i][0];
		}
		return z;
	}
	
	public double[][] generate(int num){
		double[][] z = new double[num][this.d];
		for (int i=0; i<num; i++){
			z[i] = this.generate();
		}
		return z;
	}
	
	public double[][] getSigma(){
		return this.sigma.toArray();
	}
	
	public double[] getMu(){
		double[] m= new double[this.d];
		double[][] a = this.mu.toArray();
		for (int i=0; i<this.d; i++){
			m[i] = a[i][0];
		}
		return m;
	}
}