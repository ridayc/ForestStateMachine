package flib.algorithms.clustering;

import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import flib.math.VectorFun;
import flib.math.random.Shuffle;
import flib.math.distributions.MultivariateNormal;
import flib.algorithms.clustering.Kmeans;

public class GMM {
	private int nclust;
	private int d;
	private int n;
	private double beta;
	private double tol;
	private double[][] mu;
	private double[][][] sigma;
	private double[][][] sigma_old;
	private double[] weights;
	private double[][] points;
	private double[][] tau;
	private double[] stau;
	private ArrayList<MultivariateNormal> mvn;
	private int it;
	
	public GMM(final double[][] x, int nclust, int maxit, int type, double tol, double beta, final double[][] mu, final double[][][] sigma, final double[] weights){
		this.nclust = nclust;
		this.d = x[0].length;
		this.n = x.length;
		this.beta = beta;
		this.tol = tol;
		this.points = new double[this.n][this.d];
		for (int i=0; i<this.n; i++){
			this.points[i] = x[i].clone();
		}
		this.mu = new double[this.nclust][this.d];
		this.sigma = new double[this.nclust][this.d][this.d];
		this.sigma_old = new double[this.nclust][this.d][this.d];
		this.weights = new double[this.nclust];
		this.tau = new double[this.nclust][this.n];
		this.stau = new double[this.nclust];
		this.mvn  = new ArrayList<MultivariateNormal>();
		if (type==0){
			for (int i=0; i<this.nclust; i++){
				this.mu[i] = mu[i].clone();
				for (int j=0; j<this.d; j++){
					this.sigma[i][j] = sigma[i][j].clone();
				}
			}
			this.weights = weights.clone();
		}
		else if (type==1){
			int[] r = Shuffle.randPerm(this.n);
			for (int i=0; i<this.nclust; i++){
				this.weights[i] = 1/this.nclust;
				this.mu[i] = this.points[r[i]].clone();
			}
			double[] m = new double[this.d];
			double[] v = new double[this.d];
			for (int i=0; i<this.d; i++){
				for (int j=0; j<this.n; j++){
					m[i]+=this.points[j][i];
				}
				m[i]/=this.n;
			}
			for (int i=0; i<this.d; i++){
				for (int j=0; j<this.n; j++){
					v[i]+=(this.points[j][i]-m[i])*(this.points[j][i]-m[i]);
				}
				v[i]/=this.n;
			}
			for (int i=0; i<this.nclust; i++){
				for (int j=0; j<this.d; j++){
					this.sigma[i][j][j] = v[j];
				}
				this.weights[i] = 1.0/this.nclust;
			}
		}
		else {
			Kmeans km = new Kmeans(this.points,this.nclust,maxit);
			double[][] m = km.getCenters();
			double[][] v = km.getDist();
			int[] t = km.getSize();
			for (int i=0; i<this.nclust; i++){
				this.mu[i] = m[i].clone();
				for (int j=0; j<this.d; j++){
					this.sigma[i][j][j] = v[i][j]/this.n;
				}
				this.weights[i] = (double)t[i]/this.n;
			}
		}
		for (int i=0; i<this.nclust; i++){
			this.mvn.add(new MultivariateNormal(this.mu[i],this.sigma[i]));
		}
		// here we start the GMM iteration
		this.it = this.run(maxit);		
	}
	
	public GMM(final double[][] x, int nclust, int maxit, int type, double tol, double beta){
		this(x,nclust,maxit,type,tol,beta,null,null,null);
	}
	
	public GMM(final double[][] x, int nclust, int maxit, int type, double tol){
		this(x,nclust,maxit,type,tol,1);
	}
	
	public GMM(final double[][] x, int nclust, int maxit, int type){
		this(x,nclust,maxit,type,0,1);
	}
	
	public GMM(final double[][] x, int nclust, int maxit){
		this(x,nclust,maxit,2,0,1);
	}
	
	public GMM(final double[][] x, int nclust){
		this(x,nclust,200,2,0,1);
	}
	
	public void updateMvn(){
		for (int i=0; i<this.nclust; i++){
			this.mvn.set(i,new MultivariateNormal(this.mu[i],this.sigma[i]));
		}
	}
	
	public void updateTau(){
		double[] t = new double[this.nclust];
		double s;
		for (int i=0; i<this.n; i++){
			s = 0;
			for (int j=0; j<this.nclust; j++){
				t[j] = mvn.get(j).density(this.points[i]);
				if (this.beta!=1){
					t[j] = Math.pow(t[j],this.beta);
				}
				s+=this.weights[j]*t[j];
			}
			for (int j=0; j<this.nclust; j++){
				if (s!=0){
					this.tau[j][i] = this.weights[j]*t[j]/s;
				}
				else {
					this.tau[j][i] = 1.0/this.nclust;
				}
			}
		}
	}
	
	public void updateWeights(){
		for (int i=0; i<this.nclust; i++){
			this.weights[i] = VectorFun.sum(tau[i])/this.n;
		}
	}
	
	public void updateMu(){
		for (int i=0; i<this.nclust; i++){
			this.stau[i] = VectorFun.sum(this.tau[i]);
			for (int j=0; j<this.d; j++){
				this.mu[i][j] = 0;
				for (int k=0; k<this.n; k++){
					this.mu[i][j]+=tau[i][k]*this.points[k][j];
				}
				this.mu[i][j]/=this.stau[i];
			}
		}
	}
	
	public void updateSigma(){
		for (int i=0; i<this.nclust; i++){
			for (int j=0; j<this.d; j++){
				for (int k=0; k<this.d; k++){
					this.sigma[i][j][k] = 0;
					for (int l=0; l<this.n; l++){
						this.sigma[i][j][k]+=tau[i][l]*(this.points[l][j]-this.mu[i][j])*(this.points[l][k]-this.mu[i][k]);
					}
					this.sigma[i][j][k]/=this.stau[i];
				}
			}
		}
	}
	
	public int run(int maxit){
		int count;
		for (count=0; count<maxit; count++){
			for (int j=0; j<this.nclust; j++){
				for (int k=0; k<this.d; k++){
					this.sigma_old[j][k] = this.sigma[j][k].clone();
				}
			}
			this.updateMvn();
			this.updateTau();
			this.updateWeights();
			this.updateMu();
			this.updateSigma();
			double s = 0;
			for (int j=0; j<this.nclust; j++){
				for (int k=0; k<this.d; k++){
					for (int l=0; l<this.d; l++){
						s+=Math.abs(this.sigma_old[j][k][l]-this.sigma[j][k][l]);
					}
				}
			}
			if (s<=this.tol){
				break;
			}
		}
		return count;
	}
	
	public double[] assignWeights(final double[] point){
		double[] t = new double[this.nclust];
		for (int i=0; i<this.nclust; i++){
			t[i] = this.mvn.get(i).density(point);
		}
		return t;
	}
	
	public double[][] assignWeights(final double[][] points){
		int l = points.length;
		double[][] t = new double[l][this.nclust];
		for (int i=0; i<l; i++){
			t[i] = assignWeights(points[i]);
		}
		return t;
	}
	
	public double[] assignFractions(final double[] point){
		double[] t = assignWeights(point);
		double s = VectorFun.sum(t);
		return VectorFun.div(t,s);
	}
	
	public double[][] assignFractions(final double[][] points){
		int l = points.length;
		double[][] t = new double[l][this.nclust];
		for (int i=0; i<l; i++){
			t[i] = assignFractions(points[i]);
		}
		return t;
	}
	
	public int assignCluster(final double[] point){
		return VectorFun.maxind(assignWeights(point));
	}
	
	public int[] assignCluster(final double[][] points){
		int l = points.length;
		int[] t = new int[l];
		for (int i=0; i<l; i++){
			t[i] = assignCluster(points[i]);
		}
		return t;
	}
	
	public double getLikelihood(){
		double lk = 0;
		double a;
		for (int i=0; i<this.n; i++){
			a = 0;
			for (int j=0; j<this.nclust; j++){
				a+=this.weights[j]*this.mvn.get(j).density(this.points[i]);
			}
			lk+=Math.log(a);
		}
		return lk;
	}
	
	// getter functions
	public double[] getWeights(){
		return this.weights.clone();
	}
	
	public double[][] getMu(){
		double[][] cent = new double[this.nclust][this.d];
		for (int i=0; i<this.nclust; i++){
			cent[i] = this.mu[i].clone();
		}
		return cent;
	}
	
	public double[][][] getSigma(){
		double[][][] sig = new double[this.nclust][this.d][this.d];
		for (int i=0; i<this.nclust; i++){
			for (int j=0; j<this.d; j++){
				sig[i][j] = this.sigma[i][j].clone();
			}
		}
		return sig;
	}
	
	public int getIter(){
		return this.it;
	}
}