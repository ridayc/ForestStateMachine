package flib.algorithms.extrapolation;

import java.lang.Math;
import java.util.ArrayList;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import flib.math.random.Shuffle;
import flib.math.VectorAccess;
import flib.math.VectorFun;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.splitfunctions.CrossingRegression;

public class FunctionExtrapolator{
	// variables:
	// diffn: highest differential degree. At least 0
	// polyn: highest polynomial degree
	// dp: sum of diffn and polyn
	// seql: n-gram sequence length
	// sml: smoothing length for differential estimation
	// set: points used for training
	// nc: number of approximation classes (not used yet, since we can explicity 
	// np: number of points for forest training
	// n: length of the trainingset minus the sequence length plus one
	// regression forest variables
	// pl: predictor length
	// mtry, maxdepth, maxleafsize, splitpurity = 1, splittype, splittype = 2, ntree
	
	
	private int diffn, polyn, dp, seql,np,n,pl, ntree;
	private double sml;
	private int[] set;
	private double[] x, fx, diffv, polyv, parameters, splitpurity;
	private double[][] ffx, intv, c, pred;
	private ArrayList<RandomForest> lpredictors, rpredictors;
	
	
	public FunctionExtrapolator(final double[] x, final double[] fx, int diffn, int polyn, int seql, double sml, int np, int pl, int mtry, int maxdepth, int maxleafsize, int dimtype, double crossing, int ntree){
		// copy function internal variables
		this.x = x.clone();
		this.fx = fx.clone();
		if (diffn>=0){
			this.diffn = diffn;
		}
		else {
			this.diffn = 0;
		}
		if (polyn>=-1){
			this.polyn = polyn+1;
		}
		else {
			this.polyn = 0;
		}
		this.dp = this.diffn+this.polyn;
		if (seql%2==1){
			this.seql = seql;
		}
		else {
			this.seql = seql+1;
		}
		this.sml = 1/(2*sml*sml);
		this.np = np;
		this.pl = pl;
		this.n = x.length-this.seql+1;
		this.parameters = new double[7];
		this.parameters[0] = 1;
		this.parameters[1] = mtry;
		this.parameters[2]  = maxdepth;
		this.parameters[3] = maxleafsize;
		this.parameters[4] = dimtype;
		this.parameters[5] = crossing;
		this.splitpurity = new double[1];
		this.splitpurity[0] = 1e-8;
		this.ntree = ntree;
		this.generateDifferences();
		this.generateInternal();
		this.generatePredictors();
	}
	
	public void generateDifferences(){
		this.diffv = new double[diffn];
		this.polyv = new double[polyn];
		this.ffx = new double[diffn+1][this.x.length];
		this.ffx[0] = this.fx.clone();
		double f,wt,w,h;
		int a;
		// successively calculate all derivatives
		for (int i=0; i<this.diffn; i++){
			for (int j=0; j<this.x.length; j++){
				f = 0;
				wt = 0;
				for (int k=0; k<this.seql; k++){
					a = j-k;
					if (a>=0&&a<this.x.length){
						h = x[j]-x[a];
						if (h!=0){
							// weight of the current finite difference
							w = Math.exp(-h*h*this.sml);
							wt+=w;
							// calculation of the finite differences
							f+=(ffx[i][j]-ffx[i][a])/h*w;
						}
					}
				}
				ffx[i+1][j] = f/wt;
			}
		}
	}
	
	public void generateInternal(){
		this.intv = new double[this.x.length][this.dp];
		for (int i=0; i<this.x.length; i++){
			for (int j=0; j<this.diffn; j++){
				this.intv[i][j] = this.ffx[j+1][i];
			}
		}
		for (int i=0; i<this.x.length; i++){
			for (int j=0; j<this.polyn; j++){
				this.intv[i][j+this.diffn] = Math.pow(this.x[i],j);
			}
		}
		DenseDoubleMatrix2D A, B;
		Algebra Alg = new Algebra();
		this.pred = new double[this.n][this.dp];
		double[][] Ad = new double[this.seql][this.dp];
		double[][] Bd = new double[this.seql][1];
		double[][] temp;
		for (int i=this.seql/2+1; i<this.x.length-this.seql/2; i++){
			for (int j=-this.seql/2; j<this.seql/2; j++){
				Ad[j+this.seql/2] = this.intv[i+j].clone();
				Bd[j+this.seql/2][0] = this.ffx[0][i+j+1];
			}
			A = new DenseDoubleMatrix2D(Ad);
			B = new DenseDoubleMatrix2D(Bd);
			// least squares approximation for the internal variables
			temp = (Alg.solve(A,B)).toArray();
			for (int j=0; j<this.dp; j++){
				this.pred[i-this.seql/2][j] = temp[j][0];
			}
		}
	}
	
	public void generatePredictors(){
		// this is one of the more time consuming and challenging parts of the 
		// algorithm
		this.lpredictors = new ArrayList<RandomForest>();
		this.rpredictors = new ArrayList<RandomForest>();
		double[][] trainingset = new double[this.np][this.pl*this.dp];
		double[] labels = new double[this.np];
		int l = this.n-this.pl;
		int[] ind;
		double[] weights = new double[this.np];
		boolean[] categorical = new boolean[this.pl*this.dp];
		double[] dimweights = VectorFun.add(new double[this.pl*this.dp],1);
		for (int i=0; i<this.dp; i++){
			ind = Shuffle.randPerm(l);
			ind = VectorAccess.access(ind,0,this.np);
			for (int j=0; j<this.np; j++){
				for (int k=0; k<this.pl; k++){
					VectorAccess.write(trainingset[j],this.pred[ind[j]+k],k*dp);
				}
				labels[j] = this.pred[ind[j]+this.pl][i];
			}
			CrossingRegression CR = new CrossingRegression();
			this.rpredictors.add(new RandomForest(trainingset,labels,weights,categorical,dimweights,parameters,splitpurity,CR,this.ntree));
			// left side prediction
			ind = Shuffle.randPerm(l);
			ind = VectorAccess.access(ind,0,this.np);
			for (int j=0; j<this.np; j++){
				for (int k=0; k<this.pl; k++){
					VectorAccess.write(trainingset[j],this.pred[ind[j]+(k+1)],k*dp);
				}
				labels[j] = this.pred[ind[j]][i];
			}
			this.lpredictors.add(new RandomForest(trainingset,labels,weights,categorical,dimweights,parameters,splitpurity,CR,this.ntree));
		}
	}
	
	public double[][] predict(final double[][] pred, int it, boolean direction){
		double[][] seq = new double[it+this.pl][this.dp];
		double[][] temp = new double[it][this.dp];
		double[] a = new double[this.dp*this.pl];
		if (direction){
			for (int i=0; i<this.pl; i++){
				seq[i] = pred[i].clone();
			}
			for (int i=0; i<it; i++){
				for (int j=0; j<this.pl; j++){
					VectorAccess.write(a,seq[i+j],j*this.dp);
				}
				for (int j=0; j<this.dp; j++){
					seq[i+this.pl][j] = VectorFun.sum(this.rpredictors.get(j).applyForest(a))/this.ntree;
				}
			}
			for (int i=0; i<it; i++){
				temp[i] = seq[i+this.pl].clone();
			}
		}
		else{
			for (int i=0; i<this.pl; i++){
				seq[it+i] = pred[i].clone();
			}
			for (int i=0; i<it; i++){
				for (int j=0; j<this.pl; j++){
					VectorAccess.write(a,seq[it-i+j],j*this.dp);
				}
				for (int j=0; j<this.dp; j++){
					seq[it-1-i][j] = VectorFun.sum(this.lpredictors.get(j).applyForest(a))/this.ntree;
				}
			}
			for (int i=0; i<it; i++){
				temp[i] = seq[i].clone();
			}
		}
		return temp;
	}
	
	public double[][] functionEval(final double[][] pred, final double[][] f0, final double x0[], double step, boolean direction){
		int l = pred.length;
		double[][] f = new double[this.seql+l][this.diffn+1];
		double[] x = new double[this.seql+l];
		for (int i=0; i<this.seql; i++){
			for (int j=0; j<this.diffn+1; j++){
				f[i][j] = f0[i][j];
			}
			x[i] = x0[i];
		}
		double w,wt,ft,h;
		for (int i=0; i<l; i++){
			x[this.seql+i] = x[this.seql-1+i]+step;
			f[this.seql+i][0] = 0;
			for (int j=0; j<this.diffn; j++){
				f[this.seql+i][0]+=pred[i][j]*f[this.seql+i-1][j+1];
			}
			for (int j=0; j<this.polyn; j++){
				f[this.seql+i][0]+=pred[i][this.diffn+j]*Math.pow(x[this.seql+i-1],j);
			}
			for (int j=0; j<this.diffn; j++){
				wt = 0;
				ft = 0;
				for (int k=0; k<this.seql; k++){
					h = x[this.seql+i]-x[this.seql+i-1-k];
					if (h!=0){
						w = Math.exp(-h*h*this.sml);
						wt+=w;
						ft+=(f[this.seql+i][j]-f[this.seql+i-1-k][j])/h*w;
					}
				}
				f[this.seql+i][j+1] = ft/wt;
			}
		}
		return f;
	}

	public ArrayList<RandomForest> getLPredictors(){
		return this.lpredictors;
	}
	
	public ArrayList<RandomForest> getRPredictors(){
		return this.rpredictors;
	}
	
	public double[][] getPredictors(){
		double[][] temp = new double[this.n][this.dp];
		for (int i=0; i<this.n; i++){
			temp[i] = this.pred[i].clone();
		}
		return temp;
	}
	
	public double[][] getDerivatives(){
		double[][] temp = new double[this.diffn+1][this.x.length];
		for (int i=0; i<this.diffn+1; i++){
			temp[i] = this.ffx[i].clone();
		}
		return temp;
	}
}	