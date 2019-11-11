package flib.algorithms.sampling;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.splitfunctions.GiniClusterSplit;
import flib.algorithms.randomforest.splitfunctions.CrossingRegression;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.UniqueSorted;
import flib.math.RankSort;
import flib.math.random.Shuffle;

public class RFSampler implements
java.io.Serializable {
	// collection of Random Forests. One for each dimension
	private ArrayList<RandomForest> RF;
	// if the dimensions are categorical or not
	// this makes the difference if a classification or a regression forest
	// is used for classification
	private boolean[] categorical;
	// the complete input trainingset
	private double[][] trainingset;
	// each dimension reordered so that regression predictions are based
	// only on the rank of the input values of a given dimension
	private double[][] rankset;
	private double[][] rankorder;
	// size of the training data set
	int n, dim;
	
	// constructor
	public RFSampler(final double[][] trainingset, final double[] weights, final boolean[] categorical, double[] parameters, int ntree){
		// store all necessary private variables
		this.n = trainingset.length;
		this.dim = trainingset[0].length;
		this.trainingset = new double[n][dim];
		this.rankset = new double[dim][n];
		this.rankorder = new double[dim][n];
		RF = new ArrayList<RandomForest>();
		for (int i=0; i<n; i++){
			this.trainingset[i] = trainingset[i].clone();
		}
		double[][] tstranspose = VectorAccess.flip(trainingset);
		// rank ordering
		for (int i=0; i<dim; i++){
			RankSort r = new RankSort(tstranspose[i]);
			rankset[i] = r.getSorted();
			rankorder[i] = (new RankSort(r.getDRank())).getDRank();
		}
		// create a prediction forest for each dimension
		for (int i=0; i<dim; i++){
			// create a temporary training set with the current dimension 
			// not being present for prediction
			double[][] ts = new double[n][dim-1];
			boolean[] cat = new boolean[dim-1];
			for (int j=0; j<dim; j++){
				if (j<i){
					for (int k=0; k<n; k++){
						ts[k][j] = trainingset[k][j];
					}
					cat[j] = categorical[j];
				}
				else if(j>i){
					for (int k=0; k<n; k++){
						ts[k][j-1] = trainingset[k][j];
					}
					cat[j-1] = categorical[j];
				}
			}
			if (categorical[i]){
				parameters[0] = VectorFun.max(UniqueSorted.unique(tstranspose[i]))[0]+1;
				RF.add(new RandomForest(ts,tstranspose[i],weights,cat,VectorFun.add(new double[dim-1],1),parameters,VectorFun.add(new double[(int)parameters[0]],1),new GiniClusterSplit(),ntree));
			}
			else {
				parameters[0] = 1;
				RF.add(new RandomForest(ts,rankorder[i],weights,cat,VectorFun.add(new double[dim-1],1),parameters,new double[1],new CrossingRegression(),ntree));
			}
		}
	}
	
	public double[] sample(final double[] point, final int[] repl, int it, int type, double p, double s){
		int t = type%2+2;
		double[] samp = point.clone();
		for (int i=0; i<it; i++){
			int[] rp = Shuffle.randPerm(repl.length);
			for (int j=0; j<rp.length; j++){
				double[] samp2 = new double[dim-1];
				for (int k=0; k<dim; k++){
					if (k<repl[rp[j]]){
						samp2[k] = samp[k];
					}
					else if (k>repl[rp[j]]){
						samp2[k-1] = samp[k];
					}
				}
				samp[repl[rp[j]]] = (samp[repl[rp[j]]]+s*trainingset[(int)RF.get(repl[rp[j]]).sampledNeighbor(samp2,new int[0],t,p)[0]][repl[rp[j]]])/(s+1);
			}
		}
		return samp;
	}
	
	public double[] sample(final double[] point, int it, int type, double p, double s){
		int[] repl = new int[dim];
		for (int i=0; i<dim; i++){
			repl[i] = i;
		}
		return sample(point,repl,it,type,p,s);
	}
	
	public double[] sample(int it, int type, double p, double s){
		return sample(trainingset[(new Random()).nextInt(n)].clone(),it,type,p,s);
	}
	
	public double[][] sample2(final double[] point, final int[] repl, int it, int type, double p, double s){
		int t = type%2+2;
		double[][] samp = new double[it][dim];
		samp[0] = point.clone();
		for (int i=1; i<it; i++){
			int[] rp = Shuffle.randPerm(repl.length);
			samp[i] = samp[i-1].clone();
			for (int j=0; j<rp.length; j++){
				double[] samp2 = new double[dim-1];
				for (int k=0; k<dim; k++){
					if (k<repl[rp[j]]){
						samp2[k] = samp[i][k];
					}
					else if (k>repl[rp[j]]){
						samp2[k-1] = samp[i][k];
					}
				}
				samp[i][repl[rp[j]]] = (samp[i][repl[rp[j]]]+s*trainingset[(int)RF.get(repl[rp[j]]).sampledNeighbor(samp2,new int[0],t,p)[0]][repl[rp[j]]])/(s+1);
			}
		}
		return samp;
	}
	
	public double[][] sample2(final double[] point, int it, int type, double p, double s){
		int[] repl = new int[dim];
		for (int i=0; i<dim; i++){
			repl[i] = i;
		}
		return sample2(point,repl,it,type,p,s);
	}
	
	public double[][] sample2(int it, int type, double p, double s){
		return sample2(trainingset[(new Random()).nextInt(n)].clone(),it,type,p,s);
	}
	
	public double[] bayesianSample(final double[] point, final int[] missing, int type, double p){
		double[] p2 = new double[dim];
		Random rng = new Random();
		int t = type%2+2;
		if (missing.length>=dim){
			p2 = trainingset[rng.nextInt(n)].clone();
		}
		else {
			p2 = point.clone();
		}
		int[] rp = Shuffle.randPerm(missing.length);
		for (int i=0; i<missing.length; i++){
			int c = missing[rp[i]];
			int[] m2 = new int[missing.length-i-1];
			for (int j=0; j<m2.length; j++){
				if (missing[rp[i+j+1]]<c){
					m2[j] = missing[rp[i+j+1]];
				}
				else {
					m2[j] = missing[rp[i+j+1]]-1;
				}
			}
			Arrays.sort(m2);
			p2[rp[i]] = trainingset[(int)RF.get(rp[i]).sampledNeighbor(p2,m2,t,p)[0]][rp[i]];
		}
		return p2;
	}
	
	public double[] bayesianSample(int type, double p){
		int[] missing = new int[dim];
		for (int i=0; i<dim; i++){
			missing[i] = i;
		}
		return bayesianSample(new double[0],missing,type,p);
	}
	
	public ArrayList<RandomForest> getForest(){
		return this.RF;
	}
}