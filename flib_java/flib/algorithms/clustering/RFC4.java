package flib.algorithms.clustering;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.RankSort;
import flib.math.random.Shuffle;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.ForestFunctions;
import flib.algorithms.randomforest.RewriteForest;
import flib.algorithms.randomforest.splitfunctions.SortingFunctions;
import flib.algorithms.randomforest.splitfunctions.VarianceSplit;
import flib.algorithms.randomforest.splitfunctions.MixedSplit;
import flib.algorithms.randomforest.splitfunctions.MixedRegression;
import flib.algorithms.clustering.RFC;
import flib.algorithms.clustering.PermClustering2;

public class RFC4 implements
java.io.Serializable {
	private RandomForest clusterrf;
	private RandomForest RF;
	private RandomForest[] regressors;
	private RandomForest[] predictors;
	private RFC cluster;
	private int leafsize1, leafsize2, leafsize3, nclust, maxit, np, np2, np3, ntree1, ntree2, ntree3, n,d, d2, mtry1, mtry2, mtry3, it, reset, pred, shr;
	private double comp,reduc;
	private double[] weights, dim_oobe, dim_oobe2, rf_param, sigma, labels;
	private boolean[] categorical;
	private double[][] trainingset;
	private int[][] ordering;
	private int[] num_dim;
	
	public RFC4(final double[] param, final double[] rf_param, final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final double[] sigma){
		// copy all input values
		this.nclust = (int)param[0];
		this.leafsize1 = (int)param[1];
		this.leafsize2 = (int)param[2];
		this.leafsize3 = (int)param[3];
		this.maxit = (int)param[4];
		this.ntree1 = (int)param[5];
		this.ntree2 = (int)param[6];
		this.ntree3 = (int)param[7];
		this.mtry1 = (int)param[8];
		this.mtry2 = (int)param[9];
		this.mtry3 = (int)param[10];
		this.it = (int)param[11];
		this.shr = (int)param[12];
		// out of bag error balancing factor
		// the standard of 1 should be fine...
		this.reset = (int)param[13];
		this.pred = (int)param[14];
		this.reduc = param[15];
		this.sigma = sigma.clone();
		this.comp = param[16];
		this.np = (int)param[17];
		this.np2 = (int)param[18];
		this.np3 = (int)param[19];
		this.n = trainingset.length;
		this.d = trainingset[0].length;
		this.weights = weights.clone();
		this.categorical = categorical.clone();
		this.rf_param = rf_param.clone();
		this.trainingset = new double[n][d];
		for (int i=0; i<this.n; i++){
			this.trainingset[i] = trainingset[i].clone();
		}
		int[] rp = Shuffle.randPerm(this.n);
		int count = 0;
		ordering = new int[nclust][np];
		for (int i=0; i<nclust; i++){
			ordering[i] = Shuffle.randPerm(np);
		}
		// variance split
		// categorical dimensions might have bizarre effects on this...
		if(labels.length!=trainingset.length){
			// rank sort the input dimensions
			double[][] trs = new double[trainingset.length][trainingset[0].length];
			double[][] sorted_ts = new double[trainingset.length][trainingset[0].length];
			// go through all dimensions
			for (int i=0; i<trs[0].length; i++){
				double[] temp = new double[trs.length];
				for (int j=0; j<trs.length; j++){
					temp[j] = trainingset[j][i];
				}
				// sort the values according to dimension
				// and then scales the values to the range [0...1]
				RankSort rs = new RankSort(temp);
				int[] r = rs.getRank();
				double[] s = rs.getSorted();
				for (int j=0; j<r.length; j++){
					trs[r[j]][i] = (double)j/r.length;
					sorted_ts[j][i] = s[j];
				}
			}
			this.labels = new double[trainingset.length];
			this.rf_param[1] = this.mtry1;
			this.rf_param[3] = this.leafsize1;
			this.RF = new RandomForest(trs,this.labels,this.weights,this.categorical,VectorFun.add(new double[this.categorical.length],1),this.rf_param,VectorFun.add(new double[1],0),new VarianceSplit(),this.ntree1);
			this.RF = RewriteForest.rewriteForest(this.RF,sorted_ts);
			this.RF.setTrainingset(this.trainingset);
		}
		// else supervised forest
		// classification
		else {
			this.labels = labels.clone();
			this.rf_param[1] = this.mtry1;
			this.rf_param[3] = this.leafsize1;
			// classification
			if (rf_param[0]>1){
				this.RF = new RandomForest(trainingset,this.labels,this.weights,this.categorical,VectorFun.add(new double[this.categorical.length],1),this.rf_param,VectorFun.add(new double[(int)rf_param[0]],1),new MixedSplit(),this.ntree1);
			}
			// regression
			else {
				this.RF = new RandomForest(trainingset,this.labels,this.weights,this.categorical,VectorFun.add(new double[this.categorical.length],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree1);
			}
		}
		// prepare a reduced swapping set
		int[][] set = trainingSetup(rp,np);
		double[][] regressset = regressionset(rp,np);
		this.dim_oobe = new double[this.nclust];
		this.dim_oobe2 = new double[this.d];
		double[] weight = VectorFun.add(new double[np],1);
		this.regressors = new RandomForest[this.nclust];
		this.predictors = new RandomForest[this.d];
		// iterate the swapping process multiple times
		for (int l=0; l<it; l++){
			System.out.println("Regression Iteration: "+Integer.toString(l));
			if (l%this.shr==0&&l>0&&this.sigma[0]>this.sigma[5]){
				this.sigma[0]-=reduc;
				if (this.sigma[0]<this.sigma[5]){
					this.sigma[0] = this.sigma[5];
				}
				this.sigma[1] = Math.pow(this.sigma[0]*this.sigma[7],this.sigma[8]);
			}
			PermClustering2.ordering(set,ordering,this.sigma,this.maxit,this.comp*nclust*np*np);
			train(regressset);
			// getting the new orderings
			if(l%this.reset==0&&l>0){
				rp = Shuffle.randPerm(n);
				// new trainingset
				set = trainingSetup(rp,np);
				// new regression set
				regressset = regressionset(rp,np);
				if(l%(int)this.pred==0&&l>0){
					double[][] regressset_pred = regressionset2(rp,np2);
					predictionSetup(regressset_pred,rp);
					System.out.println("Out of bag prediction errors: "+Arrays.toString(this.dim_oobe2));
				}
			}
			order(regressset);
		}
		//this.leafsize2 = 1;
		rp = Shuffle.randPerm(n);
		// new trainingset
		set = trainingSetup(rp,np);
		// new regression set
		regressset = regressionset(rp,np);
		double[][] regressset_pred = regressionset2(rp,np3);
		predictionSetup(regressset_pred,rp);
		System.out.println("Final out of bag prediction errors: "+Arrays.toString(this.dim_oobe2));
	}
	
	private double[][] regressionset(final int[] rp, int np){
		double[][] regressset = new double[np][d];
		for (int i=0; i<d; i++){
			for (int j=0; j<np; j++){
				regressset[j][i] = trainingset[rp[j]][i];
			}
		}
		return regressset;
	}
	
	private double[][] regressionset2(final int[] rp, int np){
		double[][] regressset = new double[np][d];
		for (int i=0; i<d; i++){
			for (int j=regressset.length-1; j>regressset.length-1-np; j--){
				regressset[j][i] = trainingset[rp[j]][i];
			}
		}
		return regressset;
	}
	
	private int[][] trainingSetup(final int[] rp, int np){
		double[][] set = new double[np][d];
		for (int i=0; i<np; i++){
			for (int j=0; j<d; j++){
				set[i][j] = trainingset[rp[i]][j];
			}
		}
		return RF.generateProximities(set);
	}
	
	private void train(final double[][] regressset){
		this.rf_param[0] = 1;
		this.rf_param[1] = this.mtry2;
		this.rf_param[3] = this.leafsize2;
		for (int i=0; i<this.nclust; i++){
			double[] val = new double[this.np];
			for (int j=0; j<this.np; j++){
				val[j] = ((double)ordering[i][j])/np;
			}
			RandomForest[] RF = new RandomForest[4];
			RF[0] = new RandomForest(regressset,val,VectorFun.add(new double[np],1),categorical,VectorFun.add(new double[d2],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree2);
			this.rf_param[1] = 1;
			RF[1] = new RandomForest(regressset,val,VectorFun.add(new double[np],1),categorical,VectorFun.add(new double[d2],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree2);
			this.rf_param[1] = 2;
			RF[2] = new RandomForest(regressset,val,VectorFun.add(new double[np],1),categorical,VectorFun.add(new double[d2],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree2);
			this.rf_param[1] = (int)(Math.log(this.d)/Math.log(2)+1);
			RF[3] = new RandomForest(regressset,val,VectorFun.add(new double[np],1),categorical,VectorFun.add(new double[d2],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree2);
			this.regressors[i] = ForestFunctions.mergeForests(RF);
			ForestFunctions.weightTrees(this.regressors[i],regressset,val,VectorFun.add(new double[this.regressors[i].getNtree()],1./this.regressors[i].getNtree()));
			ForestFunctions.removeTrees(this.regressors[i],0.1);
			this.regressors[i] = ForestFunctions.mergeForests(new RandomForest[]{this.regressors[i]});
			//System.out.println("votes after: "+Arrays.toString(val));
			// moved below
			//ordering[i] = (new RankSort(val)).getRank();
			dim_oobe[i] = Math.sqrt(this.regressors[i].outOfBagError()[0]);
		}
		System.out.println("Regression prediction errors: "+Arrays.toString(this.dim_oobe));
	}
	
	private void predictionSetup(double[][] set, int[] rp){
		double[][] votes = new double[1][0];
		votes = applyRegressors(set);
		int rl = set.length;
		this.num_dim = new int[this.d];
		for (int i=0; i<this.d; i++){
			double[] val = new double[rl];
			for (int j=0; j<rl; j++){
				val[j] = trainingset[rp[j]][i];
			}
			this.rf_param[1] = this.mtry3;
			this.rf_param[3] = this.leafsize3;
			if (categorical[i]){
				//double[] temp = VectorConv.int2double(SortingFunctions.uniqueLabels(val));
				//this.num_dim[i] = (int)VectorFun.max(temp)[0]+1;
				this.num_dim[i] = (int)VectorFun.max(val)[0]+1;
				this.rf_param[0] = this.num_dim[i];
				RandomForest[] RF = new RandomForest[4];
				RF[0] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[this.num_dim[i]],1),new MixedSplit(),this.ntree3);
				this.rf_param[1] = 1;
				RF[1] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[this.num_dim[i]],1),new MixedSplit(),this.ntree3);
				if (this.nclust>1){
					this.rf_param[1] = 2;
				}
				RF[2] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[this.num_dim[i]],1),new MixedSplit(),this.ntree3);
				this.rf_param[1] = (int)(Math.log(this.nclust)+1);
				RF[3] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[this.num_dim[i]],1),new MixedSplit(),this.ntree3);
				this.predictors[i] = ForestFunctions.mergeForests(RF);
				ForestFunctions.weightTrees(this.predictors[i],votes,val,VectorFun.add(new double[this.predictors[i].getNtree()],1./this.predictors[i].getNtree()));
				ForestFunctions.removeTrees(this.predictors[i],0.1);
				this.predictors[i] = ForestFunctions.mergeForests(new RandomForest[]{this.predictors[i]});
				this.dim_oobe2[i] = this.predictors[i].outOfBagError()[0];
			}
			else {
				this.rf_param[0] = 1;
				RandomForest[] RF = new RandomForest[4];
				RF[0] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree3);
				this.rf_param[1] = 1;
				RF[1] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree3);
				if (this.nclust>1){
					this.rf_param[1] = 2;
				}
				RF[2] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree3);
				this.rf_param[1] = (int)(Math.log(this.nclust)+1);
				RF[3] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree3);
				this.predictors[i] = ForestFunctions.mergeForests(RF);
				ForestFunctions.weightTrees(this.predictors[i],votes,val,VectorFun.add(new double[this.predictors[i].getNtree()],1./this.predictors[i].getNtree()));
				ForestFunctions.removeTrees(this.predictors[i],0.1);
				this.predictors[i] = ForestFunctions.mergeForests(new RandomForest[]{this.predictors[i]});
				this.dim_oobe2[i] = Math.sqrt(this.predictors[i].outOfBagError()[0]);
			}
		}
	}
	
	private void order(final double[][] regresset){
		double[][] temp = new double[1][0];
		temp = applyRegressors(regresset);
		for (int i=0; i<nclust; i++){
			double[] val = new double[np];
			for (int j=0; j<np; j++){
				val[j] = temp[j][i];
			}
			// rearrange the point order for the permclustering algorithm
			int[] temp2 = (new RankSort(val)).getRank();
			for (int j=0; j<np; j++){
				ordering[i][temp2[j]] = j;
			}
		}
	}
	
	public RFC cluster(int nclust, double[][] set){
		// create a collection of all the regression forests
		this.clusterrf = ForestFunctions.mergeForests(this.regressors);
		return new RFC(clusterrf.getLeafIndices(set),nclust,clusterrf.getTreeSizes());
	}
		
	
	public double[][] applyRegressors(double[][] set){
		int sl = set.length;
		double[][] votes = new double[sl][nclust];
		for (int i=0; i<nclust; i++){
			double[][] temp = regressors[i].applyForest(set);
			for (int j=0; j<sl; j++){
				votes[j][i] =  temp[j][0];
			}
		}
		return votes;
	}
	
	public double[][] applyPredictors(double[][] set){
		int sl = set.length;
		double[][] votes = new double[sl][d];
		for (int i=0; i<d; i++){
			double[][] temp = predictors[i].applyForest(set);
			for (int j=0; j<sl; j++){
				if (!categorical[i]){
					votes[j][i] =  temp[j][0];
				}
				else{
					votes[j][i] =  VectorFun.maxind(temp[j]);
				}
			}
		}
		return votes;
	}
	
	public double[][] apply(double[][] set){
		return applyRegressors(set);
	}
	
	public RandomForest[] getRegressors(){
		return this.regressors;
	}
	
	public RandomForest[] getPredictors(){
		return this.predictors;
	}
	
	public RandomForest clusterForest(){
		return this.clusterrf;
	}
	
	public RandomForest varianceForest(){
		return this.RF;
	}
}
