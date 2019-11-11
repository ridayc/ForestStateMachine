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
import flib.algorithms.randomforest.splitfunctions.SortingFunctions;
import flib.algorithms.randomforest.splitfunctions.MixedSplit;
import flib.algorithms.randomforest.splitfunctions.MixedRegression;
import flib.algorithms.clustering.RFC;
import flib.algorithms.clustering.PermClustering;

public class RFC2 implements
java.io.Serializable {
	private RandomForest clusterrf;
	private RandomForest[] regressors, regressors2;
	private RandomForest[] predictors;
	private RFC cluster;
	private RFC[] cluster2;
	private int leafsize1, leafsize2, nclust, clustit, maxit, scrit, np, np2, dimclass, ntree1, ntree2, n,d, d2, mtry1, mtry2, ob, balance;
	private double comp,err,sg2,scrdim;
	private double[] weights,weights2, dim_oobe, dim_oobe2, dim_oobe3, rf_param, sigma2;
	private boolean[] categorical, categorical2, categorical3, hidden;
	private double[][] trainingset;
	private int[] num_dim, ord;
	private int[][] ordering;
	private boolean level2 = false;
	
	public RFC2(final double[] param, final double[] rf_param, final double[][] trainingset, final double[] weights, final boolean[] categorical, final double[] weights2, final double[] sigma2, final boolean[] hidden){
		// copy all input values
		this.leafsize1 = (int)param[0];
		this.nclust = (int)param[1];
		this.clustit = (int)param[2];
		this.maxit = (int)param[3];
		this.dimclass = (int)param[4];
		if (dimclass==2){
			level2 = true;
		}
		this.ntree1 = (int)param[5];
		this.ntree2 = (int)param[6];
		this.balance = (int)param[7];
		// out of bag error balancing factor
		// the standard of 1 should be fine...
		this.ob = (int)param[8];
		this.mtry1 = (int)param[9];
		this.mtry2 = (int)param[10];
		this.leafsize2 = (int)param[11];
		this.sigma2 = sigma2.clone();
		this.sg2 = sigma2[1];
		this.comp = param[12];
		this.err = param[13];
		this.scrit = (int)param[14];
		this.scrdim = param[15];
		this.np = (int)param[16];
		this.np2 = (int)param[17];
		if (ob>scrit){
			ob = balance*(int)(ob/balance);
		}
		this.n = trainingset.length;
		this.d = trainingset[0].length;
		this.weights = weights.clone();
		this.weights2 = weights2.clone();
		this.categorical = categorical.clone();
		this.hidden = hidden.clone();
		this.rf_param = rf_param.clone();
		this.trainingset = new double[n][d];
		this.ord = new int[nclust];
		for (int i=0; i<this.n; i++){
			this.trainingset[i] = trainingset[i].clone();
		}
		int dim = 0;
		for (int i=0; i<d; i++){
			if(!hidden[i]){
				dim++;
			}
		}
		categorical2 = new boolean[dim];
		categorical3 = new boolean[dim+nclust];
		int[] rp = Shuffle.randPerm(this.n);
		int count = 0;
		for (int i=0; i<d; i++){
			if(!hidden[i]){
				categorical2[count] = categorical[i];
				categorical3[count] = categorical[i];
				count++;
			}
		}
		this.d2 = count;
		double[][] regressset = regressionset(rp,np);
		ordering = new int[nclust][np];
		for (int i=0; i<nclust; i++){
			ordering[i] = Shuffle.randPerm(np);
		}
		// prepare a reduced swapping set
		double[][] set = trainingSetup(rp,np);
		this.dim_oobe = new double[this.nclust];
		this.dim_oobe2 = new double[this.nclust];
		this.dim_oobe3 = new double[this.d];
		this.sigma2[1] = (int)this.sigma2[0]*this.sigma2[1];
		double[] weight = VectorFun.add(new double[np],1);
		this.regressors = new RandomForest[nclust];
		this.regressors2 = new RandomForest[nclust];
		this.predictors = new RandomForest[this.d];
		for (int l=0; l<ob; l++){
			System.out.println("Regression Iteration: "+Integer.toString(l));
			if (l%this.scrit==0&&l>0&&this.sigma2[1]>this.sigma2[3]){
				this.sigma2[1]--;
			}
			PermClustering.ordering(set,ordering,categorical,weights2,this.sigma2,this.scrdim,this.clustit,this.comp*nclust*np*np,this.err);
			train(regressset);
			if(level2){
				double[][] regressset2 = regression(regressset);
				train2(regressset2);
			}
			// getting the new orderings
			if(l%this.balance==0&&l>0){
				rp = Shuffle.randPerm(n);
				// new trainingset
				set = trainingSetup(rp,np);
				// new regression set
				regressset = regressionset(rp,np);
				if(l%(int)this.scrdim==0&&l>0){
					double[][] regressset_pred = regressionset2(rp,np2);
					predictionSetup(regressset_pred,rp);
					System.out.println("Out of bag prediction errors: "+Arrays.toString(this.dim_oobe3));
				}
			}
			order(regressset);
		}
	}
	
	private double[][] regressionset(final int[] rp, int np){
		double[][] regressset = new double[np][d2];
		int count = 0;
		for (int i=0; i<d; i++){
			if(!hidden[i]){
				for (int j=0; j<np; j++){
					regressset[j][count] = trainingset[rp[j]][i];
				}
				count++;
			}
		}
		return regressset;
	}
	
	private double[][] regressionset2(final int[] rp, int np){
		double[][] regressset = new double[np][d2];
		int count = 0;
		for (int i=0; i<d; i++){
			if(!hidden[i]){
				for (int j=regressset.length-1; j>regressset.length-1-np; j--){
					regressset[j][count] = trainingset[rp[j]][i];
				}
				count++;
			}
		}
		return regressset;
	}
	
	private double[][] trainingSetup(final int[] rp, int np){
		double[][] set = new double[np][d];
		for (int i=0; i<np; i++){
			for (int j=0; j<d; j++){
				set[i][j] = trainingset[rp[i]][j];
			}
		}
		return set;
	}
	
	private void train(final double[][] regressset){
		this.rf_param[0] = 1;
		this.rf_param[1] = this.mtry1;
		this.rf_param[3] = this.leafsize1;
		for (int i=0; i<this.nclust; i++){
			double[] val = new double[this.np];
			for (int j=0; j<this.np; j++){
				//val[ordering[i][j]] = Math.signum(j-np*0.5)*Math.pow(Math.abs(j-np*0.5),balance)/this.np;
				val[ordering[i][j]] = ((double)j)/np;
			}
			RandomForest[] RF = new RandomForest[4];
			RF[0] = new RandomForest(regressset,val,VectorFun.add(new double[np],1),categorical2,VectorFun.add(new double[d2],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree1);
			this.rf_param[1] = 1;
			RF[1] = new RandomForest(regressset,val,VectorFun.add(new double[np],1),categorical2,VectorFun.add(new double[d2],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree1);
			this.rf_param[1] = 2;
			RF[2] = new RandomForest(regressset,val,VectorFun.add(new double[np],1),categorical2,VectorFun.add(new double[d2],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree1);
			this.rf_param[1] = (int)(Math.log(this.d2)+1);
			RF[3] = new RandomForest(regressset,val,VectorFun.add(new double[np],1),categorical2,VectorFun.add(new double[d2],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree1);
			this.regressors[i] = ForestFunctions.mergeForests(RF);
			ForestFunctions.weightTrees(this.regressors[i],regressset,val,VectorFun.add(new double[this.regressors[i].getNtree()],1./this.regressors[i].getNtree()));
			ForestFunctions.removeTrees(this.regressors[i],0.1);
			this.regressors[i] = ForestFunctions.mergeForests(new RandomForest[]{this.regressors[i]});
			//System.out.println("votes after: "+Arrays.toString(val));
			// moved below
			//ordering[i] = (new RankSort(val)).getRank();
			dim_oobe[i] = Math.sqrt(this.regressors[i].outOfBagError()[0]);
		}
		System.out.println("Regression prediction errors (1): "+Arrays.toString(this.dim_oobe));
	}
	
	private void train2(final double[][] regressset2){
		this.rf_param[0] = 1;
		this.rf_param[1] = this.mtry1;
		this.rf_param[3] = this.leafsize1;
		ord = (new RankSort(dim_oobe)).getRank();
		for (int i=0; i<this.nclust; i++){
			double[] val = new double[this.np];
			for (int j=0; j<this.np; j++){
				//val[ordering[ord[i]][j]] = Math.signum(j-np*0.5)*Math.pow(Math.abs(j-np*0.5),balance)/this.np;
				val[ordering[i][j]] = ((double)j)/np;
			}
			this.regressors2[ord[i]] = new RandomForest(regressset2,val,VectorFun.add(new double[np],1),categorical3,VectorFun.add(new double[categorical3.length],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree1);
			double[][] temp = this.regressors2[ord[i]].applyForest(regressset2);
			for (int j=0; j<this.np; j++){
				regressset2[j][d2+ord[i]] =  temp[j][0];
			}
			//System.out.println("votes after: "+Arrays.toString(val));
			// moved below
			//ordering[i] = (new RankSort(val)).getRank();
			dim_oobe2[ord[i]] = Math.sqrt(this.regressors2[ord[i]].outOfBagError()[0]);
		}
		System.out.println("Regression prediction errors (2): "+Arrays.toString(this.dim_oobe2));
	}
	
	private double[][] regression(double[][] regressset){
		int rl = regressset.length;
		double[][] regressset2 = new double[rl][d2+nclust];
		for (int i=0; i<rl; i++){
			for (int j=0; j<d2; j++){
				regressset2[i][j] = regressset[i][j];
			}
		}
		for (int i=0; i<nclust; i++){
			double[][] temp = this.regressors[i].applyForest(regressset);
			for (int j=0; j<this.np; j++){
				regressset2[j][d2+i] =  temp[j][0];
			}
		}
		return regressset2;
	}
	
	private void predictionSetup(double[][] set, int[] rp){
		double[][] votes = new double[1][0];
		if (level2){
			votes = applyRegressors2(set);
		}
		else {
			votes = applyRegressors(set);
		}
		int rl = set.length;
		this.num_dim = new int[this.d];
		for (int i=0; i<this.d; i++){
			double[] val = new double[rl];
			for (int j=0; j<rl; j++){
				val[j] = trainingset[rp[j]][i];
			}
			/*
			if (!categorical[i]){
				int[] val2 = (new RankSort(val)).getRank();
				for (int j=0; j<rl; j++){
					val[val2[j]] = ((double)j)/rl;
				}
			}
			*/
			this.rf_param[1] = this.mtry2;
			this.rf_param[3] = this.leafsize2;
			if (categorical[i]){
				//double[] temp = VectorConv.int2double(SortingFunctions.uniqueLabels(val));
				//this.num_dim[i] = (int)VectorFun.max(temp)[0]+1;
				this.num_dim[i] = (int)VectorFun.max(val)[0]+1;
				this.rf_param[0] = this.num_dim[i];
				RandomForest[] RF = new RandomForest[4];
				RF[0] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[this.num_dim[i]],1),new MixedSplit(),this.ntree2);
				this.rf_param[1] = 1;
				RF[1] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[this.num_dim[i]],1),new MixedSplit(),this.ntree2);
				if (this.nclust>1){
					this.rf_param[1] = 2;
				}
				RF[2] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[this.num_dim[i]],1),new MixedSplit(),this.ntree2);
				this.rf_param[1] = (int)(Math.log(this.nclust)+1);
				RF[3] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[this.num_dim[i]],1),new MixedSplit(),this.ntree2);
				this.predictors[i] = ForestFunctions.mergeForests(RF);
				ForestFunctions.weightTrees(this.predictors[i],votes,val,VectorFun.add(new double[this.predictors[i].getNtree()],1./this.predictors[i].getNtree()));
				ForestFunctions.removeTrees(this.predictors[i],0.1);
				this.predictors[i] = ForestFunctions.mergeForests(new RandomForest[]{this.predictors[i]});
				this.dim_oobe3[i] = this.predictors[i].outOfBagError()[0];
			}
			else {
				this.rf_param[0] = 1;
				RandomForest[] RF = new RandomForest[4];
				RF[0] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree2);
				this.rf_param[1] = 1;
				RF[1] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree2);
				if (this.nclust>1){
					this.rf_param[1] = 2;
				}
				RF[2] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree2);
				this.rf_param[1] = (int)(Math.log(this.nclust)+1);
				RF[3] = new RandomForest(votes,val,VectorFun.add(new double[rl],1),new boolean[this.nclust],VectorFun.add(new double[this.nclust],1),this.rf_param,VectorFun.add(new double[1],0),new MixedRegression(),this.ntree2);
				this.predictors[i] = ForestFunctions.mergeForests(RF);
				ForestFunctions.weightTrees(this.predictors[i],votes,val,VectorFun.add(new double[this.predictors[i].getNtree()],1./this.predictors[i].getNtree()));
				ForestFunctions.removeTrees(this.predictors[i],0.1);
				this.predictors[i] = ForestFunctions.mergeForests(new RandomForest[]{this.predictors[i]});
				this.dim_oobe3[i] = Math.sqrt(this.predictors[i].outOfBagError()[0]);
			}
		}
	}
	
	private void order(final double[][] regresset){
		double[][] temp = new double[1][0];
		if(level2){
			temp = applyRegressors2(regresset);
		}
		else {
			temp = applyRegressors(regresset);
		}
		for (int i=0; i<nclust; i++){
			double[] val = new double[np];
			for (int j=0; j<np; j++){
				val[j] = temp[j][i];
			}
			ordering[i] = (new RankSort(val)).getRank();
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
	
	public double[][] applyRegressors2(double[][] set){
		int sl = set.length;
		double[][] regressset = applyRegressors(set);
		double[][] regressset2 = new double[sl][d2+nclust];
		for (int i=0; i<sl; i++){
			for (int j=0; j<d2; j++){
				regressset2[i][j] = set[i][j];
			}
		}
		for (int i=0; i<sl; i++){
			for (int j=0; j<nclust; j++){
				regressset2[i][j+d2] = regressset[i][j];
			}
		}
		for (int i=0; i<nclust; i++){
			double[][] temp = regressors2[ord[i]].applyForest(regressset2);
			for (int j=0; j<sl; j++){
				regressset2[j][d2+ord[i]] =  temp[j][0];
			}
		}
		double[][] votes = new double[sl][nclust];
		for (int i=0; i<sl; i++){
			for (int j=0; j<nclust; j++){
				votes[i][j] = regressset2[i][d2+j];
			}
		}
		return votes;
	}
	
	public double[][] apply(double[][] set){
		if (level2){
			return applyRegressors2(set);
		}
		else {
			return applyRegressors(set);
		}
	}
	
	public RandomForest[] getRegressors(){
		return this.regressors;
	}
	
	public RandomForest[] getRegressors2(){
		return this.regressors2;
	}
	
	public RandomForest[] getPredictors(){
		return this.predictors;
	}
	
	public RandomForest clusterForest(){
		return this.clusterrf;
	}
}
