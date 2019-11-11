package flib.algorithms.clustering;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Random;
import java.lang.Math;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.RankSort;
import flib.math.random.Shuffle;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.splitfunctions.SortingFunctions;
import flib.algorithms.randomforest.splitfunctions.GiniSplit;
import flib.algorithms.randomforest.splitfunctions.VarianceSplit;
import flib.algorithms.randomforest.RewriteForest;
import flib.algorithms.randomforest.splitfunctions.ClassicRegression;

public class RFC3 implements
java.io.Serializable {
	private RandomForest clusterrf;
	private RandomForest[] predictors;
	private int type, tclass, regtype, ntree1, ntree2, top, bottom, np;
	private double alpha, alpha2;
	private double[] labels, weights, param, rf_param1, rf_param2;
	private boolean[] categorical;
	private double[][] trainingset;
	private int[] pp;
	
	public RFC3(final double[] param, final double[] rf_param1, final double[] rf_param2, final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical){
		// copy all input values
		this.type = (int)param[0];
		this.tclass = (int)param[1];
		this.top = (int)param[2];
		// not used at the moment
		this.alpha2 = param[3];
		this.ntree1 = (int)param[4];
		this.ntree2 = (int)param[5];
		this.regtype = (int)param[6];
		this.np = (int)param[7];
		this.alpha = param[8];
		this.param = param.clone();
		this.rf_param1 = rf_param1.clone();
		this.rf_param2 = rf_param2.clone();
		this.categorical = categorical.clone();
		this.trainingset = new double[1][0];
		this.labels = new double[0];
		this.weights = new double[0];
		if (type<3){
			this.labels = labels.clone();
			this.weights = weights.clone();
			this.trainingset = new double[trainingset.length][trainingset[0].length];
			for (int i=0; i<this.trainingset.length; i++){
				this.trainingset[i] = trainingset[i].clone();
			}
		}
		// otherwise we create an artificial mixture
		else {
			int n = trainingset.length;
			int d = trainingset[0].length;
			this.labels = new double[2*n];
			this.trainingset = new double[2*n][d];
			this.weights = VectorFun.add(new double[2*n],1);
			for (int i=0; i<n; i++){
				this.trainingset[i] = trainingset[i].clone();
				this.labels[i+n] = 1;
			}
			Random rng = new Random();
			for (int i=0; i<d; i++){
				int[] perm = Shuffle.randPerm(n);
				for (int j=0; j<n; j++){
					this.trainingset[j+n][i] = trainingset[perm[j]][i];
				}
			}
		}
		//
		// now we set up the random forest used for distance calculation
		//
		this.clusterrf = new RandomForest();
		// regression forest
		if (type==0){
			this.clusterrf = new RandomForest(this.trainingset,this.labels,this.weights,this.categorical,VectorFun.add(new double[this.categorical.length],1),this.rf_param1,VectorFun.add(new double[1],0),new ClassicRegression(),this.ntree1);
			this.tclass = -1;
		}
		// variance split
		// categorical dimensions might have bizarre effects on this...
		else if(type==1){
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
			this.clusterrf = new RandomForest(trs,this.labels,this.weights,this.categorical,VectorFun.add(new double[this.categorical.length],1),this.rf_param1,VectorFun.add(new double[1],0),new VarianceSplit(),this.ntree1);
			this.clusterrf = RewriteForest.rewriteForest(this.clusterrf,sorted_ts);
			this.clusterrf.setTrainingset(this.trainingset);
			this.tclass = -1;
		}
		// supervised split
		else if (type==2){
			this.clusterrf = new RandomForest(this.trainingset,this.labels,this.weights,this.categorical,VectorFun.add(new double[this.categorical.length],1),this.rf_param1,VectorFun.add(new double[(int)rf_param1[0]],1),new GiniSplit(),this.ntree1);
		}
		// otherwise we separate against articial data
		else {
			this.clusterrf = new RandomForest(this.trainingset,this.labels,this.weights,this.categorical,VectorFun.add(new double[this.categorical.length],1),this.rf_param1,VectorFun.add(new double[2],1),new GiniSplit(),this.ntree1);
		}
		// get a list of potential points to use for the multidimensional scaling
		this.pp = new int[0];
		int[] tp = new int[0];
		if (this.tclass>-1){
			int count = 0;
			for (int i=0; i<this.trainingset.length; i++){
				if (this.labels[i]==this.tclass){
					count++;
				}
			}
			tp = new int[count];
			count = 0;
			for (int i=0; i<this.trainingset.length; i++){
				if (this.labels[i]==this.tclass){
					tp[count] = i;
					count++;
				}
			}
		}
		else {
			tp = new int[this.trainingset.length];
			for (int i=0; i<this.trainingset.length; i++){
				tp[i] = i;
			}
		}
		if (this.np>tp.length){
			this.np = tp.length;
		}
		if (this.np>-1){
			int[] perm = Shuffle.randPerm(tp.length);
			pp = new int[np];
			for (int i=0; i<this.np; i++){
				pp[i] = tp[perm[i]];
			}
		}
		else {
			pp = tp.clone();
		}
		Arrays.sort(pp);
		// temporary set to aqu
		double[][] temp = new double[pp.length][this.trainingset[0].length];
		for (int i=0; i<pp.length; i++){
			temp[i] = this.trainingset[pp[i]].clone();
		}
		// set leaf indices
		this.clusterrf.setProximities(pp);
		int[][] tmp = this.clusterrf.pointToSetProximities(temp,pp);
		int n = pp.length;
		double[][] distmat = new double[n][n];
		double[][] sub = new double[n][n];
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				distmat[i][j] = Math.pow((1.-(double)tmp[i][j]/ntree1),alpha);
				sub[i][j] = -1./n;
			}
			sub[i][i]+=1;
		}
		// correct the distance matrix using the Floyd-Warshall algorithm
		///*
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				for (int k=0; k<n; k++){
					// distances are adjusted to fulfilled the euclidean triangle inequality
					
					if (Math.sqrt(distmat[i][j])>Math.sqrt(distmat[i][k])+Math.sqrt(distmat[k][j])){
						distmat[i][j] = (Math.sqrt(distmat[i][k])+Math.sqrt(distmat[k][j]));
						distmat[i][j]*=distmat[i][j];
					}
					
					/*
					if (distmat[i][j]>distmat[i][k]+distmat[k][j]){
						distmat[i][j] = distmat[i][k]+distmat[k][j];
					}
					*/
				}
			}
		}
		//*/
		//System.out.println(Arrays.toString(distmat[0]));
		DenseDoubleMatrix2D KC = new DenseDoubleMatrix2D(distmat);
		//System.out.println(KC.toString());
		//System.out.println(KC.viewPart(0,0,10,10).toString());
		DenseDoubleMatrix2D IM = new DenseDoubleMatrix2D(sub);
		//System.out.println(IM.viewPart(0,0,10,10).toString());
		DenseDoubleMatrix2D DM = new DenseDoubleMatrix2D(n,n);
		// create the crossproduct matrix
		IM.zMult(KC,DM);
		DM.zMult(IM,KC,-0.5,0,false,false);
		//System.out.println(KC.toString());
		// eigenanalysis time!
		EigenvalueDecomposition eig = new EigenvalueDecomposition(KC);
		double[] diag = eig.getRealEigenvalues().toArray();
		System.out.println(Arrays.toString(diag));
		for (int i=0; i<n; i++){
			diag[i] = Math.sqrt(diag[i]);
		}
		if (top>n){
			top = n;
		}
		// get the initial centered coordinates
		double[][] X = new double[top][n];
		for (int i=0; i<n; i++){
			for (int j=0; j<top; j++){
				int v = n-j-1;
				//X[j][i] = eig.getV().get(i,v)*diag[v];
				X[j][i] = eig.getV().get(i,v);
			}
		}
		// trainingset for the predictor
		double[][] predictionset = new double[1][0];
		boolean[] cat = new boolean[0];
		if (this.regtype==0){
			predictionset = new double[n][this.trainingset[0].length];
			for (int i=0; i<n; i++){
				predictionset[i] = this.trainingset[pp[i]].clone();
			}
			cat = this.categorical.clone();
		}
		else {
			predictionset = new double[n][n];
			for (int i=0; i<n; i++){
				for (int j=0; j<n; j++){
					predictionset[i][j] = tmp[i][j];
				}
			}
			cat = new boolean[n];
		}
		// train regressors for all dimensions
		this.predictors = new RandomForest[top];
		double[] oobe = new double[top];
		for (int i=0; i<top; i++){
			this.predictors[i] = new RandomForest(predictionset,X[i],VectorFun.add(new double[n],1),cat,VectorFun.add(new double[cat.length],1),this.rf_param2,VectorFun.add(new double[1],0),new ClassicRegression(),this.ntree2);
			oobe[i] = Math.sqrt(this.predictors[i].outOfBagError()[0])/Math.sqrt(VectorFun.sum(VectorFun.mult(X[i],X[i]))/top-VectorFun.sum(X[i])*VectorFun.sum(X[i])/top/top);
		}
		System.out.println("Out of bag regression errors: "+Arrays.toString(oobe));
		// estimate of backwards prediction strength
		if (true){
			int d = trainingset[0].length;
			double[] oobe2 = new double[d];
			double[][] X2 = new double[n][top];
			for (int i=0; i<n; i++){
				for (int j=0; j<top; j++){
					X2[i][j] = X[j][i];
				}
			}
			for (int i=0; i<d; i++){
				double[] lab = new double[n];
				for (int j=0; j<n; j++){
					lab[j] = this.trainingset[pp[j]][i];
				}
				if (categorical[i]){
					double[] temp2 = VectorConv.int2double(SortingFunctions.uniqueLabels(lab));
					int num_dim = (int)VectorFun.max(temp2)[0]+1;
					double[] rf_param = rf_param1.clone();
					rf_param[0] = num_dim;
					rf_param[1] = top;
					oobe2[i] = (new RandomForest(X2,temp2,VectorFun.add(new double[n],1),new boolean[top],VectorFun.add(new double[top],1),rf_param,VectorFun.add(new double[num_dim],1),new GiniSplit(),this.ntree1)).outOfBagError()[0];
				}
				else {
					double[] rf_param = rf_param1.clone();
					rf_param[0] = 1;
					rf_param[1] = top;
					oobe2[i] = Math.sqrt((new RandomForest(X2,lab,VectorFun.add(new double[n],1),new boolean[top],VectorFun.add(new double[top],1),rf_param,VectorFun.add(new double[1],0),new ClassicRegression(),this.ntree1)).outOfBagError()[0]);
				}
			}
			System.out.println("Reverse out of bag regression errors: "+Arrays.toString(oobe2));
		}
	}
	
	public RandomForest getForest(){
		return this.clusterrf;
	}
	
	public RandomForest[] getPredictors(){
		return this.predictors;
	}
	
	public int getRegtype(){
		return this.regtype;
	}
	
	public int[] getPP(){
		return pp;
	}
}
