package flib.algorithms.randomforest.splitfunctions;

import java.util.ArrayList;
import java.util.Random;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Arrays;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.lang.Math;
import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;
import flib.algorithms.randomforest.DecisionTree;
import flib.algorithms.randomforest.splitfunctions.SplitFunction;
import flib.math.VectorFun;
import flib.math.SortPair2;
import flib.math.RankSort;
import flib.math.random.Sample;
import flib.math.random.Shuffle;

public abstract class ClassificationSplit extends SplitFunction {
	
	@Override protected boolean checkLeaf(final int[] points, final double[] labels, final double[] weights, final double[] parameters, final double[] splitpurity, int depth){
		// check if any of the conditions of a leaf node are fulfilled
		// check if any of the splitpurity conditions are fulfilled
		if (points.length==0){
			System.out.println("Bad stuff in the check leaf function...");
		}
		boolean pure = false;
		double[] w = new double[(int)parameters[0]];
		for (int i=0; i<points.length; i++){
			w[(int)labels[points[i]]]+=weights[points[i]];
		}
		double wt = VectorFun.sum(w);
		for (int i=0; i<parameters[0]; i++){
			if (w[i]/wt>=splitpurity[i]){
				pure = true;
			}
		}
		// stopping conditions
		if (pure||depth>=parameters[2]||wt<=parameters[3]){
			return true;
		}
		else {
			return false;
		}
	}
	
	@Override protected void setLeaf(TreeNode<DecisionNode> current, final int[] points, final double[] labels, final double[] weights, final boolean[] categorical, final double[] parameters, int leafindex){
		// calculate the purity fractions
		double[] w = new double[(int)parameters[0]];
		for (int i=0; i<points.length; i++){
			w[(int)labels[points[i]]]+=weights[points[i]];
		}
		// prepare the node content
		DecisionNode leaf = new DecisionNode();
		double wt = VectorFun.sum(w);
		leaf.setLeaf(VectorFun.div(w,wt));
		leaf.setWeight(wt);
		leaf.setLeafIndex(leafindex);
		if (parameters[12]!=1){
			leaf.setPoints(points);
		}
		else {
			leaf.setPoints(new int[0]);
		}
		// set these purity fractions in the current node
		current.setData(leaf);
	}
	
	@Override public double[] oobe(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters){
		final int n = trainingset.length;
		int t = (int)(n*(1-1./Math.E));
		if (parameters[10]>0&&parameters[10]<1){
			t = (int)(n*parameters[10]);
		}
		final int nt = t;
		int ntree = forest.size();
		final double[][] votes = new double[n][num_class];
		// for each tree
		for (int i=0; i<ntree; i++){
			final int NUM_CORES = Runtime.getRuntime().availableProcessors();
			ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
			final int i2 = i;
			try {
				for (int k=0; k<NUM_CORES; k++){
					final int k2 = k;
					exec.submit(new Runnable() {
						@Override
						public void run(){
							try {
								// all out of bag samples
								for (int j=nt; j<n; j++){
									if (samples[i2][j]%NUM_CORES==k2){
										// sum all the individual votes
										VectorFun.addi(votes[samples[i2][j]],VectorFun.mult(forest.get(i2).applyTree(trainingset[samples[i2][j]]),tree_weights[i2]));
									}
								}
							}
							catch (Throwable t){
								System.out.println("Failure in the oobe convergence");
								//t.printStackTrace();
							}
						}
					});
				}
			}
			finally {
				exec.shutdown();
			}
			//exec.awaitTermination(time, time_unit);
			while(!exec.isTerminated()){
				// wait
			}
		}
		double[] oobe2 = new double[num_class+1];
		double w = VectorFun.sum(weights);
		double[] counter = new double[num_class];
		for (int i=0; i<n; i++){
			counter[(int)labels[i]]+=weights[i];
		}
		for (int i=0; i<n; i++){
			if (parameters[11]==0){
				int v = VectorFun.maxind(votes[i]);
				if ((int)labels[i]==v){
					oobe2[0]+=weights[i];
					oobe2[v+1]+=weights[i];
				}
			}
			else {
				double v = votes[i][(int)labels[i]]/VectorFun.sum(votes[i]);
				if (Double.isNaN(v)){
					v = 0;
				}
				oobe2[0]+=weights[i]*v*v;
				oobe2[(int)labels[i]+1]+=weights[i]*v*v;
			}
		}
		oobe2[0] = 1-oobe2[0]/w;
		for (int i=0; i<num_class; i++){
			if (counter[i]>0){
				oobe2[i+1] = 1-oobe2[i+1]/counter[i];
			}
		}
		return oobe2;
	}
	
	@Override public double[][] oobeConvergence(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters){
		final int n = trainingset.length;
		int t = (int)(n*(1-1./Math.E));
		if (parameters[10]>0&&parameters[10]<1){
			t = (int)(n*parameters[10]);
		}
		final int nt = t;
		int ntree = forest.size();
		final double[][] votes = new double[n][num_class];
		double[][] oobe2 = new double[num_class+1][ntree];
		double w = VectorFun.sum(weights);
		double[] counter = new double[num_class];
		for (int i=0; i<n; i++){
			counter[(int)labels[i]]+=weights[i];
		}
		// for each tree
		for (int i=0; i<ntree; i++){
			final int NUM_CORES = Runtime.getRuntime().availableProcessors();
			ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
			final int i2 = i;
			try {
				for (int k=0; k<NUM_CORES; k++){
					final int k2 = k;
					exec.submit(new Runnable() {
						@Override
						public void run(){
							try {
								// all out of bag samples
								for (int j=nt; j<n; j++){
									if (samples[i2][j]%NUM_CORES==k2){
										// sum all the individual votes
										VectorFun.addi(votes[samples[i2][j]],VectorFun.mult(forest.get(i2).applyTree(trainingset[samples[i2][j]]),tree_weights[i2]));
									}
								}
							}
							catch (Throwable t){
								System.out.println("Failure in the oobe convergence");
								//t.printStackTrace();
							}
						}
					});
				}
			}
			finally {
				exec.shutdown();
			}
			//exec.awaitTermination(time, time_unit);
			while(!exec.isTerminated()){
				// wait
			}
			// go through all samples to estimate an intermediate vote
			for (int j=0; j<n; j++){
				if (parameters[11]==0){
					int v = VectorFun.maxind(votes[j]);
					if ((int)labels[j]==v){
						oobe2[0][i]+=weights[j];
						oobe2[v+1][i]+=weights[j];
					}
				}
				else {
					double v = votes[j][(int)labels[j]]/VectorFun.sum(votes[j]);
					if (Double.isNaN(v)){
						v = 0;
					}
					oobe2[0][i]+=weights[j]*v*v;
					oobe2[(int)labels[j]+1][i]+=weights[j]*v*v;
				}
			}
			oobe2[0][i] = 1-oobe2[0][i]/w;
			for (int j=0; j<num_class; j++){
				if (counter[j]>0){
					oobe2[j+1][i] = 1-oobe2[j+1][i]/counter[j];
				}
			}
		}
		return oobe2;
	}
	
	
	
	@Override public double[][] variableImportance(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters){
		// variable preparation
		Random rng = new Random();
		final int n = trainingset.length;
		int t = (int)(n*(1-1./Math.E));
		if (parameters[10]>0&&parameters[10]<1){
			t = (int)(n*parameters[10]);
		}
		final int nt = t;
		int ntree = forest.size();
		int dim = trainingset[0].length;
		double[][] votes = new double[n][num_class];
		double[] oobe2 = new double[num_class+1];
		double[][] oobe3 = new double[num_class+1][dim];
		for (int i=0; i<ntree; i++){
			// all out of bag samples
			for (int j=nt; j<n; j++){
				// sum all the individual votes
				VectorFun.addi(votes[samples[i][j]],VectorFun.mult(forest.get(i).applyTree(trainingset[samples[i][j]]),tree_weights[i]));
			}
		}
		int[] v = new int[n];
		for (int i=0; i<n; i++){
			v[i] = VectorFun.maxind(votes[i]);
			if ((int)labels[i]==v[i]){
				oobe2[0]+=weights[i];
				oobe2[v[i]+1]+=weights[i];
			}
		}
		double w = VectorFun.sum(weights);
		double[] counter = new double[num_class];
		for (int i=0; i<n; i++){
			counter[(int)labels[i]]+=weights[i];
		}
		// loop through all dimensions
		for (int i=0; i<dim; i++){
			final double[][] votes2 = new double[n][num_class];
			final int i2 = i;
			// draw a random sample location for this dimension
			final int[] perm = Sample.sample(n,n,rng);
			// go through each tree
			for (int j=0; j<ntree; j++){
				final int NUM_CORES = Runtime.getRuntime().availableProcessors();
				ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
				final int j2 = j;
				try {
					for (int l=0; l<NUM_CORES; l++){
						final int l2 = l;
						exec.submit(new Runnable() {
							@Override
							public void run(){
								try {
									// all out of bag samples
									for (int k=nt; k<n; k++){
										// draw a random sample location for this dimension
										if (samples[j2][k]%NUM_CORES==l2){
											// sum all the individual votes
											double[] permtrain = trainingset[samples[j2][k]];
											// we store the current dimension value
											// and later need to reinsert the value
											double a = permtrain[i2];
											// create a permuted version of the out of bag sample
											permtrain[i2] = trainingset[perm[samples[j2][k]]][i2];
											// sum all the individual votes
											VectorFun.addi(votes2[samples[j2][k]],VectorFun.mult(forest.get(j2).applyTree(permtrain),tree_weights[j2]));
											// reinserting the original value
											permtrain[i2] = a;
										}
									}
								}
								catch (Throwable t){
									System.out.println("Failure in the variable importance");
									//t.printStackTrace();
								}
							}
						});
					}
				}
				finally {
					exec.shutdown();
				}
				//exec.awaitTermination(time, time_unit);
				while(!exec.isTerminated()){
					// wait
				}
			}
			for (int j=0; j<n; j++){
				if (v[j]==(int)labels[j]){
					int v2 = VectorFun.maxind(votes2[j]);
					if (v2==(int)labels[j]){
						oobe3[0][i]+=weights[j];
						oobe3[v[j]+1][i]+=weights[j];
					}
				}
			}
			oobe3[0][i] = (oobe2[0]-oobe3[0][i])/w;
			for (int j=0; j<num_class; j++){
				oobe3[j+1][i] = (oobe2[j+1]-oobe3[j+1][i])/counter[j];
			}
		}
		return oobe3;
	}
	
	@Override public double predict(final double[] point, final double[] tree_weights, ArrayList<DecisionTree> forest){
		return (double)VectorFun.maxind(applyForest(point,tree_weights,forest));
	}
	
	@Override public double[] sample(final double[] point, final int[] missing, final double label[], ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels, final double[] weights){
		Random rng = new Random();
		int n = trainingset.length;
		int dim = trainingset[0].length;
		int l = missing.length;
		double[] samp = point.clone();
		int[] randind = Shuffle.randPerm(l);
		for (int i=0; i<l; i++){
			int[] miss = new int[l-i];
			for (int j=0; j<l-i;j++){
				miss[j] = missing[randind[i+j]];
			}
			Arrays.sort(miss);
			TreeSet<SortPair2> list = new TreeSet<SortPair2>();
			for (int j=0; j<forest.size(); j++){
				forest.get(j).neighbors(samp,miss,list,weights);
			}
			Iterator<SortPair2> itr = list.iterator();
			int count = 0;
			while (itr.hasNext()){
				SortPair2 sp = itr.next();
				if (labels[(int)sp.getValue()]==label[0]){
					count++;
				}
			}
			itr = list.iterator();
			int[] neighbors = new int[count];
			double[] w = new double[count];
			SortPair2 sp;
			while (itr.hasNext()){
				sp = itr.next();
				if (labels[(int)sp.getValue()]==label[0]){
					neighbors[0] = (int)sp.getValue();
					w[0] = sp.getOriginalIndex();
					break;
				}
			}
			int counter = 1;
			while (itr.hasNext()){
				sp = itr.next();
				if (labels[(int)sp.getValue()]==label[0]){
					neighbors[counter] = (int)sp.getValue();
					w[counter] = w[counter-1]+sp.getOriginalIndex();
					counter++;
				}
			}
			int b = VectorFun.binarySearch(w,rng.nextDouble()*(w[counter-1]));
			samp[missing[randind[i]]] = trainingset[neighbors[b]][missing[randind[i]]];
		}
		return samp;
	}
	
	public double[] nearestNeighbor(final double[] point, final int[] missing, final double[] label, ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels){
		TreeSet<SortPair2> list = new TreeSet<SortPair2>();
		for (int i=0; i<forest.size(); i++){
			forest.get(i).neighbors(point,missing,list);
		}
		double[] m = new double[2];
		Iterator<SortPair2> itr = list.iterator();
		while (itr.hasNext()){
			SortPair2 sp = itr.next();
			if (label[0]==labels[(int)sp.getValue()]){
				if (sp.getOriginalIndex()>m[1]){
					m[1] = sp.getOriginalIndex();
					m[0] = sp.getValue();
				}
			}
		}
		return trainingset[(int)m[0]].clone();
	}
	
	public double[] sampledNeighbor(final double[] point, final int[] missing, final double[] label, ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels, int type, double p){
		TreeSet<SortPair2> list = new TreeSet<SortPair2>();
		for (int i=0; i<forest.size(); i++){
			forest.get(i).neighbors(point,missing,list);
		}
		double[] cs = new double[list.size()];
		int[] o = new int[list.size()];
		Iterator<SortPair2> itr = list.iterator();
		SortPair2 sp = itr.next();
		if (label[0]==labels[(int)sp.getValue()]){
			cs[0] = Math.pow(sp.getOriginalIndex(),p);
		}
		o[0] = (int)sp.getValue();
		int counter = 1;
		while (itr.hasNext()){
			sp = itr.next();
			if (label[0]==labels[(int)sp.getValue()]){
				if (type%2==0){
					cs[counter] = cs[counter-1]+Math.pow(sp.getOriginalIndex(),p);
				}
				else {
					cs[counter] = sp.getOriginalIndex();
				}
			}
			else {
				if (type%2==0){
					cs[counter]  = cs[counter-1];
				}
				else {
					cs[counter] = 0;
				}
			}
			o[counter] = (int)sp.getValue();
			counter++;
		}
		int a = 0;
		if (type%2==0){
			a = o[VectorFun.binarySearch(cs,(new Random()).nextDouble()*cs[counter-1])];
		}
		else {
			int[] o2 = (new RankSort(cs)).getRank();
			int b = (int)p;
			if (b>o2.length){
				b = o2.length;
			}
			a = o[o2[o2.length-1-(new Random()).nextInt(b)]];
		}
		if (type/2==0){
			return trainingset[a].clone();
		}
		else {
			double[] temp = new double[1];
			temp[0] = a;
			return temp;
		}
	}
	
	public double[] applyForest(final double[] point, final int[] missing, final ArrayList<DecisionTree> forest, int num_class){
		double[] votes = forest.get(0).applyTree(point,missing,num_class).clone();
		for (int i=1; i<forest.size(); i++){
			double[] a = forest.get(i).applyTree(point,missing,num_class);
			for (int j=0; j<a.length; j++){
				votes[j]+=a[j];
			}
		}
		for (int j=0; j<votes.length; j++){
			votes[j]/=forest.size();
		}
		return votes;
	}
}