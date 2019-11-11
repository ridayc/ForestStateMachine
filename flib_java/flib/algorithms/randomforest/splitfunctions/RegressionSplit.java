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
import flib.math.RankSort;
import flib.math.SortPair2;
import flib.math.random.Sample;
import flib.math.random.Shuffle;

public abstract class RegressionSplit extends SplitFunction {
	
	@Override protected boolean checkLeaf(final int[] points, final double[] labels, final double[] weights, final double[] parameters, final double[] splitpurity, int depth){
		// check if any of the conditions of a leaf node are fulfilled
		double m = 0, wt = 0;
		for (int i=0; i<points.length; i++){
			m+=weights[points[i]]*labels[points[i]];
			wt+=weights[points[i]];
		}
		m/=wt;
		double err = 0;
		boolean pure = false;
		for (int i=0; i<points.length; i++){
			err+=(m-labels[points[i]])*(m-labels[points[i]])*weights[points[i]];
		}
		err/=wt;
		// checking the mean error is less than some tolerance value
		if (err<=splitpurity[0]){
			pure = true;
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
		int n = points.length;
		double[] w = new double[n];
		double[] l = new double[n];
		for (int i=0; i<n; i++){
			w[i] = weights[points[i]];
			l[i] = labels[points[i]];
		}
		double[] p = new double[2];
		double wt = 0;
		for (int i=0; i<n; i++){
			// weighted mean and mean of squares
			p[0]+=l[i]*w[i];
			p[1]+=l[i]*l[i]*w[i];
			// total leaf weight
			wt+=w[i];
		}
		p[0]/=wt;
		p[1]/=wt;
		// prepare the node content
		DecisionNode leaf = new DecisionNode();
		leaf.setLeaf(p);
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
		final double[] votes = new double[n];
		final double[] w = new double[n];
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
										votes[samples[i2][j]]+=forest.get(i2).applyTree(trainingset[samples[i2][j]])[0]*tree_weights[i2];
										w[samples[i2][j]]+=tree_weights[i2];
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
		double[] oobe2 = new double[1];
		double wt = 0;
		for (int i =0; i<n; i++){
			if (w[i]>0){
				double d = (labels[i]-votes[i]/w[i]);
				oobe2[0]+=d*d*weights[i];
				wt+=weights[i];
			}
		}
		oobe2[0] = oobe2[0]/wt;
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
		final double[] votes = new double[n];
		final double[] w = new double[n];
		double[][] oobe2 = new double[1][ntree];
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
										votes[samples[i2][j]]+=forest.get(i2).applyTree(trainingset[samples[i2][j]])[0]*tree_weights[i2];
										w[samples[i2][j]]+=tree_weights[i2];
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
			double wt = 0;
			// go through all samples to estimate an intermediate vote
			for (int j=0; j<n; j++){
				if (w[j]>0){
					double d = (labels[j]-votes[j]/w[j]);
					oobe2[0][i]+=d*d*weights[j];
					wt+=weights[j];
				}
			}
			oobe2[0][i]/=wt;
		}
		return oobe2;
	}
	
	@Override public double[][] variableImportance(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters){
		// variable preparation
		Random rng = new Random();
		int n = trainingset.length;
		int nt = (int)(n*1./Math.E);
		if (parameters[10]>0&&parameters[10]<1){
			nt = (int)(n*parameters[10]);
		}
		int ntree = forest.size();
		int dim = trainingset[0].length;
		double[] m = new double[n];
		double[] w = new double[n];
		double[] oobe2 = new double[1];
		double[][] oobe3 = new double[dim][1];
		double wt = 0;
		// for each tree
		for (int i=0; i<ntree; i++){
			// all out of bag samples
			for (int j=nt; j<n; j++){
				// sum all the individual votes
				double[] a = forest.get(i).applyTree(trainingset[samples[i][j]]);
				m[samples[i][j]]+=a[0]*tree_weights[i];
				w[samples[i][j]]+=tree_weights[i];
			}
		}
		for (int i =0; i<n; i++){
			double d = (labels[i]-m[i]/w[i]);
			oobe2[0]+=d*d*weights[i];
			wt+=weights[i];
		}
		oobe2[0]/=wt;
		
		// loop through all dimensions
		for (int i=0; i<dim; i++){
			double[] votes2 = new double[n];
			double[] m_ = new double[n];
			double[] m2_ = new double[n];
			double[] w_ = new double[n];
			int[] perm = Sample.sample(n,nt,rng);
			for (int j=0; j<ntree; j++){
				// all out of bag samples
				for (int k=n-nt; k<n; k++){
					double[] permtrain = trainingset[samples[j][k]];
					// we store the current dimension value
					// and later need to reinsert the value
					double t = permtrain[i];
					permtrain[i] = trainingset[perm[k-n+nt]][i];
					// sum all the individual votes
					double[] a = forest.get(i).applyTree(permtrain);
					permtrain[i] = t;
					//int l = a.length/2;
					//int b = VectorFun.weightSearch(a,a[2*l-1]*0.5);
					//m_[samples[j][k]]+=a[b];
					//m2_[samples[j][k]]+=a[b]*a[b];
					//w_[samples[j][k]]++;
					m_[samples[j][k]]+=a[0]*tree_weights[j];
					w_[samples[j][k]]+=tree_weights[j];
				}
			}
			for (int j =0; j<n; j++){
				double d = (labels[j]-m_[j]/w_[j]);
				oobe3[i][0]+=d*d*weights[j];
			}
			oobe3[i][0] = (Math.sqrt(oobe3[i][0])-Math.sqrt(oobe2[0]))/Math.sqrt(oobe2[0])/wt;
		}
		return oobe3;
	}
	
	@Override public double predict(final double[] point, final double[] tree_weights, ArrayList<DecisionTree> forest){
		double[] votes = applyForest(point,tree_weights,forest);
		return votes[0];
	}
	
	@Override public double[] sample(final double[] point, final int[] missing, final double[] label, ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels, final double[] weights){
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
			int[] neighbors = new int[list.size()];
			double[] w = new double[list.size()];
			Iterator<SortPair2> itr = list.iterator();
			SortPair2 sp = itr.next();
			neighbors[0] = (int)sp.getValue();
			double a = label[0]-labels[neighbors[0]];
			w[0] = sp.getOriginalIndex()*Math.exp(-a*a/label[1]/label[1]*0.5);
			int counter = 1;
			while (itr.hasNext()){
				sp = itr.next();
				neighbors[counter] = (int)sp.getValue();
				a = label[0]-labels[neighbors[counter]];
				w[counter] = w[counter-1]+sp.getOriginalIndex()*Math.exp(-a*a/label[1]/label[1]*0.5);
				counter++;
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
			double a = label[0]-labels[(int)sp.getValue()];
			double b = sp.getOriginalIndex()*Math.exp(-a*a/label[1]/label[1]*0.5);
			if (b>m[1]){
				m[1] = b;
				m[0] = sp.getValue();
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
		double a = label[0]-labels[(int)sp.getValue()];
		double b = sp.getOriginalIndex()*Math.exp(-a*a/label[1]/label[1]*0.5);
		cs[0] = Math.pow(b,p);
		o[0] = (int)sp.getValue();
		int counter = 1;
		while (itr.hasNext()){
			sp = itr.next();
			a = label[0]-labels[(int)sp.getValue()];
			b = sp.getOriginalIndex()*Math.exp(-a*a/label[1]/label[1]*0.5);
			if (type%2==0){
				cs[counter] = cs[counter-1]+Math.pow(b,p);
			}
			else {
				cs[counter] = b;
			}
			o[counter] = (int)sp.getValue();
			counter++;
		}
		int a2 = 0;
		if (type%2==0){
			a2 = o[VectorFun.binarySearch(cs,(new Random()).nextDouble()*cs[counter-1])];
		}
		else {
			int[] o2 = (new RankSort(cs)).getRank();
			int b2 = (int)p;
			if (b2>o2.length){
				b2 = o2.length;
			}
			a2 = o[o2[o2.length-1-(new Random()).nextInt(b2)]];
		}
		if (type/2==0){
			return trainingset[a2].clone();
		}
		else {
			double[] temp = new double[1];
			temp[0] = a2;
			return temp;
		}
	}
	
	public double[] applyForest(final double[] point, final int[] missing, final ArrayList<DecisionTree> forest, int num_class){
		double[] votes = forest.get(0).applyTree(point,missing,num_class+1);
		for (int i=1; i<forest.size(); i++){
			double[] a = forest.get(i).applyTree(point,missing,num_class+1);
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