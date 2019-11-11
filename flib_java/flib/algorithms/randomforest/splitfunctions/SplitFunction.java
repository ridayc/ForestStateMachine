package flib.algorithms.randomforest.splitfunctions;

import java.lang.Runtime;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import flib.algorithms.randomforest.DecisionTree;
import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;

public abstract class SplitFunction implements
java.io.Serializable {
	
	public boolean isLeaf(TreeNode<DecisionNode> current, final int[] points, final double[] labels, final double[] weights, final boolean[] categorical, final double[] parameters, final double[] splitpurity, int depth, int leafindex){
		// first use an abstract leaf check
		if (checkLeaf(points,labels,weights,parameters,splitpurity,depth)){
			// set the internal leaf parameters
			setLeaf(current,points,labels,weights,categorical,parameters,leafindex);
			return true;
		}
		return false;
	}
	
	public double[] applyForest(final double[] point, final double[] tree_weight, ArrayList<DecisionTree> forest){
		double[] votes = forest.get(0).applyTree(point).clone();
		for (int j=0; j<votes.length; j++){
			votes[j]*=tree_weight[0];
		}
		for (int i=1; i<forest.size(); i++){
			double[] a = forest.get(i).applyTree(point);
			for (int j=0; j<a.length; j++){
				votes[j]+=a[j]*tree_weight[i];
			}
		}
		return votes;
	}
	
	public double[][] applyForest(final double[][] point, final double[] tree_weight, final ArrayList<DecisionTree> forest){
		final double[] tmp = forest.get(0).applyTree(point[0]).clone();
		//final double[][][] votes_tmp = new double[forest.size()][tmp.length][point.length];
		final double[][] votes = new double[point.length][tmp.length];
		final int NUM_CORES = Runtime.getRuntime().availableProcessors();
		//final int NUM_CORES = 1;
		//System.out.println("Cores for use: "+Integer.toString(NUM_CORES));
		ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
		try {
			for (int i=0; i<NUM_CORES; i++){
				final int i2 = i;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						for (int j=0; j<forest.size(); j++){
							for (int k=i2; k<point.length; k+=NUM_CORES){
								double[] a = forest.get(j).applyTree(point[k]);
								for (int l=0; l<a.length; l++){
									votes[k][l]+=a[l]*tree_weight[j];
								}
							}
						}
					}
				});
			}
		}
		finally {
			exec.shutdown();
		}
		while(!exec.isTerminated()){
			// wait
		}
		return votes;
	}
	
	// method is synchronized because it manipulates
	// a private static field for temporary use
	public int[][] getLeafIndices(final double[][] point, final ArrayList<DecisionTree> forest){
		final int[][] indices = new int[point.length][forest.size()];
		final int NUM_CORES = Runtime.getRuntime().availableProcessors();
		ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
		try {
			for (int i=0; i<NUM_CORES; i++){
				final int i2 = i;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						try {
							for (int j=0; j<forest.size(); j++){
								for (int k=i2; k<point.length; k+=NUM_CORES){
									indices[k][j] = forest.get(j).getLeafIndex(point[k]);
								}
							}
						}
						catch (Throwable t){
							System.out.println("Don't want to be here");
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
		return indices;
	}

	protected abstract boolean checkLeaf(final int[]  points, final double[] labels, final double[] weights, final double[] parameters, final double[] splitpurity, int depth);
	
	protected abstract void setLeaf(TreeNode<DecisionNode> current, final int[] points, final double[] labels, final double[] weights, final boolean[] categorical, final double[] parameters, int leafindex);
	
	public abstract boolean split(TreeNode<DecisionNode> current, final int[] points, final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final int[] dim, final double[] parameters, final double[] splitpurity, int leafindex);
	
	public abstract double[] oobe(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters);
	
	public abstract double[][] oobeConvergence(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters);
	
	public abstract double[][] variableImportance(final int[][] samples, int num_class, final double[][] trainingset, final double[] labels, final double[] weights, final double[] tree_weights, final ArrayList<DecisionTree> forest, final double[] parameters);
	
	public abstract double[] applyForest(final double[] point, final int[] missing, final ArrayList<DecisionTree> forest, int num_class);
	
	public abstract double predict(final double[] point, final double[] tree_weights, ArrayList<DecisionTree> forest);
	
	public abstract double[] sample(final double[] point, final int[] missing, final double[] label, ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels, final double[] weights);
	
	public abstract double[] nearestNeighbor(final double[] point, final int[] missing, final double[] label, ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels);
	
	public abstract double[] sampledNeighbor(final double[] point, final int[] missing, final double[] label, ArrayList<DecisionTree> forest, final double[][] trainingset, final double[] labels, int type, double p);
}