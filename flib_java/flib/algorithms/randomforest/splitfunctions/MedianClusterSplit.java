package flib.algorithms.randomforest.splitfunctions;

import flib.algorithms.randomforest.splitfunctions.ClusterSplit;
import flib.algorithms.randomforest.splitfunctions.SplittingFunctions;

public class MedianClusterSplit extends ClusterSplit {
	
	@Override protected double splitMethod(double[][] loc, boolean cat){
		return SplittingFunctions.binaryMedian(loc,cat)[1];
	}
	
	@Override protected double[] splitMethod2(double[][] loc, boolean cat, int num_class){
		return SplittingFunctions.gini(loc,cat,num_class,1);
	}
}