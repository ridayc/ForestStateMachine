package flib.algorithms.randomforest.splitfunctions;

import flib.algorithms.randomforest.splitfunctions.ClusterSplit;
import flib.algorithms.randomforest.splitfunctions.SplittingFunctions;

public class GainClusterSplit extends ClusterSplit {
	@Override protected double splitMethod(double[][] loc, boolean cat){
		return SplittingFunctions.binaryInfGain(loc,cat)[1];
	}
	
	@Override protected double[] splitMethod2(double[][] loc, boolean cat, int num_class){
		return SplittingFunctions.infGain(loc,cat,num_class,1);
	}
}