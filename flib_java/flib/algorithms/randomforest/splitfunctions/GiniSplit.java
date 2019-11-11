package flib.algorithms.randomforest.splitfunctions;

import flib.algorithms.randomforest.splitfunctions.ClassicSplit;
import flib.algorithms.randomforest.splitfunctions.SplittingFunctions;

public class GiniSplit extends ClassicSplit {
	@Override protected double[] splitMethod(double[][] loc, boolean cat, int num_class, double[] parameters){
		return SplittingFunctions.gini(loc,cat,num_class,parameters[10]);
	}
}