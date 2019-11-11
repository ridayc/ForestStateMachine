package flib.algorithms.randomforest.splitfunctions;

import flib.algorithms.randomforest.splitfunctions.ClassicSplit;
import flib.algorithms.randomforest.splitfunctions.SplittingFunctions;

public class GainSplit extends ClassicSplit {
	@Override protected double[] splitMethod(double[][] loc, boolean cat, int num_class, double[] parameters){
		return SplittingFunctions.infGain(loc,cat,num_class,parameters[10]);
	}
}