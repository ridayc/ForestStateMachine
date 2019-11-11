package flib.algorithms.randomforest;

import flib.algorithm.randomforest.RandomForest;
import flib.algorithm.clustering.SpectralClustering;
import flib.math.VectorConv;
import flib.math.VectorFun;

public class ClusterForest {
	
	public static RandomForest SpectralCluster(final double[][] set, final double[] weights, final boolean[] categorical, final double[] dimweights, final double[] parameters, final double[] splitpurity, final SplitFunction G, int ntree){
		if (parameters[0]>2){
			return RandomForest(set,SpectralClustering.cluster(proximities,(int)parameters[0]),weights,RF.getCategorical(),RF.getDW(),RF.getParameters(),RF.getSplitPurity(),RF.getSplitFunction,RF.getNtree());
		}
		else {
			int[] ind = new int[set.length];
			SpectralClustering.cluster(proximities,ind);
			return RandomForest(set,ind,weights,RF.getCategorical(),RF.getDW(),RF.getParameters(),RF.getSplitPurity(),RF.getSplitFunction,RF.getNtree());
		}
	}
	
	public static RandomForest minCluster(final RandomForest RF, final double[][] set, final double[] weights, int nc){
		
}