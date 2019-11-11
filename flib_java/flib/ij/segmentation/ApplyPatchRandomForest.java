package flib.ij.segmentation;

import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.algorithms.randomforest.MissingValues;
import flib.math.VectorAccess;

public class ApplyPatchRandomForest {
	
	public static double[][] getVotes(final RandomForest RF, int w, int h, int n, double fill, final double[]... im){
		MissingValues MV = new MissingValues(RF);
		double[][] votes = new double[im[0].length][RF.getNumClasses()];
		double[] point;
		int[] missingloc;
		int[] p = new int[1];
		int[][] shape = NeighborhoodSample.circleCoord(n);
		for (int i=0; i<votes.length; i++){
			// generate the neighborhood of each pixel
			p[0] = i;
			point = (NeighborhoodSample.sample2d(p, w, h, shape,0,fill,im))[0];
			int count = 0;
			for (int j=0; j<point.length; j++){
				if (point[j]==fill){
					count++;
				}
			}
			if (count>0){
				missingloc = new int[count];
				count = 0;
				for (int j=0; j<point.length; j++){
					if (point[j]==fill){
						missingloc[count] = j;
						count++;
					}
				}
				votes[i] = MV.getVotes(point,missingloc);
			}
			else {
				votes[i] = MV.getForest().applyForest(point);
			}
		}
		return VectorAccess.flip(votes);
	}
	
	// once again... super redundant :/
	public static double[][] getVotes(final RandomForest RF, int w, int h, int n, double fill, final double[] im){
		MissingValues MV = new MissingValues(RF);
		double[][] votes = new double[im.length][RF.getNumClasses()];
		double[] point;
		int[] missingloc;
		int[] p = new int[1];
		int[][] shape = NeighborhoodSample.circleCoord(n);
		for (int i=0; i<votes.length; i++){
			// generate the neighborhood of each pixel
			p[0] = i;
			point = (NeighborhoodSample.sample2d(p, w, h, shape,0,fill,im))[0];
			int count = 0;
			for (int j=0; j<point.length; j++){
				if (point[j]==fill){
					count++;
				}
			}
			if (count>0){
				missingloc = new int[count];
				count = 0;
				for (int j=0; j<point.length; j++){
					if (point[j]==fill){
						missingloc[count] = j;
						count++;
					}
				}
				votes[i] = MV.getVotes(point,missingloc);
			}
			else {
				votes[i] = MV.getForest().applyForest(point);
			}
		}
		return VectorAccess.flip(votes);
	}
}