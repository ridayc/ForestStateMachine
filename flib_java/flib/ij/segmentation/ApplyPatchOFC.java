package flib.ij.segmentation;

import flib.algorithms.randomforest.OrthoForest;
import flib.algorithms.clustering.RFC;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.algorithms.randomforest.MissingValuesO;

public class ApplyPatchOFC {
	
	public static int[] getIndices(final OrthoForest RF, final RFC rfc, int w, int h, int s, double fill, int numit, final double[]... im){
		MissingValuesO MV = new MissingValuesO(RF);
		int[] indices = new int[im.length];
		double[] point;
		int[] missingloc;
		int[] p = new int[1];
		int[][] shape = NeighborhoodSample.circleCoord(w,h,s);
		for (int i=0; i<indices.length; i++){
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
				indices[i] = rfc.assignCluster(RF.getLeafIndices(MV.completePoint(point,missingloc,0,numit)));
			}
			else {
				indices[i] = rfc.assignCluster(RF.getLeafIndices(point));
			}
		}
		return indices;
	}
	
	// redundancy? Why yes of course...
	public static int[] getIndices(final OrthoForest RF, final RFC rfc, int w, int h, int s, double fill, int numit, final double[] im){
		MissingValuesO MV = new MissingValuesO(RF);
		int[] indices = new int[im.length];
		double[] point;
		int[] missingloc;
		int[] p = new int[1];
		int[][] shape = NeighborhoodSample.circleCoord(w,h,s);
		for (int i=0; i<indices.length; i++){
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
				indices[i] = rfc.assignCluster(RF.getLeafIndices(MV.completePoint(point,missingloc,0,numit)));
			}
			else {
				indices[i] = rfc.assignCluster(RF.getLeafIndices(point));
			}
		}
		return indices;
	}
}