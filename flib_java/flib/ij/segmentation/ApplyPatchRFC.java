package flib.ij.segmentation;

import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.clustering.RFC;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.algorithms.randomforest.MissingValues;

public class ApplyPatchRFC {
	
	public static int[] getIndices(final RandomForest RF, final RFC rfc, int w, int h, double r, double rad, double fill, int numit,final int[] ind, final double[]... im){
		MissingValues MV = new MissingValues(RF);
		int[] indices = new int[ind.length];
		double[] point;
		int[] missingloc;
		int[] p = new int[1];
		double phi = 1;
		double b = 1;
		double m = 1;
		int arms = 4;
		int[][] shape = NeighborhoodSample.spiralCoord(phi,r,b,m,rad,arms);
		for (int i=0; i<indices.length; i++){
			// generate the neighborhood of each pixel
			p[0] = ind[i];
			point = (NeighborhoodSample.sample2d(p,w,h,shape,0,fill,im))[0];
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
	
	public static int[] getIndices(final RandomForest RF, final RFC rfc, int w, int h, double r, double rad, double fill, int numit, final double[]... im){
		int[] ind = new int[im[0].length];
		for (int i=0; i<ind.length; i++){
			ind[i] = i;
		}
		return getIndices(RF,rfc,w,h,r,rad,fill,numit,ind,im);
	}	
}