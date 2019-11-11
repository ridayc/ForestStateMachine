package flib.algorithms.regions;

import flib.math.BV;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.algorithms.SeededWatershed;
import flib.algorithms.BLabel;
import flib.algorithms.AssignToRandomNeighbor;
import flib.algorithms.regions.RegionFunctions;

public class ScaleRegions {
	private int[] key_points = null;
	private int[][] region_pixels = null;
	private int[] pixel_labels = null;
	private double[] im = null;
	
	public ScaleRegions(int w, int h, final double[] x){
		// obtain all pixels from the maximum based watershed with positive
		// pixel values
		// after that label all connected regions
		// then reassign watershed pixels to a random neighboring region
		int[] regp = VectorConv.double2int((new AssignToRandomNeighbor(w,h,VectorFun.add((new BLabel(w,h,VectorConv.bool2double(BV.gt(VectorFun.mult(VectorConv.int2double((new SeededWatershed(w,h,VectorFun.mult(x,-1),8)).getRegionNumber()),x),0)),8)).getBlobNumber(),1),8)).getImage());
		int n = VectorFun.max(regp)[0];
		// do the same for values based on the minimum based watershed with
		// negative pixelvalues
		int[] regn = VectorConv.double2int((new AssignToRandomNeighbor(w,h,VectorFun.add((new BLabel(w,h,VectorConv.bool2double(BV.gt(VectorFun.mult(VectorConv.int2double((new SeededWatershed(w,h,x,8)).getRegionNumber()),VectorFun.mult(x,-1)),0)),8)).getBlobNumber(),1),8)).getImage());
		pixel_labels = new int[w*h];
		// collect the labels from both disjoint sets
		for (int i=0; i<w*h; i++){
			if (x[i]>0){
				pixel_labels[i] = regp[i]-1;
			}
			else {
				pixel_labels[i] = n+regn[i]-1;
			}
		}
		im = x.clone();
	}
	
	public int[] getLabels(){
		return pixel_labels.clone();
	}
	
	public double[] getLabels2(){
		return VectorConv.int2double(pixel_labels);
	}
	
	public int[][] getRegions(){
		if (region_pixels==null){
			region_pixels = RegionFunctions.getRegions(VectorConv.int2double(pixel_labels));
		}
		int[][] temp = new int[region_pixels.length][];
		for (int i=0; i< temp.length; i++){
			temp[i] = region_pixels[i].clone();
		}
		return temp;
	}
	
	public int[] getKeys(){
		if (key_points==null){
			if (region_pixels==null){
				region_pixels = RegionFunctions.getRegions(VectorConv.int2double(pixel_labels));
			}
			key_points = RegionFunctions.getKeys(region_pixels,im);
		}
		return key_points.clone();
	}
}