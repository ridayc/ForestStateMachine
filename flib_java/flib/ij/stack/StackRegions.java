package flib.ij.stack;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.algorithms.regions.RegionPartitioning;

public class StackRegions {
	
	public static ImagePlus stackRegions(final ImagePlus imp){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			stack2.addSlice("None", new FloatProcessor(w,h,(new RegionPartitioning(w,h,x)).getImage()));
		}
		return new ImagePlus("Regions",stack2);
	}
}