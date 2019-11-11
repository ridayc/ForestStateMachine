package flib.ij.stack;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.algorithms.SeededWatershed;
import flib.algorithms.DistanceTransform;

public class StackWatershed {
	
	public static ImagePlus stackWatershed(final ImagePlus imp, int connectivity, int direction){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		SeededWatershed WS;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			if (direction==1){
				WS = new SeededWatershed(w,h,x,connectivity);
			}
			else {
				WS = new SeededWatershed(w,h,VectorFun.mult(x,-1),connectivity);
			}
			stack2.addSlice("None", new FloatProcessor(w,h,VectorConv.int2double(WS.getRegionNumber())));
		}
		return new ImagePlus("Watershed",stack2);
	}
	
	public static ImagePlus stackDistanceTransform(final ImagePlus imp, int type){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			x = DistanceTransform.dt2d(w,h,x,type);
			stack2.addSlice("None", new FloatProcessor(w,h,x));
		}
		return new ImagePlus("Distance Transform",stack2);
	}
}