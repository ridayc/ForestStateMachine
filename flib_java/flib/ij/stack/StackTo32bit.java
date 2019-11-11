package flib.ij.stack;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorConv;

public class StackTo32bit {
	static public ImagePlus convert(final ImagePlus imp){
		int w = imp.getWidth();
		int h = imp.getHeight();
		double min = imp.getProcessor().getMin();
		double max = imp.getProcessor().getMax();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			stack2.addSlice("None", new FloatProcessor(w,h,x));
		}
		ImagePlus imp2 = new ImagePlus("Stack converted to 32bit",stack2);
		imp2.getProcessor().setMinAndMax(min,max);
		return imp2;
	}
}