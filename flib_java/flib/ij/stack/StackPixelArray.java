package flib.ij.stack;

import ij.ImagePlus;
import ij.ImageStack;
import flib.math.VectorAccess;
import flib.math.VectorConv;

public class StackPixelArray {
	
	public static double[] StackPixelArray(ImagePlus imp){
		int n = imp.getNSlices();
		int w= imp.getWidth();
		int h= imp.getHeight();
		ImageStack stack = imp.getImageStack();
		double[] pixels = new double[w*h*n];
		for (int i=1; i<n+1; i++){
			VectorAccess.write(pixels,VectorConv.float2double((float[])stack.getProcessor(i).convertToFloat().getPixels()),(i-1)*w*h);
		}
		return pixels;
	}
}