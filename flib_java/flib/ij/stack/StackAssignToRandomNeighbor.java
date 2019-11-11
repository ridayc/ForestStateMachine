package flib.ij.stack;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.algorithms.AssignToRandomNeighbor;

public class StackAssignToRandomNeighbor {
	
	public static ImagePlus assign(final ImagePlus imp, int connectivity){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		AssignToRandomNeighbor AtRN;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			AtRN = new AssignToRandomNeighbor(w,h,x,connectivity);
			stack2.addSlice("None", new FloatProcessor(w,h,AtRN.getImage()));
		}
		return new ImagePlus("Reassigned",stack2);
	}
}