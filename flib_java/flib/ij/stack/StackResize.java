package flib.ij.stack;

import ij.ImagePlus;
import ij.ImageStack;

public class StackResize {
	private ImagePlus imp;
	
	public StackResize(ImagePlus imp, int w, int h){
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		for (int i=1; i<n+1; i++){
			stack2.addSlice("None",stack.getProcessor(i).resize(w,h));
		}
		this.imp = new ImagePlus("resized", stack2);
	}
	
	public ImagePlus getImage(){
		return this.imp;
	}
}