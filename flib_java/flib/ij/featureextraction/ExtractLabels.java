package flib.ij.featureextraction;

import java.io.File;
import java.util.TreeSet;
import java.util.Iterator;
import flib.math.SortPair2;
import flib.math.VectorConv;
import flib.io.TypeReader;
import flib.ij.stack.StackTo32bit;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class ExtractLabels {
	public static TreeSet<SortPair2> labelConversion(final String foldername){
		// get all files in the target directory
		File[] directoryListing = (new File(foldername)).listFiles();
		int[] file_number = TypeReader.fileNumber(directoryListing);
		TreeSet<SortPair2> labels = new TreeSet<SortPair2>();
		for (int i=0; i<file_number.length; i++){
			String name = directoryListing[file_number[i]].getAbsolutePath();
			courseImage(name,labels);
		}
		Iterator<SortPair2> it = labels.iterator();
		TreeSet<SortPair2> labels2 = new TreeSet<SortPair2>();
		int count = 0;
		while (it.hasNext()){
			SortPair2 sp = it.next();
			if (count!=labels.size()-1){
				labels2.add(new SortPair2(sp.getValue(),count));
				count++;
			}
		}
		return labels2;
	}
	
	private static void courseImage(final String filename, TreeSet<SortPair2> labels){
		// open the original image file and convert it (if neccessary)
		ImagePlus imp = StackTo32bit.convert(IJ.openImage(filename));
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		// go through all images in the original (if there are more than one.... which we just assume there are not)
		for (int i=1; i<=n; i++){
			double[] x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			for (int j=0; j<x.length; j++){
				SortPair2 sp = new SortPair2(x[j],1);
				if (labels.contains(sp)){
					sp = labels.floor(sp);
					sp.setOriginalIndex(sp.getOriginalIndex()+1);
				}
				else {
					labels.add(sp);
				}
			}
		}
	}
}