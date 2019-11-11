package flib.ij.featureextraction;

import java.io.File;
import flib.io.TypeReader;
import flib.ij.io.ImageReader;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class SetThreshold {

	public static void run(String orig, String targetdir, final double[] threshold){
		// find all fiji readable files inside the original directory
		File[] directoryListing = (new File(orig)).listFiles();
		new File(targetdir).mkdirs();
		for (int i=0; i<directoryListing.length; i++){
			String name = directoryListing[i].getAbsolutePath();
			// find all images in this directory
			if (TypeReader.isImage(name)){
				String name2 = targetdir+File.separator+TypeReader.imageBase(directoryListing[i].getName())+".tif";
				if (!(new File(name2)).exists()){
					ImagePlus imp = IJ.openImage(name);
					int w = imp.getWidth();
					int h = imp.getHeight();
					int n = imp.getNSlices();
					// create a stack for storage
					ImageStack stack = new ImageStack(w,h);
					// go through all subimages
					for (int j=0; j<n; j++){
						// here things will start to depend on the type of 
						// investigation at hand
						// we assume there is only one image to be investigated at this point
						float[] im = (float[])(imp.getStack().getPixels(j+1));
						for (int k=0; k<threshold.length; k++){
							float[] im2 = new float[w*h];
							for (int l=0; l<w*h; l++){
								if (im[l]>threshold[k]){
									im2[l] = 1;
								}
							}
							stack.addSlice("None",new FloatProcessor(w,h,im2));
						}
					}
					ImagePlus imp2 = new ImagePlus("Thresholded",stack);
					imp.getProcessor().setMinAndMax(0,1);
					IJ.saveAsTiff(imp2,name2);
				}
			}
		}
	}
	
	public static void thresholdSum(String orig, String normscale, String targetdir, final boolean[] scales){
		// find all fiji readable files inside the original directory
		File[] directoryListing = (new File(orig)).listFiles();
		new File(targetdir).mkdirs();
		int counter = 0;
		for (int i=0; i<directoryListing.length; i++){
			String name = directoryListing[i].getAbsolutePath();
			// find all images in this directory
			if (TypeReader.isImage(name)){
				String name2 = targetdir+File.separator+TypeReader.imageBase(directoryListing[i].getName())+".tif";
				String name3 = normscale+File.separator+TypeReader.imageBase(directoryListing[i].getName())+".tif";
				if (!(new File(name2)).exists()&&(new File(name3)).exists()){
					ImagePlus imp = IJ.openImage(name);
					int w = imp.getWidth();
					int h = imp.getHeight();
					int n = imp.getNSlices();
					// create a stack for storage
					ImageStack stack = new ImageStack(w,h);
					int m = ImageReader.getTiffSize(name3)[0]/n;
					// go through all subimages
					for (int j=0; j<n; j++){
						// here things will start to depend on the type of 
						// investigation at hand
						// we assume there is only one image to be investigated at this point
						float[] im = (float[])(imp.getStack().getPixels(j+1));
						float[] im2 = new float[w*h];
						int counter2 = 0;
						for (int l=0; l<m; l++){
							if (scales[l]){
								float[] im3 = ImageReader.tiffLayerArray(name3,j*m+l);
								for (int k=0; k<w*h; k++){
									if (im3[k]>0){
										im2[k]++;
									}
								}
								counter2++;
							}
						}
						// make sure m>1
						for (int k=0; k<w*h; k++){
							im2[k]/=counter2;
						}
						stack.addSlice("None",new FloatProcessor(w,h,im2));
					}
					ImagePlus imp2 = new ImagePlus("Thresholded",stack);
					imp.getProcessor().setMinAndMax(0,1);
					IJ.saveAsTiff(imp2,name2);
				}
				counter++;
			}
		}
	}
}