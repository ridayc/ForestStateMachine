package flib.ij.featureextraction;

import java.io.File;
import java.lang.Math;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.featureextraction.FolderNames;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.algorithms.BLabel;
import flib.algorithms.regions.RegionFunctions;
import ij.io.DirectoryChooser;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class ConnectedRegions {
	
	// subfolder names for various purposes
	static String BLABEL = FolderNames.BLABEL;
	static String DISTANCE = FolderNames.DISTANCE;
	static String ANGLE = FolderNames.ANGLE;
	static String PCA = FolderNames.PCAANGLE;
	// the following watershed will be based on a scale decomposition
	// location: the base folder which contains the normalized scales folder
	public static void run(final String location){
		String targetdir;
		// check if we need to let the user choose a directory
		// containing the images to be transformed
		if (location.equals("")){
			targetdir = (new DirectoryChooser("Choose a folder")).getDirectory();
			if (targetdir.isEmpty()){
				IJ.log("User canceled the ConnectedRegions dialog! Bye!");
				return;
			}
		}
		// or if the user has provided an explicit target directory
		else {
			targetdir = new String(location);
		}
		// find all fiji readable files inside the normalized scales directory
		File[] directoryListing = (new File(targetdir)).listFiles();
		String blab = targetdir+File.separator+BLABEL;
		String keyd = targetdir+File.separator+DISTANCE;
		String keya = targetdir+File.separator+ANGLE;
		String keyp = targetdir+File.separator+PCA;
		// create the directory for the watershed scale images if it doesn't exist
		new File(blab).mkdirs();
		new File(keyd).mkdirs();
		new File(keya).mkdirs();
		new File(keyp).mkdirs();
		for (int i=0; i<directoryListing.length; i++){
			String name = directoryListing[i].getAbsolutePath();
			if (TypeReader.isTiff(name)){
				// this file should already be a .tif file
				// if the same image doesn't already exist in the
				// watershed directory create it
				String target_name = blab+File.separator+directoryListing[i].getName();
				if (!(new File(target_name)).exists()){
					bLabel(name,blab,keyd,keya,keyp);
				}
			}
		}
	}
	
	// alternative run functions
	public static void run(){
		run("");
	}
	
	// filename: scales image file name
	// wshed: directory where the watershed regions image will be stored
	// keyd: distances of all region points to their local minima/maxima
	// at the corresponding scale
	// keya: same as for keyd, just for the angle relationship
	private static void bLabel(final String filename, final String blab, final String keyd, final String keya, final String keyp){
		// open the original image file
		ImagePlus imp = IJ.openImage(filename);
		// image properties
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		// prepare a stack for the regions image
		ImageStack stack2 = new ImageStack(w,h);
		// prepare a stack for the key distances
		ImageStack stack3 = new ImageStack(w,h);
		// prepare a stack for the key angles
		ImageStack stack4 = new ImageStack(w,h);
		// prepare a stack for the pca angles
		ImageStack stack5 = new ImageStack(w,h);
		// maxima and minima for the new stacks
		double min2 = Double.MAX_VALUE;
		double max2 = Double.MIN_VALUE;
		double min3 = Double.MAX_VALUE;
		double max3 = Double.MIN_VALUE;
		double min4 = Double.MAX_VALUE;
		double max4 = Double.MIN_VALUE;
		double min5 = Double.MAX_VALUE;
		double max5 = Double.MIN_VALUE;
		// in this case the keys are the region centroids
		int[][] keys = new int[n][];
		// go through all scales or images
		for (int i=1; i<n+1; i++){
			double[] x = VectorConv.float2double((float[])(stack.getPixels(i)));
			double t;
			// we use four way connectivity to prevent some region merging...
			// shouldn't matter all that much
			BLabel bl = new BLabel(w,h,VectorFun.add(x,1),4);
			x = bl.getBlobNumber();
			int[][] reg = RegionFunctions.getRegions(x);
			keys[i-1] = RegionFunctions.regionCentroid(w,h,reg);
			double[][] z = RegionFunctions.keyDistances(w,h,reg,keys[i-1]);
			stack2.addSlice("None", new FloatProcessor(w,h,x));
			t = VectorFun.min(x)[0];
			if (t<min2){
				min2 = t;
			}
			t = VectorFun.max(x)[0];
			if (t>max2){
				max2 = t;
			}
			stack3.addSlice("None", new FloatProcessor(w,h,z[0]));
			t = VectorFun.min(z[0])[0];
			if (t<min3){
				min3 = t;
			}
			t = VectorFun.max(z[0])[0];
			if (t>max3){
				max3 = t;
			}
			stack4.addSlice("None", new FloatProcessor(w,h,z[2]));
			t = VectorFun.min(z[2])[0];
			if (t<min4){
				min4 = t;
			}
			t = VectorFun.max(z[2])[0];
			if (t>max4){
				max4 = t;
			}
			double[] temp = new double[z[2].length];
			for (int j=0; j<z[2].length; j++){
				temp[j] = z[2][j]-z[1][j];
				while (temp[j]<0){
					temp[j]+=Math.PI;
				}
				while (temp[j]>=Math.PI){
					temp[j]-=Math.PI;
				}
			}
			stack5.addSlice("None", new FloatProcessor(w,h,temp));
			t = VectorFun.min(temp)[0];
			if (t<min5){
				min5 = t;
			}
			t = VectorFun.max(temp)[0];
			if (t>max5){
				max5 = t;
			}
		}
		// create the ImagePlus containers
		ImagePlus imp2 = new ImagePlus("RegionLabels",stack2);
		imp2.getProcessor().setMinAndMax(min2,max2);
		ImagePlus imp3 = new ImagePlus("KeyDistances",stack3);
		imp3.getProcessor().setMinAndMax(min3,max3);
		ImagePlus imp4 = new ImagePlus("KeyAngles",stack4);
		imp4.getProcessor().setMinAndMax(min4,max4);
		ImagePlus imp5 = new ImagePlus("KeyPCA",stack5);
		imp5.getProcessor().setMinAndMax(min5,max5);
		// save the ImagePlus container
		String name = (new File(filename)).getName();
		IJ.saveAsTiff(imp2,blab+File.separator+name);
		IJ.saveAsTiff(imp3,keyd+File.separator+name);
		IJ.saveAsTiff(imp4,keya+File.separator+name);
		IJ.saveAsTiff(imp5,keyp+File.separator+name);
		String name2 = TypeReader.imageBase((new File(filename)).getName());
		ReadWrite.writeObject(blab+File.separator+name2+".key",keys);
	}
}