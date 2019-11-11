package flib.ij.featureextraction;

import java.io.File;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.featureextraction.FolderNames;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.fftfunctions.Convolution;
import flib.complexkernels.Gaussian;
import flib.algorithms.regions.ScaleRegions;
import flib.algorithms.regions.RegionFunctions;
import flib.algorithms.AssignToRandomNeighbor;
import flib.algorithms.SeededWatershed;
import ij.io.DirectoryChooser;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class ScaleWatershedRegions {
	// subfolder names for various purposes
	static String NORMALIZEDSCALES = FolderNames.NORMALIZEDSCALES;
	static String WATERSHED = FolderNames.WATERSHED;
	static String DISTANCE = FolderNames.DISTANCE;
	static String ANGLE = FolderNames.ANGLE;
	// the following watershed will be based on a scale decomposition
	// location: the base folder which contains the normalized scales folder
	public static void run(final String location, String orig, int type){
		String targetdir;
		// check if we need to let the user choose a directory
		// containing the images to be transformed
		if (location.equals("")){
			targetdir = (new DirectoryChooser("Choose a folder")).getDirectory();
			if (targetdir.isEmpty()){
				IJ.log("User canceled the ScaleWatershedRegions dialog! Bye!");
				return;
			}
		}
		// or if the user has provided an explicit target directory
		else {
			targetdir = new String(location);
		}
		String normscale = orig;
		// a normalized scales directory needs to exist already before hand
		if (orig.equals("")){
			normscale = targetdir+File.separator+NORMALIZEDSCALES;
		}
		// find all fiji readable files inside the normalized scales directory
		File[] directoryListing = (new File(normscale)).listFiles();
		String wshed = targetdir+File.separator+WATERSHED;
		String keyd = wshed+File.separator+DISTANCE;
		String keya = wshed+File.separator+ANGLE;
		// create the directory for the watershed scale images if it doesn't exist
		new File(wshed).mkdirs();
		new File(keyd).mkdirs();
		new File(keya).mkdirs();
		for (int i=0; i<directoryListing.length; i++){
			String name = directoryListing[i].getAbsolutePath();
			if (TypeReader.isTiff(name)){
				// this file should already be a .tif file
				// if the same image doesn't already exist in the
				// watershed directory create it
				String target_name = wshed+File.separator+directoryListing[i].getName();
				if (!(new File(target_name)).exists()){
					scaleWatershed(name,wshed,keyd,keya,type);
				}
			}
		}
	}
	
	// alternative run functions
	public static void run(){
		run("",0);
	}
	
	public static void run(String location){
		run(location,0);
	}
	
	public static void run(String location, int type){
		run(location,"",type);
	}
	
	synchronized private static void minmax(double[] mm0, double[] mm){
		if (mm0[0]>mm[0]){
			mm[0] = mm0[0];
		}
		if (mm0[1]<mm[1]){
			mm[1] = mm0[1];
		}
	}
	
	synchronized private static void vectorSum(double[] x, double[] y){
		VectorFun.addi(x,y);
	}
	
	// filename: scales image file name
	// wshed: directory where the watershed regions image will be stored
	// keyd: distances of all region points to their local minima/maxima
	// at the corresponding scale
	// keya: same as for keyd, just for the angle relationship
	private static void scaleWatershed(final String filename, final String wshed, final String keyd, final String keya, final int type){
		// open the original image file
		ImagePlus imp = IJ.openImage(filename);
		// image properties
		final int w = imp.getWidth();
		final int h = imp.getHeight();
		int n = imp.getNSlices();
		final ImageStack stack = imp.getImageStack();
		// prepare a stack for the watershed image
		final ImageStack stack2 = new ImageStack(w,h,n);
		// prepare a stack for the key distances
		final ImageStack stack3 = new ImageStack(w,h,n);
		// prepare a stack for the key angles
		final ImageStack stack4 = new ImageStack(w,h,n);
		// maxima and minima for the new stacks
		final double[] mm2 = new double[2];
		mm2[0] = Double.MIN_VALUE;
		mm2[1] = Double.MAX_VALUE;
		final double[] mm3 = new double[2];
		mm3[0] = Double.MIN_VALUE;
		mm3[1] = Double.MAX_VALUE;
		final double[] mm4 = new double[2];
		mm4[0] = Double.MIN_VALUE;
		mm4[1] = Double.MAX_VALUE;
		final int[][] keys = new int[n][];
		int NUM_CORES = Runtime.getRuntime().availableProcessors();
		ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
		// go through all scales/images
		try {
			for (int i=1; i<n+1; i++){
				final int i2 = i;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						try {
							double[] x = VectorConv.float2double((float[])(stack.getPixels(i2)));
							int[][] reg = new int[1][0];
							// inverted watershed
							if (type==1){
								for (int j=0; j<x.length; j++){
									x[j] = -x[j];
								}
							}
							// two-sided watershed
							if (type==0){
								ScaleRegions sc = new ScaleRegions(w,h,x);
								x = sc.getLabels2();
								keys[i2-1] = sc.getKeys();
								reg = sc.getRegions();
							}
							// watershed with minimal smoothing
							else if (type!=0){
								Convolution conv = new Convolution(w,h);
								Gaussian G = new Gaussian(new int[]{w,h},0.25);
								int[] seeds = SeededWatershed.connectedMinima2(w,h,x,8);
								double[] y = conv.convolveComplexKernel(x,G.getKernelr());
								x = (new AssignToRandomNeighbor(w,h,VectorConv.int2double((new SeededWatershed(w,h,y,8,seeds)).getRegionNumber()),8)).getImage();
								VectorFun.addi(x,-1);
								reg = RegionFunctions.getRegions(x);
								keys[i2-1] = RegionFunctions.getKeys(reg,VectorFun.abs(y));
							}							
							double[][] z = RegionFunctions.keyDistances(w,h,reg,keys[i2-1]);
							stack2.setProcessor(new FloatProcessor(w,h,x),i2);
							stack3.setProcessor(new FloatProcessor(w,h,z[0]),i2);
							stack4.setProcessor(new FloatProcessor(w,h,z[1]),i2);
							double[] mm02 = new double[2];
							mm02[0] = VectorFun.max(x)[0];
							mm02[1] = VectorFun.min(x)[0];
							double[] mm03 = new double[2];
							mm03[0] = VectorFun.max(z[0])[0];
							mm03[1] = VectorFun.min(z[0])[0];
							double[] mm04 = new double[2];
							mm04[0] = VectorFun.max(z[1])[0];
							mm04[1] = VectorFun.min(z[1])[0];
							minmax(mm02,mm2);
							minmax(mm03,mm3);
							minmax(mm04,mm4);
						}
						catch (Throwable t){
							System.out.println("Problem with the watershed evaluation");
							t.printStackTrace();
						}
					}
				});
			}
		}
		finally {
			exec.shutdown();
		}
		while(!exec.isTerminated()){
			// wait
		}
		// create the ImagePlus containers
		ImagePlus imp2 = new ImagePlus("ScaleWatershed",stack2);
		imp2.getProcessor().setMinAndMax(mm2[1],mm2[0]);
		ImagePlus imp3 = new ImagePlus("KeyDistances",stack3);
		imp3.getProcessor().setMinAndMax(mm3[1],mm3[0]);
		ImagePlus imp4 = new ImagePlus("KeyAngles",stack4);
		imp4.getProcessor().setMinAndMax(mm4[1],mm4[0]);
		// save the ImagePlus container
		String name = (new File(filename)).getName();
		IJ.saveAsTiff(imp2,wshed+File.separator+name);
		IJ.saveAsTiff(imp3,keyd+File.separator+name);
		IJ.saveAsTiff(imp4,keya+File.separator+name);
		String name2 = TypeReader.imageBase((new File(filename)).getName());
		ReadWrite.writeObject(wshed+File.separator+name2+".key",keys);
	}
}