package flib.ij.featureextraction;

import java.io.File;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.io.ImageReader;
import flib.ij.featureextraction.FolderNames;
import flib.ij.featureextraction.FileInterpreter;
import flib.algorithms.SeededWatershed;
import flib.algorithms.BLabel;
import flib.algorithms.AssignToRandomNeighbor;
import flib.algorithms.regions.RegionFunctions;
import flib.fftfunctions.Convolution;
import flib.complexkernels.Gaussian;
import flib.math.VectorFun;
import flib.math.VectorConv;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class DiscreteWatershed {
	// subfolder names for various purposes
	static String BLABEL = FolderNames.BLABEL;
	static String DISTANCE = FolderNames.DISTANCE;
	static String ANGLE = FolderNames.ANGLE;
	
	public static void run(String orig, String targetdir,double threshold){
		// find all fiji readable files inside the origin directory
		FileInterpreter FI = new FileInterpreter(orig);
		new File(targetdir).mkdirs();
		String blab = targetdir+File.separator+BLABEL;
		String keyd = targetdir+File.separator+DISTANCE;
		String keya = targetdir+File.separator+ANGLE;
		new File(blab).mkdirs();
		new File(keyd).mkdirs();
		new File(keya).mkdirs();
		// go through all images in the origin folder
		for (int i=0; i<FI.getBases().length; i++){
			String name = blab+File.separator+FI.getBases()[i]+TypeReader.TIFF[0];
			if(!(new File(name)).exists()){
				// get the image properties
				int[] size = ImageReader.getTiffSize(FI.getNames()[i]);
				// prepare the Gaussian convolution
				Convolution conv = new Convolution(size[1],size[2]);
				Gaussian G = new Gaussian(new int[]{size[1],size[2]},0.5);
				// prepare the image stack for storage
				ImageStack stack2 = new ImageStack(size[1],size[2]);
				ImageStack stack3 = new ImageStack(size[1],size[2]);
				ImageStack stack4 = new ImageStack(size[1],size[2]);
				// maxima and minima for the new stacks
				double[] max2 = {Double.MAX_VALUE,Double.MIN_VALUE};
				double[] max3 ={Double.MAX_VALUE,Double.MIN_VALUE};
				double[] max4 = {Double.MAX_VALUE,Double.MIN_VALUE};
				// in this case the keys are the region centroids
				int[][] keys = new int[size[0]*4][];
				// go through all subimages
				for (int j=0; j<size[0]; j++){
					// read the image... which should be a tiff file
					double[] im = VectorConv.float2double(ImageReader.tiffLayerArray(FI.getNames()[i],j));
					// threshold the image
					double[] binim = new double[im.length];
					for (int k=0; k<im.length; k++){
						if(im[k]>threshold){
							binim[k] = 2;
						}
						else {
							binim[k] = 1;
						}
					}
					// get the connected minima of the discrete image
					int[] seeds = SeededWatershed.connectedMinima2(size[1],size[2],im,8);
					// create a smoothed version of the image
					double[] im2 = conv.convolveComplexKernel(im,G.getKernelr());
					// apply the watershed to this image
					double[] temp = VectorConv.int2double((new SeededWatershed(size[1],size[2],im2,8,seeds)).getRegionNumber());
					for (int k=0; k<im2.length; k++){
						if (temp[k]==0){
							temp[k] = 0;
						}
						else {
							temp[k] = binim[k];
						}
						im[k] = -im[k];
						im2[k] = -im2[k];
					}
					updateArrays(stack2,stack3,stack4,max2,max3,max4,keys,temp,size,j*4);
					for (int k=0; k<im.length; k++){
						if(temp[k]>=1){
							temp[k] = 1;
						}
					}
					updateArrays(stack2,stack3,stack4,max2,max3,max4,keys,temp,size,j*4+1);
					// same procedure in the other direction
					// get the connected minima of the discrete image
					seeds = SeededWatershed.connectedMinima2(size[1],size[2],im,8);
					// apply the watershed to this image
					temp = VectorConv.int2double((new SeededWatershed(size[1],size[2],im2,8,seeds)).getRegionNumber());
					for (int k=0; k<im2.length; k++){
						if (temp[k]==0){
							temp[k] = 0;
						}
						else {
							temp[k] = binim[k];
						}
					}
					updateArrays(stack2,stack3,stack4,max2,max3,max4,keys,temp,size,j*4+2);
					for (int k=0; k<im.length; k++){
						if(temp[k]>=1){
							temp[k] = 1;
						}
					}
					updateArrays(stack2,stack3,stack4,max2,max3,max4,keys,temp,size,j*4+3);
					
				}
				// save the stacks as tifs
				// create the ImagePlus containers
				ImagePlus imp2 = new ImagePlus("RegionLabels",stack2);
				imp2.getProcessor().setMinAndMax(max2[0],max2[1]);
				ImagePlus imp3 = new ImagePlus("KeyDistances",stack3);
				imp3.getProcessor().setMinAndMax(max3[0],max3[1]);
				ImagePlus imp4 = new ImagePlus("KeyAngles",stack4);
				imp4.getProcessor().setMinAndMax(max4[0],max4[1]);
				// save the ImagePlus container
				name = FI.getBases()[i]+TypeReader.TIFF[0];
				IJ.saveAsTiff(imp2,blab+File.separator+name);
				IJ.saveAsTiff(imp3,keyd+File.separator+name);
				IJ.saveAsTiff(imp4,keya+File.separator+name);
				ReadWrite.writeObject(blab+File.separator+FI.getBases()[i]+".key",keys);
			}
		}
	}
	
	private static void updateArrays(ImageStack stack2, ImageStack stack3, ImageStack stack4, double[] max2, double[] max3, double[] max4, int[][] keys, final double[] temp, final int[] size, int j){
		double[] x = (new BLabel(size[1],size[2],temp,8)).getBlobNumber();
		VectorFun.addi(x,1);
		x = (new AssignToRandomNeighbor(size[1],size[2],x,8)).getImage();
		VectorFun.addi(x,-1);
		// store the image
		stack2.addSlice(new FloatProcessor(size[1],size[2],x));
		double t = VectorFun.min(x)[0];
		if (t<max2[0]){
			max2[0] = t;
		}
		t = VectorFun.max(x)[0];
		if (t>max2[1]){
			max2[1] = t;
		}
		int[][] reg = RegionFunctions.getRegions(x);
		keys[j] = RegionFunctions.regionCentroid(size[1],size[2],reg);
		double[][] z = RegionFunctions.keyDistances(size[1],size[2],reg,keys[j]);
		stack3.addSlice("None", new FloatProcessor(size[1],size[2],z[0]));
		t = VectorFun.min(z[0])[0];
		if (t<max3[0]){
			max3[0] = t;
		}
		t = VectorFun.max(z[0])[0];
		if (t>max3[1]){
			max3[1] = t;
		}
		stack4.addSlice("None", new FloatProcessor(size[1],size[2],z[1]));
		t = VectorFun.min(z[1])[0];
		if (t<max4[0]){
			max4[0] = t;
		}
		t = VectorFun.max(z[1])[0];
		if (t>max4[1]){
			max4[1] = t;
		}
	}
}