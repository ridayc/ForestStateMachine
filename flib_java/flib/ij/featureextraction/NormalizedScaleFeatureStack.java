package flib.ij.featureextraction;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.util.Arrays;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.lang.Math;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.featureextraction.FolderNames;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.algorithms.images.ScaleAnalysis;
import flib.ij.stack.StackTo32bit;
import flib.ij.stack.StackOperations;
import ij.io.DirectoryChooser;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class NormalizedScaleFeatureStack {
	// folder names for relative storage paths
	static String NORMALIZEDSCALES = FolderNames.NORMALIZEDSCALES;
	static String CONTRASTED = FolderNames.CONTRASTED;
	// perform a normalized scale band pass filtering
	// location: folder where the files to be band pass filtered reside
	// parameters: parameters such as number of scales and angles, etc.
	// target: alternative storage folder with the same type of relative storage
	// structure
	public static void run(final String location, final double[] parameters, final String target){
		String targetdir, targetdir2;
		double[] param;
		// check if we need to let the user choose a directory
		// containing the images to be transformed
		if (location.equals("")){
			targetdir = (new DirectoryChooser("Choose a folder")).getDirectory();
			if (targetdir.isEmpty()){
				IJ.log("User canceled the NormalizedScaleFeatureStack dialog! Bye!");
				return;
			}
		}
		// or if the user has provided an explicit target directory
		else {
			targetdir = new String(location);
		}
		// check if an alternative target storage directory is neccessary
		if (!target.equals("")){
			targetdir2 = target;
		}
		else {
			targetdir2 = targetdir;
		}
		// find all fiji readable files inside the target directory
		File[] directoryListing = (new File(targetdir)).listFiles();
		String normscale = targetdir2+File.separator+NORMALIZEDSCALES;
		// create the directory for the normalized scale images if it doesn't exist
		new File(normscale).mkdirs();
		String contrasted = targetdir2+File.separator+CONTRASTED;
		// create the directory for the normalized scale images if it doesn't exist
		new File(contrasted).mkdirs();
		// check if a file with the scale parameters exists in this folder
		String config = normscale+File.separator+"config.ser";
		String configtxt = normscale+File.separator+"config.txt";
		// only overwrite if the config file does not exist
		if((new File(config)).exists()){
			// take the parameters from the existing configuration
			param = ((double[])ReadWrite.readObject(config)).clone();
			writeConfigTXT(configtxt,param);
			// go through all files in the directory
			for (int i=0; i<directoryListing.length; i++){
				String name = directoryListing[i].getAbsolutePath();
				if (TypeReader.isImage(name)){
					// if the same image doesn't already exist in the
					// normscale directory create it
					String target_name = normscale+File.separator+TypeReader.imageBase(directoryListing[i].getName())+TypeReader.TIFF[0];
					if (!(new File(target_name)).exists()){
						normalizedScales(name,targetdir2,param);
					}
				}
			}
			
		}
		else {
			param = parameters.clone();
			writeConfigTXT(configtxt,param);
			ReadWrite.writeObject(config,param);
			for (int i=0; i<directoryListing.length; i++){
				String name = directoryListing[i].getAbsolutePath();
				if (TypeReader.isImage(name)){
					// if the same image doesn't already exist in the
					// normscale directory create it
					normalizedScales(name,targetdir2,param);
				}
			}
		}
	}
	
	// run functions with default inputs
	public static void run(final String location, final double[] parameters){
		run(location,parameters,"");
	}
	
	public static void run(final double[] parameters, final String target){
		run("",parameters,target);
	}
	
	public static void run(final double[] parameters){
		run("",parameters,"");
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
	
	// the actual scales calculation
	// filename: name of the image file
	// paramters: scaling information
	private static void normalizedScales(String filename, String targetdir, final double[] parameters){
		// open the original image file and convert it (if neccessary)
		ImagePlus imp = StackTo32bit.convert(IJ.openImage(filename));
		final int w = imp.getWidth();
		final int h = imp.getHeight();
		final int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		// prepare a stack for the scales and for the contrasted image
		final ImageStack stack2 = new ImageStack(w,h,(int)parameters[0]*(int)parameters[5]*n);
		ImageStack stack3 = new ImageStack(w,h);
		// maxima and minima for the new stacks
		final double[] mm2 = new double[2];
		mm2[0] = Double.MIN_VALUE;
		mm2[1] = Double.MAX_VALUE;
		final double[] mm3 = new double[2];
		mm3[0] = Double.MIN_VALUE;
		mm3[1] = Double.MAX_VALUE;
		final double lambda = parameters[1];
		// go through all images in the original (if there are more than one)
		for (int i=1; i<n+1; i++){
			final int i2 = i;
			final double[] x = new double[w*h];
			final double[] z = VectorConv.float2double((float[])(stack.getPixels(i)));
			/*
			int NUM_CORES = Runtime.getRuntime().availableProcessors();
			ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
			try {
				// go through all scales
				for (int j=0; j<(int)parameters[0]; j++){
					final int j2 = j;
					for (int k=0; k<(int)parameters[5]; k++){
						final int k2 = k;
						exec.submit(new Runnable() {
							@Override
							public void run(){
								try {
									// normalized scale
									double[] y = ScaleAnalysis.normalize(ScaleAnalysis.singleDecomposition(z,w,h,lambda*Math.pow(parameters[3],j2),parameters[2],(int)parameters[5],k2,parameters[6]),w,h,parameters[4]*lambda*Math.pow(parameters[3],j2));
									// get rid of nan values
									for (int l=0; l<y.length; l++){
										// we need to check that the pixel values
										// are regular double values
										if (Double.isNaN(y[l])){
											y[l] = 0;
										}
									}
									stack2.setProcessor(new FloatProcessor(w,h,y),(i2-1)*(int)parameters[0]*(int)parameters[5]+j2*(int)parameters[5]+k2+1);
									// maxima/minima reevaluation
									double[] mm0 = new double[2];
									mm0[0] = VectorFun.max(y)[0];
									mm0[1] = VectorFun.min(y)[0];
									minmax(mm0,mm2);
									// prepare the next scale
									// summing for the contrasted image
									vectorSum(x,y);
								}
								catch (Throwable t){
									System.out.println("Problem with the scale normalization");
									t.printStackTrace();
								}
							}
						});
					}
				}
			}
			finally {
				exec.shutdown();
			}
			while(!exec.isTerminated()){
				// wait
			}*/
			for (int j=0; j<(int)parameters[0]; j++){
				for (int k=0; k<(int)parameters[5]; k++){
					double[] y = ScaleAnalysis.normalize(ScaleAnalysis.singleDecomposition(z,w,h,lambda*Math.pow(parameters[3],j),parameters[2],(int)parameters[5],k,parameters[6]),w,h,parameters[4]*lambda*Math.pow(parameters[3],j));
					// get rid of nan values
					for (int l=0; l<y.length; l++){
						// we need to check that the pixel values
						// are regular double values
						if (Double.isNaN(y[l])){
							y[l] = 0;
						}
					}
					stack2.setProcessor(new FloatProcessor(w,h,y),(i-1)*(int)parameters[0]*(int)parameters[5]+j*(int)parameters[5]+k+1);
					// maxima/minima reevaluation
					double[] mm0 = new double[2];
					mm0[0] = VectorFun.max(y)[0];
					mm0[1] = VectorFun.min(y)[0];
					minmax(mm0,mm2);
					// prepare the next scale
					// summing for the contrasted image
					VectorFun.addi(x,y);
				}
			}
			stack3.addSlice("None", new FloatProcessor(w,h,x));
			double[] mm0 = new double[2];
			mm0[0] = VectorFun.max(x)[0];
			mm0[1] = VectorFun.min(x)[0];
			minmax(mm0,mm3);
		}
		// create the ImagePlus containers
		ImagePlus imp2 = new ImagePlus("Normalized Scales",stack2);
		imp2.getProcessor().setMinAndMax(mm2[1],mm2[0]);
		ImagePlus imp3 = new ImagePlus("Contrasted",stack3);
		imp3.getProcessor().setMinAndMax(mm3[1],mm3[0]);
		// save the ImagePlus containers
		String basedir = (new File(filename)).getParent();
		String name = TypeReader.imageBase((new File(filename)).getName());
		IJ.saveAsTiff(imp2,targetdir+File.separator+NORMALIZEDSCALES+File.separator+name+TypeReader.TIFF[0]);
		IJ.saveAsTiff(imp3,targetdir+File.separator+CONTRASTED+File.separator+name+TypeReader.TIFF[0]);
	}
	
	private static void writeConfigTXT(final String name, final double[] parameters){
		try {
			File infotxt = new File(name);
			BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
			String newline = System.getProperty("line.separator");
			writer.write("Number of scales: "+String.format("%d",(int)parameters[0])+newline+
				"Lambda0: "+String.format("%.2f",parameters[1])+newline+
				"Sigma: "+String.format("%.2f",parameters[2])+newline+
				"Scale factor: "+String.format("%.2f",parameters[3])+newline+
				"Normalization to scale ratio : "+String.format("%.2f",parameters[4])+newline+
				"Number of angles: "+String.format("%d",(int)parameters[5])+newline+
				"Angular width: "+String.format("%.2f",parameters[6])+newline+
				"Maximum Scale: "+String.format("%.2f",parameters[1]*Math.pow(parameters[3],(int)parameters[0]))+newline+
				"Maximum normalization window: "+String.format("%.2f",parameters[1]*Math.pow(parameters[3],(int)parameters[0])*parameters[4]));
			writer.close();
		}
		catch (Exception e){}
	}
}