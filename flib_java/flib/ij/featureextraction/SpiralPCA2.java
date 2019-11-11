package flib.ij.featureextraction;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.featureextraction.FolderNames;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.algorithms.regions.RegionFunctions;
import flib.ij.stack.StackTo32bit;
import flib.ij.stack.StackOperations;
import ij.io.DirectoryChooser;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.io.Opener;
import ij.io.FileInfo;
import ij.io.TiffDecoder;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

public class SpiralPCA {
	static String NORMALIZEDSCALES = FolderNames.NORMALIZEDSCALES;
	static String WATERSHED = FolderNames.WATERSHED;
	static String SPIRAL = FolderNames.SPIRAL;
	public static void run(final String location, final String target, final double[] parameters, final String orig, boolean[] scale){
		String targetdir;
		// check if we need to let the user choose a directory
		// containing the images to be transformed
		if (location.equals("")){
			targetdir = (new DirectoryChooser("Choose a folder")).getDirectory();
			if (targetdir.isEmpty()){
				IJ.log("User canceled the SpiralPCA dialog! Bye!");
				return;
			}
		}
		// or if the user has provided an explicit target directory
		else {
			targetdir = new String(location);
		}
		String normscale = targetdir+File.separator+NORMALIZEDSCALES;
		String wshed = targetdir+File.separator+WATERSHED;
		String tar = targetdir+File.separator+target;
		new File(tar).mkdirs();
		// find all fiji readable files inside the normalized scales directory
		File[] directoryListing = (new File(wshed)).listFiles();
		// find all files which are images among the directory listing
		int[] file_number = TypeReader.tiffNumber(directoryListing);
		// spiral pca parameters
		String pca_config = tar+File.separator+"spiral_config.ser";
		ReadWrite.writeObject(pca_config,parameters);
		String configtxt = tar+File.separator+"spiral_config.txt";
		writeConfigTXT(configtxt,orig,parameters);
		String scale_config = normscale+File.separator+"config.ser";
		int num_scale = (int)((double[])ReadWrite.readObject(scale_config))[0];
		if (scale.length!=num_scale){
			scale = new boolean[num_scale];
		}
		String use_scale = tar+File.separator+"scales.ser";
		ReadWrite.writeObject(use_scale,scale);
		// collection of the number of regions in each scale, each tif,
		// and for each original subimage. The value is the cumultative
		// sum of all prior image regions of that scale for all images
		int[][][] scale_regions = new int[num_scale][file_number.length][];
		for (int i=0; i<file_number.length; i++){
			String name = directoryListing[file_number[i]].getAbsolutePath();
			countRegions(name,scale_regions,i,scale);
		}
		int count = 0;
		for (int i=0; i<num_scale; i++){
			if (!scale[i]){
				for (int j=0; j<scale_regions[i].length; j++){
					for (int k=0; k<scale_regions[i][j].length; k++){
						count+=scale_regions[i][j][k];
						scale_regions[i][j][k] = count;
					}
				}
				count = 0;
			}
		}
		// scale factor
		double fact = ((double[])ReadWrite.readObject(scale_config))[3];
		// radial increment
		double ri = parameters[1]*((double[])ReadWrite.readObject(scale_config))[1];
		// cutoff radius
		double cor = parameters[0]*((double[])ReadWrite.readObject(scale_config))[1];
		Random rng = new Random();
		double[][][] pca_val = new double[num_scale][][];
		// go through all scales and calculate covariance matrices
		for (int i=0; i<num_scale; i++){
			if(!scale[i]){
				// neighborhood of the current scale
				int[][] shape = NeighborhoodSample.spiralCoord(parameters[2],cor,ri,1,0.5,6);
				int l = shape.length;
				// temporary covariance matrix
				double[][][] covmat = new double[(int)parameters[5]][l][l];
				double[][] meanmat = new double[(int)parameters[5]][l];
				pca_val[i] = new double[(int)parameters[4]*(int)parameters[5]][l];
				int[] samples = new int[(int)parameters[3]];
				int np = scale_regions[i][scale_regions[i].length-1][scale_regions[i][scale_regions[i].length-1].length-1];
				for (int j=0; j<samples.length; j++){
					samples[j] = rng.nextInt(np);
				}
				Arrays.sort(samples);
				int s = 0;
				// go through all images
				for (int j=0; j<file_number.length; j++){
					if (samples[s]<scale_regions[i][j][scale_regions[i][j].length-1]){
						String name = directoryListing[file_number[j]].getAbsolutePath();
						String name2 = orig+File.separator+directoryListing[file_number[j]].getName();
						String name3 = normscale+File.separator+directoryListing[file_number[j]].getName();
						s = covarianceUpdate(name,name2,name3,scale_regions,i,j,shape,covmat,meanmat,samples,s);
					}
					if (s==samples.length) break;
				}
				// mean subtraction for the covariance
				for (int j=0; j<(int)parameters[5]; j++){
					for (int k=0; k<l; k++){
						for (int m=0; m<l; m++){
							covmat[j][k][m] = covmat[j][k][m]/parameters[3]-meanmat[j][k]/parameters[3]*meanmat[j][m]/parameters[3];
						}
					}
				}
				// eigenvalue decomposition
				for (int j=0; j<(int)parameters[5]; j++){
					DenseDoubleMatrix2D covM = new DenseDoubleMatrix2D(covmat[j]);
					EigenvalueDecomposition eig = new EigenvalueDecomposition(covM);
					for (int k=0; k<(int)parameters[4]; k++){
						pca_val[i][(int)parameters[4]*j+k] = eig.getV().viewColumn(l-1-k).toArray();
						//pca_val[i][(int)parameters[4]*j+k] = eig.getV().viewColumn(k).toArray();
					}
				}
			}
			ri*=fact;
			cor*=fact;
		}
		// store the pca value into a file
		String pca_file = tar+File.separator+"pca.ser";
		ReadWrite.writeObject(pca_file,pca_val);
	}
	
	public static void run(final String location, final String target, final double[] parameters, final String orig){
		run(location,target,parameters,orig,new boolean[0]);
	}
	
	public static void run(final String location, final String target, final double[] parameters){
		run(location,target,parameters,location+File.separator+NORMALIZEDSCALES);
	}
	
	public static void run(final String location, final double[] parameters){
		run(location,SPIRAL,parameters);
	}
	
	public static void run(final double[] parameters){
		run("",SPIRAL,parameters);
	}
	
	private static void writeConfigTXT(final String name, final String orig, final double[] parameters){
		try {
			File infotxt = new File(name);
			BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
			String newline = System.getProperty("line.separator");
			writer.write("Spiral parameters"+newline+
				"Cutoff radius multiplier: "+String.format("%.2f",parameters[0])+newline+
				"Radial increment multiplier: "+String.format("%.2f",parameters[1])+newline+
				"Angular increment: "+String.format("%.2f",parameters[2])+newline+
				newline+"PCA parameters"+newline+
				"Number of samples per scale: "+String.format("%d",(int)parameters[3])+newline+
				"Number of components to store: "+String.format("%d",(int)parameters[4])+newline+
				"Number of classes: "+String.format("%d",(int)parameters[5])+newline+
				"Noise fraction: "+String.format("%.2f",(parameters[6]))+newline+
				"Sample origin directory: "+orig);
			writer.close();
		}
		catch (Exception e){}
	}
	
	private static void countRegions(final String filename, int[][][] scale_regions, int loc, final boolean[] scale){
		ImagePlus imp = StackTo32bit.convert(IJ.openImage(filename));
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		if (n%scale_regions.length!=0){
			IJ.log("Something is wrong with image "+filename);
		}
		else {
			int num_scale = scale_regions.length;
			int num_im = n/num_scale;
			for (int i=0; i<scale_regions.length; i++){
				if (!scale[i]){
					scale_regions[i][loc] = new int[num_im];
					for (int j=0; j<num_im; j++){
						// find the number of regions for this subimage
						scale_regions[i][loc][j] = (int)VectorFun.max(VectorConv.float2double((float[])(stack.getProcessor(1+num_scale*j+i).convertToFloat().getPixels())))[0];
					}
				}
			}
		}
	}
	
	private static int covarianceUpdate(final String filename, final String filename2, final String filename3,final int[][][] scale_regions, int loc, int im, final int[][] shape, double[][][] covmat, double[][] meanmat, final int[] samples, int s){
		// filename belongs to the watershed images
		// filename2 is for the images which the watershed keypoints should 
		// be applied to
		// number of scales in the stack
		TiffDecoder td = new TiffDecoder((new File(filename)).getParent(),(new File(filename)).getName());
		FileInfo[] info = null;
		try {
			info = td.getTiffInfo();
		} catch (IOException e) {
			String msg = e.getMessage();
			if (msg==null||msg.equals("")) msg = ""+e;
			IJ.error("Open TIFF", msg);
		}
		FileInfo fi = info[0];
		int n = 0;
		if (info.length==1 && fi.nImages>1) {
			n = fi.nImages;
		}
		else {
			n = info.length;
		}
		int num_scale = scale_regions.length;
		// number of different images in the stack
		int num_im = n/num_scale;
		td = new TiffDecoder((new File(filename2)).getParent(),(new File(filename2)).getName());
		info = null;
		try {
			info = td.getTiffInfo();
		} catch (IOException e) {
			String msg = e.getMessage();
			if (msg==null||msg.equals("")) msg = ""+e;
			IJ.error("Open TIFF", msg);
		}
		fi = info[0];
		int n2 = 0;
		if (info.length==1 && fi.nImages>1) {
			n2 = fi.nImages;
		}
		else {
			n2 = info.length;
		}
		int num_class = n2/(num_scale*num_im);
		int type = 0;
		if (num_class==0){
			type = 1;
			num_class = 1;
		}
		int a = s;
		for (int i=0; i<num_im; i++){
			int l = 0;
			// adjustment of the coordinates
			if (i>0){
				l = scale_regions[loc][im][i-1];
			}
			else if (im>0){
				l = scale_regions[loc][im-1][scale_regions[loc][im-1].length-1];
			}
			// current watershed image
			ImageStack stack = (new Opener()).openTiff(filename,1+num_scale*i+loc).getImageStack();
			double[] x = VectorConv.float2double((float[])stack.getPixels(1));
			double[] z = VectorConv.float2double((float[])(new Opener()).openTiff(filename3,1+num_scale*i+loc).getImageStack().getPixels(1));
			int w = stack.getWidth();
			int h = stack.getHeight();
			// get the regions and key points
			int[][] reg = RegionFunctions.getRegions(x);
			int[] keys = RegionFunctions.getKeys(reg,z);
			for (int j=0; j<num_class; j++){
				a = s;
				int ind = 1+num_scale*num_class*i+num_scale*j+loc;
				if (type==1){
					ind = 1+i;
				}
				double[] y = VectorConv.float2double((float[])(new Opener()).openTiff(filename2,ind).getImageStack().getPixels(1));
				while(samples[a]<scale_regions[loc][im][i]){
					// find the key point location
					int[] point = new int[1];
					point[0] = keys[samples[a]-l];
					double[] val = NeighborhoodSample.sample2d(point,w,h,shape,0,0,y)[0];
					for (int k=0; k<val.length; k++){
						meanmat[j][k]+=val[k];
						for (int m=0; m<val.length; m++){
							covmat[j][k][m]+=val[k]*val[m];
						}
					}
					a++;
					if (a==samples.length) break;
				}
			}
			s = a;
			if (s==samples.length) break;
		}
		return s;
	}
}