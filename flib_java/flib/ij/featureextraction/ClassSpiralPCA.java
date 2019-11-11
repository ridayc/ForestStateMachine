package flib.ij.featureextraction;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.TreeSet;
import java.lang.Math;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.featureextraction.FolderNames;
import flib.ij.featureextraction.ExtractLabels;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.SortPair2;
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

public class ClassSpiralPCA {
	static String NORMALIZEDSCALES = FolderNames.NORMALIZEDSCALES;
	static String WATERSHED = FolderNames.WATERSHED;
	static String SPIRAL = FolderNames.SPIRAL;
	public static void run(final String location, final String target, final String lab, final double[] parameters, final String orig, final boolean[] classes, boolean[] scale){
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
		File[] directoryListing = (new File(lab)).listFiles();
		// find all files which are images among the directory listing
		int[] file_number = TypeReader.fileNumber(directoryListing);
		// spiral pca parameters
		String pca_config = tar+File.separator+"spiral_config.ser";
		ReadWrite.writeObject(pca_config,parameters);
		String configtxt = tar+File.separator+"spiral_config.txt";
		writeConfigTXT(configtxt,lab,parameters);
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
		int[][][][] scale_regions = new int[num_scale][classes.length][file_number.length][0];
		TreeSet<SortPair2> labels;
		if ((int)parameters[7]==0){
			labels = ExtractLabels.labelConversion(lab);
		}
		else {
			labels = new TreeSet<SortPair2>();
			for (int i=0; i<classes.length; i++){
				labels.add(new SortPair2(i,i));
			}
		}
		// remove class which are on the list from the classes to be traversed
		for (int i=0; i<classes.length; i++){
			if (classes[i]){
				labels.remove(new SortPair2(i,0));
			}
		}
		for (int i=0; i<file_number.length; i++){
			String[] name = new String[4];
			name[0] = directoryListing[file_number[i]].getAbsolutePath();
			// target storage location
			name[1] = orig+File.separator+TypeReader.imageBase(directoryListing[file_number[i]].getName())+TypeReader.TIFF[0];
			name[2] = normscale+File.separator+TypeReader.imageBase(directoryListing[file_number[i]].getName())+TypeReader.TIFF[0];
			name[3] = wshed+File.separator+TypeReader.imageBase(directoryListing[file_number[i]].getName())+TypeReader.TIFF[0];
			countRegions(name,scale_regions,i,scale,labels);
		}
		int count = 0;
		// go through all scales
		for (int i=0; i<num_scale; i++){
			// go through all classes
			for (int j=0; j<scale_regions[i].length; j++){
				// go through all image files
				for (int k=0; k<scale_regions[i][j].length; k++){
					// go through all image layers
					for (int m=0; m<scale_regions[i][j][k].length; m++){
						count+=scale_regions[i][j][k][m];
						scale_regions[i][j][k][m] = count;
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
		int dim = 0;
		for (int i=0; i<num_scale; i++){
			if (!scale[i]){
				dim++;
			}
		}
		double[][][] pca_val = new double[dim][][];
		// go through all scales and calculate covariance matrices
		dim = 0;
		for (int i=0; i<num_scale; i++){
			if (!scale[i]){
				// neighborhood of the current scale
				int[][] shape = NeighborhoodSample.spiralCoord(parameters[2],cor,ri,1,0.5,6);
				int l = shape.length;
				// temporary covariance matrix
				double[][][] covmat = new double[classes.length][l][l];
				double[][] meanmat = new double[classes.length][l];
				pca_val[i] = new double[(int)parameters[4]*labels.size()][l];
				int[][] samples = new int[classes.length][(int)parameters[3]];
				for (int k=0; k<classes.length; k++){
					if (!classes[k]){
						int np = scale_regions[i][k][scale_regions[i][k].length-1][scale_regions[i][k][scale_regions[i][k].length-1].length-1];
						for (int j=0; j<samples[k].length; j++){
							samples[k][j] = rng.nextInt(np);
						}
						Arrays.sort(samples[k]);
					}
				}
				int[] s = new int[classes.length];
				int[] counter = new int[classes.length];
				// go through all images
				for (int j=0; j<file_number.length; j++){
					String[] name = new String[4];
					name[0] = directoryListing[file_number[j]].getAbsolutePath();
					// target storage location
					name[1] = orig+File.separator+TypeReader.imageBase(directoryListing[file_number[j]].getName())+TypeReader.TIFF[0];
					name[2] = normscale+File.separator+TypeReader.imageBase(directoryListing[file_number[j]].getName())+TypeReader.TIFF[0];
					name[3] = wshed+File.separator+TypeReader.imageBase(directoryListing[file_number[j]].getName())+TypeReader.TIFF[0];
					covarianceUpdate(name,scale_regions,i,j,shape,covmat,meanmat,samples,s,labels,counter);
				}
				// mean subtraction for the covariance
				for (int j=0; j<(int)parameters[5]; j++){
					if (!classes[j]){
						for (int k=0; k<l; k++){
							for (int m=0; m<l; m++){
								covmat[j][k][m] = covmat[j][k][m]/parameters[3]-meanmat[j][k]/parameters[3]*meanmat[j][m]/parameters[3];
							}
						}
					}
				}
				// eigenvalue decomposition
				int c = 0;
				for (int j=0; j<classes.length; j++){
					if (!classes[j]){
						DenseDoubleMatrix2D covM = new DenseDoubleMatrix2D(covmat[j]);
						EigenvalueDecomposition eig = new EigenvalueDecomposition(covM);
						for (int k=0; k<(int)parameters[4]; k++){
							pca_val[dim][(int)parameters[4]*c+k] = eig.getV().viewColumn(l-1-k).toArray();
							//pca_val[i][(int)parameters[4]*j+k] = eig.getV().viewColumn(k).toArray();
						}
						c++;
					}
				}
				dim++;
			}
			ri*=fact;
			cor*=fact;
		}
		// store the pca value into a file
		String pca_file = tar+File.separator+"pca.ser";
		ReadWrite.writeObject(pca_file,pca_val);
	}
	
	public static void run(final String location, final String target, final String lab, final double[] parameters, final boolean[] classes, final boolean[] scale){
		run(location,target,lab,parameters,NORMALIZEDSCALES,classes,scale);
	}

	
	private static void writeConfigTXT(final String name, final String lab, final double[] parameters){
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
				"Number of components to store: "+String.format("%d",(int)parameters[4]*(int)parameters[5])+newline+
				"Number of classes: "+String.format("%d",(int)parameters[5])+newline+
				"Noise Fraction: "+String.format("%.2f",(parameters[6]))+newline+
				"Other parameters: "+newline+
				"Label type: "+String.format("%d",(int)parameters[7])+newline+
				"Class labels: "+lab);
			writer.close();
		}
		catch (Exception e){}
	}
	
	private static void countRegions(final String[] filename, int[][][][] scale_regions, int loc, final boolean[] scale, final TreeSet<SortPair2> labels){
		TiffDecoder td = new TiffDecoder((new File(filename[2])).getParent(),(new File(filename[2])).getName());
		FileInfo[] info = null;
		try {
			info = td.getTiffInfo();
		} catch (IOException e) {
			String msg = e.getMessage();
			if (msg==null||msg.equals("")) msg = ""+e;
			IJ.error("Open TIFF", msg);
		}
		FileInfo fi = info[0];
		int num_im = 0;
		if (info.length==1 && fi.nImages>1) {
			num_im = fi.nImages;
		}
		else {
			num_im = info.length;
		}
		int num_scale = scale_regions.length;
		num_im = num_im/num_scale;
		for (int j=0; j<num_im; j++){
			double[] y;
			if (filename[0].endsWith(TypeReader.TIFF[0])){
				y = VectorConv.float2double((float[])(new Opener()).openTiff(filename[0],1+j).getImageStack().getPixels(1));
			}
			else {
				y = VectorConv.float2double((float[])IJ.openImage(filename[0]).getImageStack().convertToFloat().getPixels(1));
			}
			for (int i=0; i<scale_regions.length; i++){
				if (!scale[i]){
					double[] x = VectorConv.float2double((float[])(new Opener()).openTiff(filename[3],1+num_scale*j+i).getImageStack().getPixels(1));
					double[] x2 = VectorConv.float2double((float[])(new Opener()).openTiff(filename[2],1+num_scale*j+i).getImageStack().getPixels(1));
					int[][] reg = RegionFunctions.getRegions(x);
					int[] keys = RegionFunctions.getKeys(reg,x2);
					for (int k=0; k<scale_regions[i].length; k++){
						scale_regions[i][k][loc] = new int[num_im];
					}
					for (int k=0; k<keys.length; k++){
						SortPair2 sp = new SortPair2(y[keys[k]],1);
						if (labels.contains(sp)){
							int a = (int)labels.floor(sp).getOriginalIndex();
							scale_regions[i][a][loc][j]++;
						}
					}
				}
			}
		}
	}
	
	private static void covarianceUpdate(final String[] filename,final int[][][][] scale_regions, int loc, int im, final int[][] shape, double[][][] covmat, double[][] meanmat, final int[][] samples, int[] s, final TreeSet<SortPair2> labels, int[] counter){
		// filename belongs to the watershed images
		// filename2 is for the images which the watershed keypoints should 
		// be applied to
		// number of scales in the stack
		TiffDecoder td = new TiffDecoder((new File(filename[2])).getParent(),(new File(filename[2])).getName());
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
		td = new TiffDecoder((new File(filename[1])).getParent(),(new File(filename[1])).getName());
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
		int num_im = n/scale_regions.length;
		int num_class = scale_regions[loc].length;
		int num_scale = scale_regions.length;
		for (int i=0; i<num_im; i++){
			// current watershed image
			ImageStack stack = (new Opener()).openTiff(filename[3],1+num_scale*i+loc).getImageStack();
			int w = stack.getWidth();
			int h = stack.getHeight();
			double[] x = VectorConv.float2double((float[])stack.getPixels(1));
			double[] x2 = VectorConv.float2double((float[])(new Opener()).openTiff(filename[2],1+num_scale*i+loc).getImageStack().getPixels(1));
			int ind = 1+num_scale*i+loc;
			if (n2<n){
				ind = 1+i;
			}
			double[] y = VectorConv.float2double((float[])(new Opener()).openTiff(filename[1],ind).getImageStack().getPixels(1));
			double[] z;
			if (filename[0].endsWith(TypeReader.TIFF[0])){
				z = VectorConv.float2double((float[])(new Opener()).openTiff(filename[0],1+i).getImageStack().getPixels(1));
			}
			else {
				z = VectorConv.float2double((float[])IJ.openImage(filename[0]).getImageStack().convertToFloat().getPixels(1));
			}
			// get the regions and key points
			int[][] reg = RegionFunctions.getRegions(x);
			int[] keys = RegionFunctions.getKeys(reg,x2);
			for (int k=0; k<keys.length; k++){
				SortPair2 sp = new SortPair2(z[keys[k]],1);
				if (labels.contains(sp)){
					int a = (int)labels.floor(sp).getOriginalIndex();
					while (samples[a][s[a]]==counter[a]){
						// find the key point location
						int[] point = new int[1];
						point[0] = keys[k];
						double[] val = NeighborhoodSample.sample2d(point,w,h,shape,0,0,y)[0];
						for (int o=0; o<val.length; o++){
							meanmat[a][o]+=val[o];
							for (int m=0; m<val.length; m++){
								covmat[a][o][m]+=val[o]*val[m];
							}
						}
						s[a]++;
						if (s[a]==samples[a].length){
							s[a]=samples[a].length-1;
							break;
						}
					}
					counter[a]++;
				}
			}
		}
	}
}