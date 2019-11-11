package flib.ij.featureextraction;

import java.lang.Math;
import java.util.Random;
import java.io.File;
import java.io.IOException;
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

public class SpiralProjection {
	static String[] EXTENSIONS = TypeReader.TIFF;
	static String NORMALIZEDSCALES = FolderNames.NORMALIZEDSCALES;
	static String WATERSHED = FolderNames.WATERSHED;
	static String SPIRAL = FolderNames.SPIRAL;
	
	public static void run(final String location, final String target, final String orig){
		String targetdir;
		// check if we need to let the user choose a directory
		// containing the images to be transformed
		if (location.equals("")){
			targetdir = (new DirectoryChooser("Choose a folder")).getDirectory();
			if (targetdir.isEmpty()){
				IJ.log("User canceled the SpiralProjections dialog! Bye!");
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
		// find all fiji readable files inside the normalized scales directory
		File[] directoryListing = (new File(wshed)).listFiles();
		// find all files which are images among the directory listing
		int[] file_number = TypeReader.tiffNumber(directoryListing);
		// all different parameters parameters
		String pca_config = tar+File.separator+"pca.ser";
		double[][][] pca_val = (double[][][])ReadWrite.readObject(pca_config);
		String scale_config = normscale+File.separator+"config.ser";
		double[] scale_param = (double[])ReadWrite.readObject(scale_config);
		String spiral_config = tar+File.separator+"spiral_config.ser";
		double[] spiral_param = (double[])ReadWrite.readObject(spiral_config);
		String use_scale = tar+File.separator+"scales.ser";
		boolean[] scale = (boolean[])ReadWrite.readObject(use_scale);
		int num_scale = (int)scale_param[0];
		// go through all images
		for (int i=0; i<file_number.length; i++){
			String target_name = tar+File.separator+directoryListing[file_number[i]].getName();
			if (!(new File(target_name)).exists()){
				String name = directoryListing[file_number[i]].getAbsolutePath();
				String name2 = orig+File.separator+directoryListing[file_number[i]].getName();
				String name3 = normscale+File.separator+directoryListing[file_number[i]].getName();
				pcaEval(name,name2,name3,tar,scale_param,spiral_param,pca_val,scale);
			}
		}
	}
	
	public static void run(final String location,final String target){
		run(location,target,location+File.separator+NORMALIZEDSCALES);
	}
	
	public static void run(final String location){
		run(location,SPIRAL);
	}
	
	public static void run(){
		run("",SPIRAL);
	}
	
	private static void pcaEval(final String filename, final String filename2,final String filename3, final String spdir, final double[] scale_param, final double[] spiral_param, final double[][][] pca_val, final boolean[] scale){
		// open the original image file and convert it (if neccessary)
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
		ImageStack stack = (new Opener()).openTiff(filename,1).getImageStack();
		int w = stack.getWidth();
		int h = stack.getHeight();
		int num_im = n/((int)scale_param[0]);
		// prepare a stack for the watershed image
		ImageStack stack3 = new ImageStack(w,h);
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		Random rng = new Random();
		for (int i=0; i<num_im; i++){
			// scale factor
			double fact = scale_param[3];
			// radial increment
			double ri = spiral_param[1]*scale_param[1];
			// cutoff radius
			double cor = spiral_param[0]*scale_param[1];
			for (int j=0; j<(int)scale_param[0]; j++){
				if (!scale[j]){
					// watershed regions according to scale
					stack  = (new Opener()).openTiff(filename,1+i*(int)scale_param[0]+j).getImageStack();
					double[] x = VectorConv.float2double((float[])(stack.getPixels(1)));
					double[] x2 = VectorConv.float2double((float[])((new Opener()).openTiff(filename3,1+i*(int)scale_param[0]+j).getImageStack().getPixels(1)));
					int[][] reg = RegionFunctions.getRegions(x);
					int[] keys = RegionFunctions.getKeys(reg,x2);
					int[][] shape = NeighborhoodSample.spiralCoord(spiral_param[2],cor,ri,1,0.5,6);
					int ind = 1+i*(int)scale_param[0]+j;
					if (n2<n){
						ind = 1+i;
					}
					double[] y = VectorConv.float2double((float[])((new Opener()).openTiff(filename2,ind).getImageStack().getPixels(1)));
					for (int k=0; k<pca_val[j].length; k++){
						double[] z = new double[keys.length];
						int[] point = new int[1];
						for (int m=0; m<keys.length; m++){
							point[0] = keys[m];
							double[] val = NeighborhoodSample.sample2d(point,w,h,shape,0,0,y)[0];
							z[m] = VectorFun.sum(VectorFun.mult(val,pca_val[j][k]));
						}
						double[] temp = RegionFunctions.regionFill(w,h,reg,z);
						for (int m=0; m<temp.length; m++){
							temp[m]+=rng.nextGaussian()*temp[m]*spiral_param[6];
						}
						stack3.addSlice("None", new FloatProcessor(w,h,temp));
						double t = VectorFun.min(temp)[0];
						if (t<min){
							min = t;
						}
						t = VectorFun.max(temp)[0];
						if (t>max){
							max = t;
						}
					}
				}
				ri*=fact;
				cor*=fact;
			}
		}
		// create the new ImagePlus container
		ImagePlus imp3 = new ImagePlus("KeyPointPCA",stack3);
		imp3.getProcessor().setMinAndMax(min,max);
		// save the ImagePlus container
		String name = (new File(filename)).getName();
		IJ.saveAsTiff(imp3,spdir+File.separator+name);
	}
}