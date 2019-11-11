package flib.ij.featureextraction;

import java.io.File;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.featureextraction.FolderNames;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.algorithms.regions.RegionFunctions;
import ij.io.DirectoryChooser;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class RegionGeometry {
	
	// subfolder names for various purposes
	static String SIZE = FolderNames.SIZE;
	static String GYRRAD = FolderNames.GYRRAD;
	static String ELONG = FolderNames.ELONG;
	static String PCADIR = FolderNames.PCADIR;
	static String ISOPER = FolderNames.ISOPER;
	// the following watershed will be based on a scale decomposition
	// location: the base folder which contains the normalized scales folder
	public static void run(String location, String keydir){
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
		File[] directoryListing = (new File(keydir)).listFiles();
		String size = targetdir+File.separator+SIZE;
		String gyra = targetdir+File.separator+GYRRAD;
		String elong = targetdir+File.separator+ELONG;
		String pcadir = targetdir+File.separator+PCADIR;
		String isoper = targetdir+File.separator+ISOPER;
		// create the directory for the watershed scale images if it doesn't exist
		new File(size).mkdirs();
		new File(gyra).mkdirs();
		new File(elong).mkdirs();
		new File(pcadir).mkdirs();
		new File(isoper).mkdirs();
		ReadWrite.writeObject(size+File.separator+FolderNames.KEYPARAM,keydir);
		ReadWrite.writeObject(gyra+File.separator+FolderNames.KEYPARAM,keydir);
		ReadWrite.writeObject(elong+File.separator+FolderNames.KEYPARAM,keydir);
		ReadWrite.writeObject(pcadir+File.separator+FolderNames.KEYPARAM,keydir);
		ReadWrite.writeObject(isoper+File.separator+FolderNames.KEYPARAM,keydir);
		for (int i=0; i<directoryListing.length; i++){
			String name = directoryListing[i].getAbsolutePath();
			if (TypeReader.isTiff(name)){
				// this file should already be a .tif file
				// if the same image doesn't already exist in the
				// watershed directory create it
				String target_name = size+File.separator+TypeReader.imageBase(directoryListing[i].getName())+".key";
				if (!(new File(target_name)).exists()){
					geometry(name,size,gyra,elong,pcadir,isoper);
				}
			}
		}
	}
	
	// alternative run functions
	public static void run(String keydir){
		run("",keydir);
	}
	
	// filename: scales image file name
	// wshed: directory where the watershed regions image will be stored
	// keyd: distances of all region points to their local minima/maxima
	// at the corresponding scale
	// keya: same as for keyd, just for the angle relationship
	private static void geometry(final String filename, String size, String gyra, String elong, String pcadir, String isoper){
		// open the original image file
		ImagePlus imp = IJ.openImage(filename);
		// image properties
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		// prepare a stack for the regions image
		ImageStack stack = imp.getImageStack();
		// prepare a stack for the sizes
		double[][] sizes = new double[n][];
		// prepare a stack for the radius of gyration
		double[][] gyras = new double[n][];
		// prepare a stack for the elongation factor
		double[][] elongs = new double[n][];
		// prepare a stack for the main pca angle
		double[][] pcadirs = new double[n][];
		// maxima and minima for the new stacks
		double[][] isopers = new double[n][];
		// in this case the keys are the region centroids
		int[][] keys = new int[n][];
		// go through all scales
		for (int i=1; i<n+1; i++){
			double[] x = VectorConv.float2double((float[])(stack.getPixels(i)));
			int max = (int)VectorFun.max(x)[0]+1;
			sizes[i-1] = new double[max];
			gyras[i-1] = new double[max];
			elongs[i-1] = new double[max];
			pcadirs[i-1] = new double[max];
			isopers[i-1] = new double[max];
			int[][] reg = RegionFunctions.getRegions(x);
			double[][] covmat = RegionFunctions.regionCovMat(w,h,reg);
			double[][] rpca = RegionFunctions.regionGeometry(covmat);
			double[] iq = RegionFunctions.boundaryLength(w,h,x);
			for (int j=0; j<max; j++){
				sizes[i-1][j] = reg[j].length+1e-8;
				gyras[i-1][j]= sizes[i-1][j]/(2*Math.PI)/(rpca[j][2]+0.5);
				elongs[i-1][j] = rpca[j][1];
				pcadirs[i-1][j] = rpca[j][0];
				isopers[i-1][j] = sizes[i-1][j]*(4*Math.PI)/((iq[j]+4)*(iq[j]+4));
			}				
		}
		// save the key files
		String name2 = TypeReader.imageBase((new File(filename)).getName());
		ReadWrite.writeObject(size+File.separator+name2+".key",sizes);
		ReadWrite.writeObject(gyra+File.separator+name2+".key",gyras);
		ReadWrite.writeObject(elong+File.separator+name2+".key",elongs);
		ReadWrite.writeObject(pcadir+File.separator+name2+".key",pcadirs);
		ReadWrite.writeObject(isoper+File.separator+name2+".key",isopers);
	}
}