package flib.ij.featureextraction;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Random;
import java.lang.Math;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.featureextraction.FolderNames;
import flib.ij.featureextraction.FileInterpreter;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.algorithms.sampling.NeighborhoodSample;
import ij.IJ;
import ij.ImageStack;
import ij.io.Opener;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

public class SpiralPCA {
	
	public static void storeComponents(String orig, String keydir, String targetdir, final double[] parameters){
		// make sure the storage folder exists
		new File(targetdir).mkdirs();
		// we're going to assume that a there is a key location file for each image
		// in the image folder
		// prepare the file interpreter
		FileInterpreter FI = new FileInterpreter(orig);
		// collection of key data per scales
		int[][][] temp = new int[FI.getBases().length][][];
		for (int i=0; i<FI.getBases().length; i++){
			int[][] keys = (int[][])ReadWrite.readObject(keydir+File.separator+FI.getBases()[i]+".key");
			int scales = keys.length/(FI.getInfo()[0][i]/(int)parameters[7]);
			temp[i] = new int[scales][FI.getInfo()[0][i]/(int)parameters[7]];
			for (int j=0; j<scales; j++){
				for (int k=0; k<temp[i][j].length; k++){
					temp[i][j][k] = keys[k*scales+j].length;
				}
			}
		}
		int scales = temp[0].length;
		// offset data at all scales and images and sub-images
		int[][][][] info = new int[4][scales][FI.getBases().length][];
		for (int i=0; i<scales; i++){
			int sum = 0;
			for (int j=0; j<FI.getBases().length; j++){
				info[0][i][j] = new int[1];
				info[1][i][j] = new int[1];
				info[2][i][j] = new int[temp[j][i].length];
				info[3][i][j] = new int[temp[j][i].length];
				int sum2 = 0;
				for (int k=0; k<temp[j][i].length; k++){
					info[2][i][j][k] = temp[j][i][k];
					sum2+= temp[j][i][k];
					info[3][i][j][k] = sum2;
				}
				info[0][i][j][0] = sum2;
				sum+=sum2;
				info[1][i][j][0] = sum;
			}
		}
		// preparation of the sampling locations
		int[][] loc = new int[scales][(int)parameters[3]];
		Random rng = new Random();
		for (int i=0; i<scales; i++){
			int val = info[1][i][FI.getBases().length-1][0];
			for (int j=0; j<(int)parameters[3]; j++){
				loc[i][j] = rng.nextInt(val);
			}
		}
		// get the pca components
		int layers;
		if ((int)parameters[5]>1){
			layers = (int)parameters[5];
		}
		else {
			layers = (int)parameters[7];
		}
		double[][][] pca_val = pcaComponents(FI,keydir,loc,info,parameters,scales,layers);
		// store all relevant information
		// spiral parameters and settings
		String pca_config = targetdir+File.separator+"spiral_config.ser";
		ReadWrite.writeObject(pca_config,parameters);
		// in text file form
		String configtxt = targetdir+File.separator+"spiral_config.txt";
		writeConfigTXT(configtxt,orig,parameters);
		// store the pca components
		String pca_file = targetdir+File.separator+"pca.ser";
		ReadWrite.writeObject(pca_file,pca_val);
	}

	private static void writeConfigTXT(final String name, final String orig, final double[] parameters){
		try {
			File infotxt = new File(name);
			BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
			String newline = System.getProperty("line.separator");
			writer.write("Spiral parameters"+newline+
				"Intial cutoff radius: "+String.format("%.2f",parameters[0])+newline+
				"Radial increment multiplier: "+String.format("%.2f",parameters[1])+newline+
				"Angular increment: "+String.format("%.2f",parameters[2])+newline+
				newline+"PCA parameters"+newline+
				"Number of samples per scale: "+String.format("%d",(int)parameters[3])+newline+
				"Number of components to store: "+String.format("%d",(int)parameters[4])+newline+
				"Number of classes: "+String.format("%d",(int)parameters[5])+newline+
				"Scale factor for spirals: "+String.format("%.2f",parameters[6])+newline+
				"Number of layers in the target: "+String.format("%d",(int)parameters[7])+newline+
				"Sample origin directory: "+orig);
			writer.close();
		}
		catch (Exception e){}
	}	
	
	public static double[][][] pcaComponents(final FileInterpreter FI, String keydir, final int[][] loc, final int[][][][] info, final double[] parameters, int scales, int layers){
		double[][][] pca_val = new double[scales][layers*(int)parameters[4]][];
		// values for the spiral
		double fact = parameters[6];
		// cutoff radius
		double cor = parameters[0];
		// radial increment
		double ri = parameters[1];
		int nf = FI.getBases().length;
		int type;
		// go through all scales to aquire the size of each covariance matrix
		for (int i=0; i<scales; i++){
			int[][] shape = NeighborhoodSample.spiralCoord(parameters[2],cor,ri,1,0.5,6);
			int l = shape.length;
			int counter = 0, counter2 = 0;
			// the covariance of the current scale
			double[][][] covmat = new double[layers][l][l];
			double[][] meanmat = new double[layers][l];
			// go through all images
			for (int j=0; j<nf; j++){
				// if the current set of points is among the keys in the current image
				if (loc[i][counter]<info[1][i][j][0]){
					// load the key points of the current image
					int[][] keys = (int[][])ReadWrite.readObject(keydir+File.separator+FI.getBases()[j]+".key");
					// number of stack group for this image
					int n;
					// check the number of classes
					if ((int)parameters[5]>1){
						n = info[2][i][j].length;
					}
					else {
						n = info[2][i][j].length/layers;
					}
					// go through all stack groups belonging to this image
					for (int k=0; k<n; k++){
						if (loc[i][counter]<info[1][i][j][0]-info[0][i][j][0]+info[3][i][j][k]){
							int w=0, h=0;
							// the case where the image contains class labels
							if ((int)parameters[5]>1){
								ImageStack stack;
								if (TypeReader.isTiff(FI.getNames()[j])){
									stack = (new Opener()).openTiff(FI.getNames()[j],1+k).getImageStack();
								}
								else {
									stack = IJ.openImage(FI.getNames()[j]).getStack();
								}
								w = stack.getWidth();
								h = stack.getHeight();
								double[] im = VectorConv.float2double((float[])stack.getPixels(1));
								int[] point = new int[1];
								counter2 = 0;
								// go through the current set of keys
								while((counter+counter2)<loc[i].length&&loc[i][counter+counter2]<info[1][i][j][0]-info[0][i][j][0]+info[3][i][j][k]){
									point[0] = keys[i][loc[i][counter+counter2]-(info[1][i][j][0]-info[0][i][j][0]+info[3][i][j][k]-info[2][i][j][k])];
									double[] val = NeighborhoodSample.sample2d(point,w,h,shape,0,-1,im)[0];
									for (int o=0; o<l; o++){
										if (val[o]>=0){
											meanmat[(int)val[o]][o]++;
											for (int p=0; p<l; p++){
												if (val[p]==val[o]){
													covmat[(int)val[o]][o][p]++;
												}
											}
										}
									}
									counter2++;
								}
							}
							// case of not class labels
							else {
								for (int m=0; m<layers; m++){
									ImageStack stack;
									if (TypeReader.isTiff(FI.getNames()[j])){
										stack = (new Opener()).openTiff(FI.getNames()[j],1+k*layers+m).getImageStack();
									}
									else {
										stack = IJ.openImage(FI.getNames()[j]).getStack();
									}
									w = stack.getWidth();
									h = stack.getHeight();
									double[] im = VectorConv.float2double((float[])stack.getPixels(1));
									int[] point = new int[1];
									counter2 = 0;
									// go through the current set of keys
									while((counter+counter2)<loc[i].length&&loc[i][counter+counter2]<info[1][i][j][0]-info[0][i][j][0]+info[3][i][j][k]){
										point[0] = keys[i][loc[i][counter+counter2]-(info[1][i][j][0]-info[0][i][j][0]+info[3][i][j][k]-info[2][i][j][k])];
										double[] val = NeighborhoodSample.sample2d(point,w,h,shape,0,0,im)[0];
										for (int o=0; o<l; o++){
											meanmat[m][o]+=val[o];
											VectorFun.addi(covmat[m][o],VectorFun.mult(val,val[o]));
										}
										counter2++;
									}
								}
							}
						}
						counter+=counter2;
					}
				}
			}
			// start calculating eigenvalues and such
			for (int j=0; j<layers; j++){
				VectorFun.multi(meanmat[j],1/parameters[3]);
				// calculate the explicit covariance
				for (int k=0; k<l; k++){
					for (int m=0; m<l; m++){
						covmat[j][k][m] = covmat[j][k][m]/parameters[3]-meanmat[j][k]*meanmat[j][m];
					}
				}
				// eigenvalue decomposition
				DenseDoubleMatrix2D covM = new DenseDoubleMatrix2D(covmat[j]);
				EigenvalueDecomposition eig = new EigenvalueDecomposition(covM);
				// take the number of predetermined components
				// start by taking the eigenvectors corresponding to the largest eigenvectors
				for (int k=0; k<(int)parameters[4]; k++){
					pca_val[i][(int)parameters[4]*j+k] = eig.getV().viewColumn(l-1-k).toArray();
					//pca_val[i][(int)parameters[4]*j+k] = eig.getV().viewColumn(k).toArray();
				}
			}
			ri*=fact;
			cor*=fact;
		}
		return pca_val;
	}
	
	public static void projection(String orig, String keydir, String targetdir){
		// make sure there is a spiral pca file present in the target directory
		if (!(new File(targetdir+File.separator+"pca.ser")).exists()||!(new File(targetdir+File.separator+"spiral_config.ser")).exists()){
			IJ.log("PCA components have not been calculated yet");
			return;
		}
		// store the key file configuration for RF clustering etc.
		ReadWrite.writeObject(targetdir+File.separator+FolderNames.KEYPARAM,keydir);
		// load the spiral configuration parameters
		double[] parameters = (double[])ReadWrite.readObject(targetdir+File.separator+"spiral_config.ser");
		// load the pca components
		double[][][] pca_val = (double[][][])ReadWrite.readObject(targetdir+File.separator+"pca.ser");
		FileInterpreter FI = new FileInterpreter(orig);
		// go through all images in the origin folder
		for (int i=0; i<FI.getBases().length; i++){
			// check if the target key file doesn't already exist
			if (!(new File(targetdir+File.separator+FI.getBases()[i]+".key")).exists()){
				// load the key file
				int[][] keys = (int[][])ReadWrite.readObject(keydir+File.separator+FI.getBases()[i]+".key");
				// pca evaluation at key locations
				double[][] val;
				int n,len;
				if ((int)parameters[5]>1){
					len = (int)parameters[5]*(int)parameters[4];
					val = new double[keys.length*len][];
					n = FI.getInfo()[0][i];
				}
				else {
					len = (int)parameters[7]*(int)parameters[4];
					val = new double[keys.length*len][];
					n = FI.getInfo()[0][i]/(int)parameters[7];
				}
				// create the key vector for all classes and projections
				for (int j=0; j<keys.length; j++){
					for (int k=0; k<len; k++){
						val[j*len+k] = new double[keys[j].length];
					}
				}
				// go through the individual stack collection
				for (int j=0; j<n; j++){
					// variable preparation
					int[] point = new int[1];
					// case with classes
					if ((int)parameters[5]>1){
						ImageStack stack;
						if (TypeReader.isTiff(FI.getNames()[j])){
							stack = (new Opener()).openTiff(FI.getNames()[i],1+j).getImageStack();
						}
						else {
							stack = IJ.openImage(FI.getNames()[i]).getStack();
						}
						int w = stack.getWidth();
						int h = stack.getHeight();
						double[] im = VectorConv.float2double((float[])stack.getPixels(1));
						// scale factor
						double fact = parameters[6];
						// cutoff radius
						double cor = parameters[0];
						// radial increment
						double ri = parameters[1];
						// go through all scales
						for (int k=0; k<keys.length/n; k++){
							// current spiral neighborhood
							int[][] shape = NeighborhoodSample.spiralCoord(parameters[2],cor,ri,1,0.5,6);
							// current key block
							int a = j*keys.length/n+k;
							// go through each key point
							for (int l=0; l<keys[a].length; l++){
								point[0] = keys[a][l];
								double[] t = new double[len];
								double[] spir = NeighborhoodSample.sample2d(point,w,h,shape,0,-1,im)[0];
								// go through each spiral location
								for (int m=0; m<shape.length; m++){
									if (spir[m]>=0){
										// go through each projection
										for (int o=0; o<(int)parameters[4]; o++){
											t[(int)parameters[4]*(int)spir[m]+o]+=pca_val[k][(int)parameters[4]*(int)spir[m]+o][m];
										}
									}
								}
								// assign all the calculated pca projections
								for (int m=0; m<len; m++){
									val[a*len+m][l] = t[m];
								}
							}
							ri*=fact;
							cor*=fact;
						}
					}
					// otherwise in the non class case...
					else {
						for (int p=0; p<(int)parameters[7]; p++){
							ImageStack stack;
							if (TypeReader.isTiff(FI.getNames()[j])){
								stack = (new Opener()).openTiff(FI.getNames()[i],1+j*(int)parameters[7]+p).getImageStack();
							}
							else {
								stack = IJ.openImage(FI.getNames()[i]).getStack();
							}
							int w = stack.getWidth();
							int h = stack.getHeight();
							double[] im = VectorConv.float2double((float[])stack.getPixels(1));
							// scale factor
							double fact = parameters[6];
							// cutoff radius
							double cor = parameters[0];
							// radial increment
							double ri = parameters[1];
							// go through all scales
							for (int k=0; k<keys.length/n; k++){
								// current spiral neighborhood
								int[][] shape = NeighborhoodSample.spiralCoord(parameters[2],cor,ri,1,0.5,6);
								// current key block
								int a = j*keys.length/n+k;
								// go through each key point
								for (int l=0; l<keys[a].length; l++){
									point[0] = keys[a][l];
									double[] spir = NeighborhoodSample.sample2d(point,w,h,shape,0,-1,im)[0];
									for (int o=0; o<(int)parameters[4]; o++){
										double t = 0;
										// go through each spiral location
										for (int m=0; m<shape.length; m++){
											t+=spir[m]*pca_val[k][p*(int)parameters[4]+o][m];
										}
										val[a*len+p*(int)parameters[4]+o][l] = t;
									}
								}
								ri*=fact;
								cor*=fact;
							}
						}
					}
				}
				// store the current collection of pca keys
				ReadWrite.writeObject(targetdir+File.separator+FI.getBases()[i]+".key",val);
			}
		}
	}
}