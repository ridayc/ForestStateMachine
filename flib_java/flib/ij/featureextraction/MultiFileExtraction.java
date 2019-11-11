package flib.ij.featureextraction;

import java.util.ArrayList;
import java.io.File;
import java.io.RandomAccessFile;
import flib.io.TypeReader;
import flib.io.ReadWrite;
import flib.ij.io.ImageReader;
import flib.ij.featureextraction.FileInterpreter;
import flib.ij.featureextraction.FolderNames;
import flib.math.VectorConv;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.datastructures.TypeConversion;

public class MultiFileExtraction {
	public static double[][] trainingExtraction(String origin, final String[] folders, final FileInterpreter FI, final int[] loc){
		// info for the files to be worked on
		// prepare a temporary vector collection to store the training set
		ArrayList<ArrayList<Double>> temp = new ArrayList<ArrayList<Double>>();
		for (int i=0; i<loc.length; i++){
			temp.add(new ArrayList<Double>());
		}
		int[][] info = FI.getInfo();
		int counter = 0, counter2 = 0;
		// go through all images
		for (int i=0; i<info[2].length; i++){
			// if the current set of points is among the pixels of the current image
			if (loc[counter]<info[2][i]){
				// go through all image stacks belonging to this image
				for (int j=0; j<info[0][i]; j++){
					counter2 = 0;
					if (loc[counter]<info[2][i]-(info[0][i]-j-1)*info[1][i]){
						// time to delve through some image files
						// let's go through all folders...
						for (int k=0; k<folders.length; k++){
							// let's first check that there aren't any key
							// location related files
							if ((new File(folders[k]+File.separator+FolderNames.KEYPARAM)).exists()){
								String wsfolder = (String)ReadWrite.readObject(folders[k]+File.separator+FolderNames.KEYPARAM);
								int[][] keys = (int[][])ReadWrite.readObject(wsfolder+File.separator+FI.getBases()[i]+".key");
								// ->the number of scales is equal to the number of key arrays
								int layers = keys.length;
								// get the region contents for the current image
								double[][] content = (double[][])ReadWrite.readObject(folders[k]+File.separator+FI.getBases()[i]+".key");
								// number of layers per scale
								int lps = content.length/keys.length;
								for (int l=0; l<layers; l++){
									// load the region mapping
									float[] im = ImageReader.tiffLayerArray(wsfolder+File.separator+FI.getBases()[i]+TypeReader.TIFF[0],l);
									for (int m=0; m<lps; m++){
										counter2 = 0;
										while((counter+counter2)<loc.length&&loc[counter+counter2]<info[2][i]-(info[0][i]-j-1)*info[1][i]){
											int a = loc[counter+counter2]-(info[2][i]-(info[0][i]-j)*info[1][i]);
											temp.get(counter+counter2).add(content[l*lps+m][(int)im[a]]);
											counter2++;
										}
									}			
								}
							}
							// we could be sampling according to a spiral sampling
							else if ((new File(folders[k]+File.separator+FolderNames.SPIRALPARAM)).exists()){
								String base = (String)ReadWrite.readObject(folders[k]+File.separator+FolderNames.SPIRALORIG)+File.separator+FI.getBases()[i];
								// is the image a tif file?
								//
								// For this case we assume we are being provided a tif file!!!
								//
								String name = base+TypeReader.TIFF[0];
								// spiral parameters
								double[] parameters = (double[])ReadWrite.readObject(folders[k]+File.separator+FolderNames.SPIRALPARAM);
								// spiral neighborhood
								int[][] shape = NeighborhoodSample.spiralCoord(parameters[2],parameters[0],parameters[1],parameters[5],parameters[4],(int)parameters[6]);
								int[] point = new int[1];
								int size[] = ImageReader.getTiffSize(name);
								// check how many layers there are in the target tiff
								int layers = size[0]/info[0][i];
								// go through all these layers and start adding
								// training points
								for (int l=0; l<layers; l++){
									// load the current image
									// +1 for IJ already incorporated
									double[] im = VectorConv.float2double(ImageReader.tiffLayerArray(name,j*layers+l));
									// go through all location pixels and add them
									// to the training set
									counter2 = 0;
									while((counter+counter2)<loc.length&&loc[counter+counter2]<info[2][i]-(info[0][i]-j-1)*info[1][i]){
										point[0] = loc[counter+counter2]-(info[2][i]-(info[0][i]-j)*info[1][i]);
										double[] spir = NeighborhoodSample.sample2d(point,size[1],size[2],shape,0,parameters[3],im)[0];
										for (int m = 0; m<spir.length; m++){
											temp.get(counter+counter2).add(spir[m]);
										}
										counter2++;
									}
								}
							}
							// otherwise we are sampling from a watershed region collection
							// with key points and sampling according to a scale file
							else {
								// open up the image file of interest
								String base = folders[k]+File.separator+FI.getBases()[i];
								// is the image a tif file?
								String name = base+TypeReader.TIFF[0];
								if ((new File(name)).exists()){
									// check how many layers there are in the target tiff
									int n = ImageReader.getTiffSize(name)[0];
									int layers = n/info[0][i];
									// go through all these layers and start adding
									// training points
									for (int l=0; l<layers; l++){
										// load the current image
										// +1 for IJ already incorporated
										float[] im = ImageReader.tiffLayerArray(name,j*layers+l);
										// go through all location pixels and add them
										// to the training set
										counter2 = 0;
										while((counter+counter2)<loc.length&&loc[counter+counter2]<info[2][i]-(info[0][i]-j-1)*info[1][i]){
											int a = loc[counter+counter2]-(info[2][i]-(info[0][i]-j)*info[1][i]);
											temp.get(counter+counter2).add((double)im[a]);
											counter2++;
										}
									}
								}
								// for other image formats
								else {
									for (int l=0; l<TypeReader.IMAGES.length; l++){
										name = base+TypeReader.IMAGES[l];
										if ((new File(name)).exists()){
											float[] im = ImageReader.imageArray(name);
											counter2 = 0;
											while(loc[counter+counter2]<info[2][i]-(info[0][i]-j-1)*info[1][i]){
												int a = loc[counter+counter2]-info[2][i]-(info[0][i]-j)*info[1][i];
												temp.get(counter+counter2).add((double)im[a]);
												counter2++;
											}
											break;
										}
									}
								}
							}
						}
					}
					// increase the counter by the number of processed training pixels
					counter+=counter2;
				}
			}
		}
		// convert the training data to a 2D double array
		return (double[][])TypeConversion.cpArrayList(temp,2,"[D");
	}
	
	public static int createTempFile(String tempname, String basename, final String[] folders, int chunk_size, int n, int layer, int wh, int dim) throws Exception{
		// go through all folders
		int counter = 0;
		int cc = 0;
		// create a memory mapped file to write all our images into
		RandomAccessFile raf = new RandomAccessFile(tempname,"rw");
		// we already know how large the file is going to be
		// the input should be a long value
		raf.setLength(((long)wh)*((long)dim));
		for (int k=0; k<folders.length; k++){
			// let's first check that there aren't any key location related files
			if ((new File(folders[k]+File.separator+FolderNames.KEYPARAM)).exists()){
				String wsfolder = (String)ReadWrite.readObject(folders[k]+File.separator+FolderNames.KEYPARAM);
				int[][] keys = (int[][])ReadWrite.readObject(wsfolder+File.separator+basename+".key");
				// ->the number of scales is equal to the number of key arrays
				int layers = keys.length;
				// get the region contents for the current image
				double[][] content = (double[][])ReadWrite.readObject(folders[k]+File.separator+basename+".key");
				// number of layers per scale
				int lps = content.length/keys.length;
				for (int l=0; l<layers; l++){
					// load the region mapping
					float[] im = ImageReader.tiffLayerArray(wsfolder+File.separator+basename+".tif",l);
					for (int m=0; m<lps; m++){
						int start = 0;
						int stop = 0;
						int chunk = 0;
						while (stop<wh){
							stop+=chunk_size;
							if (stop>wh){
								stop = wh;
							}
							int len = stop-start;
							double[] temp = new double[len];
							for(int o=0; o<len; o++){
								temp[o] = content[l*lps+m][(int)im[start+o]];
							}
							// set the writing location
							long loc = (((long)start)*((long)dim)+((long)counter)*((long)len))*8;
							raf.seek(loc);
							// save the current array to memory
							raf.write(VectorConv.double2byte(temp));
							start = stop;
							chunk++;
						}
						cc = chunk;
						counter++;
					}
				}
			}
			// we could be sampling according to a spiral sampling
			else if ((new File(folders[k]+File.separator+FolderNames.SPIRALPARAM)).exists()){
				String base = (String)ReadWrite.readObject(folders[k]+File.separator+FolderNames.SPIRALORIG)+File.separator+basename;
				// is the image a tif file?
				//
				// For this case we assume we are being provided a tif file!!!
				//
				String name = base+TypeReader.TIFF[0];
				// spiral parameters
				double[] parameters = (double[])ReadWrite.readObject(folders[k]+File.separator+FolderNames.SPIRALPARAM);
				// spiral neighborhood
				int[][] shape = NeighborhoodSample.spiralCoord(parameters[2],parameters[0],parameters[1],1,parameters[4],6);
				int[] point = new int[1];
				int size[] = ImageReader.getTiffSize(name);
				// check how many layers there are in the target tiff
				int set = size[0]/n;
				// go through all these layers and start adding
				// training points
				for (int l=0; l<set; l++){
					// load the current image
					// +1 for IJ already incorporated
					double[] im = VectorConv.float2double(ImageReader.tiffLayerArray(name,set*layer+l));
					// go through all location pixels and add them
					// to the training set
					int start = 0;
					int stop = 0;
					int chunk = 0;
					while (stop<wh){
						stop+=chunk_size;
						if (stop>wh){
							stop = wh;
						}
						int len = stop-start;
						double[][] temp = new double[shape.length][len];
						for (int m=0; m<len; m++){
							point[0] = start+m;
							double[] spir = NeighborhoodSample.sample2d(point,size[1],size[2],shape,0,parameters[3],im)[0];
							for (int p = 0; p<spir.length; p++){
								temp[p][m] = spir[p];
							}
						}
						for (int p=0; p<shape.length; p++){
							// set the writing location
							long loc = (((long)start)*((long)dim)+((long)(counter+p))*((long)len))*8;
							raf.seek(loc);
							// save the current array to memory
							raf.write(VectorConv.double2byte(temp[p]));
						}
						start = stop;
						chunk++;
					}
					cc = chunk;
					counter+=shape.length;
				}
			}
			// otherwise we are sampling from a watershed region collection
			// with key points and sampling according to a scale file
			else {
				// open up the image file of interest
				// is the image a tif file?
				String name = folders[k]+File.separator+basename+TypeReader.TIFF[0];
				if ((new File(name)).exists()){
					// check how many layers per original layer
					// there are in the target tiff
					int set = ImageReader.getTiffSize(name)[0]/n;
					// read out the image of interest
					for (int l=0; l<set; l++){
						// +1 for IJ already incorporated
						float[] im = ImageReader.tiffLayerArray(name,set*layer+l);
						// go through all location pixels and add them
						// to the temporary set
						int start = 0;
						int stop = 0;
						int chunk = 0;
						while (stop<wh){
							stop+=chunk_size;
							if (stop>wh){
								stop = wh;
							}
							int len = stop-start;
							double[] temp = new double[len];
							for (int m=0; m<len; m++){
								temp[m] =(double)im[start+m];
							}
							// set the writing location
							long loc = (((long)start)*((long)dim)+((long)counter)*((long)len))*8;
							raf.seek(loc);
							// save the current array to memory
							raf.write(VectorConv.double2byte(temp));
							start = stop;
							chunk++;
						}
						cc = chunk;
						counter++;
					}
				}
				else {
					for (int l=0; l<TypeReader.IMAGES.length; l++){
						name = folders[k]+File.separator+basename+TypeReader.IMAGES[l];
						if ((new File(name)).exists()){
							float[] im = ImageReader.imageArray(name);
							int start = 0;
							int stop = 0;
							int chunk = 0;
							while (stop<wh){
								stop+=chunk_size;
								if (stop>wh){
									stop = wh;
								}
								int len = stop-start;
								double[] temp = new double[len];
								for (int m=0; m<len; m++){
									temp[m] = (double)im[start+m];
								}
								// set the writing location
								long loc = (((long)start)*((long)dim)+((long)counter)*((long)len))*8;
								raf.seek(loc);
								// save the current array to memory
								raf.write(VectorConv.double2byte(temp));
								start = stop;
								chunk++;
							}
							cc = chunk;
							counter++;
							break;
						}
					}
				}
			}
		}
		raf.close();
		return cc;
	}
	
	public static double[][] getChunk(String tempname, int start, int stop, int dim) throws Exception{
		int len = stop-start;
		// open the file for reading
		RandomAccessFile raf = new RandomAccessFile(tempname,"r");
		// prepare the output array
		double[][] temp = new double[len][dim];
		for (int i=0; i<dim; i++){
			// prepare the byte array for reading
			byte[] b = new byte[len*8];
			long loc = (((long)start)*((long)dim)+((long)i)*((long)len))*8;
			raf.seek(loc);
			raf.read(b);
			double[] tmp = VectorConv.byte2double(b);
			for (int j=0; j<len; j++){
				temp[j][i] = tmp[j];
			}
		}
		raf.close();
		return temp;
	}
	
	public static void deleteTempFile(String tempname){
		if(!(new File(tempname)).delete()){
			System.out.println("Problem deleting file: "+tempname);
		}
	}
	
	public static double[][] imageChunk(String basename, final String[] folders, int start, int chunk_size, int n, int layer, int wh, int dim){
		// stop marker
		int stop = start+chunk_size;
		// is at most the size of a single image file
		if (stop>wh){
			stop=wh;
		}
		int len = stop-start;
		// tempory storage for the output values
		double[][] temp = new double[len][dim];
		// go through all folders
		int counter = 0;
		for (int k=0; k<folders.length; k++){
			// let's first check that there aren't any key location related files
			if ((new File(folders[k]+File.separator+FolderNames.KEYPARAM)).exists()){
				String wsfolder = (String)ReadWrite.readObject(folders[k]+File.separator+FolderNames.KEYPARAM);
				int[][] keys = (int[][])ReadWrite.readObject(wsfolder+File.separator+basename+".key");
				// ->the number of scales is equal to the number of key arrays
				int layers = keys.length;
				// get the region contents for the current image
				double[][] content = (double[][])ReadWrite.readObject(folders[k]+File.separator+basename+".key");
				// number of layers per scale
				int lps = content.length/keys.length;
				for (int l=0; l<layers; l++){
					// load the region mapping
					double[] im = new double[0];
					try{
						im = ImageReader.tiffLayerArray(wsfolder+File.separator+basename+".tif",l+1,start,stop);
					}
					catch (Throwable t){
						System.out.println("Reading image problem");
						t.printStackTrace();
					}
					for (int m=0; m<lps; m++){
						for(int o=0; o<len; o++){
							temp[o][counter] = content[l*lps+m][(int)im[o]];
						}
						counter++;
					}			
				}
			}
			// we could be sampling according to a spiral sampling
			else if ((new File(folders[k]+File.separator+FolderNames.SPIRALPARAM)).exists()){
				String base = (String)ReadWrite.readObject(folders[k]+File.separator+FolderNames.SPIRALORIG)+File.separator+basename;
				// is the image a tif file?
				//
				// For this case we assume we are being provided a tif file!!!
				//
				String name = base+TypeReader.TIFF[0];
				// spiral parameters
				double[] parameters = (double[])ReadWrite.readObject(folders[k]+File.separator+FolderNames.SPIRALPARAM);
				// spiral neighborhood
				int[][] shape = NeighborhoodSample.spiralCoord(parameters[2],parameters[0],parameters[1],parameters[5],parameters[4],(int)parameters[6]);
				int[] points = new int[len];
				int size[] = ImageReader.getTiffSize(name);
				// check how many layers there are in the target tiff
				int set = size[0]/n;
				// go through all these layers and start adding
				// training points
				for (int l=0; l<set; l++){
					// load the current image
					// +1 for IJ already incorporated
					// we need to load the whole image, since we're also looking at neighboring pixels which aren't in the center point set
					double[] im = VectorConv.float2double(ImageReader.tiffLayerArray(name,set*layer+l));
					// go through all location pixels and add them
					// to the training set
					for (int m=0; m<len; m++){
						points[m] = start+m;
					}
					double[][] spir = NeighborhoodSample.sample2d(points,size[1],size[2],shape,0,parameters[3],im);
					for (int m=0; m<len; m++){
						for (int p = 0; p<shape.length; p++){
							temp[m][counter+p] = spir[m][p];
						}
					}
					counter+=shape.length;
				}
			}
			// otherwise we are sampling from a watershed region collection
			// with key points and sampling according to a scale file
			else {
				// open up the image file of interest
				// is the image a tif file?
				String name = folders[k]+File.separator+basename+TypeReader.TIFF[0];
				if ((new File(name)).exists()){
					// check how many layers per original layer
					// there are in the target tiff
					int set = ImageReader.getTiffSize(name)[0]/n;
					// read out the image of interest
					for (int l=0; l<set; l++){
						// +1 for IJ already incorporated
						double[] im = new double[0];
						try{
							im = ImageReader.tiffLayerArray(name,set*layer+l+1,start,stop);
						}
						catch (Throwable t){
							System.out.println("Reading image problem");
							t.printStackTrace();
						}
						// go through all location pixels and add them
						// to the temporary set
						for (int m=0; m<len; m++){
							temp[m][counter] = im[m];
						}
						counter++;
					}
				}
				else {
					for (int l=0; l<TypeReader.IMAGES.length; l++){
						name = folders[k]+File.separator+basename+TypeReader.IMAGES[l];
						if ((new File(name)).exists()){
							float[] im = ImageReader.imageArray(name);
							for (int m=0; m<len; m++){
								temp[m][counter] = (double)im[start+m];
							}
							counter++;
							break;
						}
					}
				}
			}
		}
		// convert the training data to a 2D double array
		return temp;
	}
}