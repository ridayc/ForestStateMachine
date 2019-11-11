package flib.ij.featureextraction;

import java.util.Arrays;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Random;
import java.io.File;
import java.io.RandomAccessFile;
import flib.io.TypeReader;
import flib.io.ReadWrite;
import flib.ij.io.ImageReader;
import flib.ij.featureextraction.FileInterpreter;
import flib.ij.featureextraction.FolderNames;
import flib.math.VectorConv;
import flib.math.VectorAccess;
import flib.math.SortPair2;
import flib.algorithms.regions.RegionFunctions;
import flib.algorithms.correlative.CovarianceRatios;

// a class which learns typical region geometry using a scaled principle axis aligned version of the region pixels
// a stores different variants
public class RegionFeatures implements
java.io.Serializable {
	private String[] directories;
	private double[] parameters, labelset;
	private TreeSet<SortPair2> labels;
	private int[] lpc;
	// initial training point locations
	private int[][] loc2;
	private double[][] regions_dist, regions_ang, boxes;
	private FileInterpreter FI;
	private double[][] proj;
	
	public RegionFeatures(final String[] directories, final double[] param){
		this.directories = directories.clone();
		this.parameters = param.clone();
		// File interpreter for the input directory
		this.FI = new FileInterpreter(directories[0]);
		// find all unique labels
		// get labels according to provided files
		TreeSet<SortPair2> labels = readLabels(FI,directories[1],directories[2],this.parameters);
		// find how many classes there are
		Iterator<SortPair2> it = labels.iterator();
		int count =0; 
		while(it.hasNext()){
			SortPair2 sp = it.next();
			if (sp.getValue()>-1){
				count++;
			}
		}
		// put the values into an array
		this.lpc = new int[count];
		it = labels.iterator();
		while(it.hasNext()){
			SortPair2 sp = it.next();
			if (sp.getValue()>-1){
				lpc[(int)sp.getValue()] = (int)sp.getOriginalIndex();
			}
		}
		// sample and setup the region boxes
		setupBoxes();
		if (directories[2]!=""){
			//this.cvrp = new CovarianceRatios(boxes,labelset,(int)this.parameters[5],1,0.5);
			CovarianceRatios cvrp = new CovarianceRatios(boxes,labelset,(int)this.parameters[5],2,1e-3);
			//this.cvrp = new CovarianceRatios(boxes,labelset,(int)this.parameters[5],4,1e-6);
			this.proj = cvrp.getEigenvectors((int)parameters[6]);
		}
		else {
			CovarianceRatios cvrp = new CovarianceRatios(boxes,labelset,(int)this.parameters[5],-2,1e-6);
			//this.cvrp = new CovarianceRatios(boxes,labelset,(int)this.parameters[5],-2,1e-2);
			this.proj = cvrp.getEigenvectors((int)parameters[6]);
		}
	}
	
	public static TreeSet<SortPair2> readLabels(final FileInterpreter FI, String orig, String classdir, final double[] parameters){
		TreeSet<SortPair2> labels = new TreeSet<SortPair2>();
		for (int i=0; i<FI.getNames().length; i++){
			// find the regions
			String clfolder = "";
			double[][] keys = (double[][])ReadWrite.readObject(orig+File.separator+FI.getBases()[i]+".key");
			double[][] keys2 = new double[0][0];
			if (classdir!=""){
				keys2 = (double[][])ReadWrite.readObject(classdir+File.separator+FI.getBases()[i]+".key");
			}
			int layers = keys.length;
			// go through all regions
			for (int j=0; j<layers; j++){
				// go through all regions in this layer
				for (int k=0; k<keys[j].length; k++){
					// orig needs to contain the region sizes!!!
					if (parameters[0]>0){
						if (keys[j][k]<parameters[0]){
							SortPair2 sp = new SortPair2(-1,1);
							// check if the current image label is already in the list
							if (labels.contains(sp)){
								// set the counter of labels of this type one higher
								sp = labels.floor(sp);
								sp.setOriginalIndex(sp.getOriginalIndex()+1);
							}
							else {
								// otherwise put the new label into the list
								labels.add(sp);
							}
						}
						// if there is a classification dire
						else if (classdir!=""){
							SortPair2 sp = new SortPair2(keys2[j][k],1);
							// check if the current image label is already in the list
							if (labels.contains(sp)){
								// set the counter of labels of this type one higher
								sp = labels.floor(sp);
								sp.setOriginalIndex(sp.getOriginalIndex()+1);
							}
							else {
								// otherwise put the new label into the list
								labels.add(sp);
							}
						}
						else {
							SortPair2 sp = new SortPair2(0,1);
							// check if the current image label is already in the list
							if (labels.contains(sp)){
								// set the counter of labels of this type one higher
								sp = labels.floor(sp);
								sp.setOriginalIndex(sp.getOriginalIndex()+1);
							}
							else {
								// otherwise put the new label into the list
								labels.add(sp);
							}
						}
					}
					else if (classdir!=""){
						SortPair2 sp = new SortPair2(keys2[j][k],1);
						// check if the current image label is already in the list
						if (labels.contains(sp)){
							// set the counter of labels of this type one higher
							sp = labels.floor(sp);
							sp.setOriginalIndex(sp.getOriginalIndex()+1);
						}
						else {
							// otherwise put the new label into the list
							labels.add(sp);
						}
					}
					else {
						SortPair2 sp = new SortPair2(0,1);
						// check if the current image label is already in the list
						if (labels.contains(sp)){
							// set the counter of labels of this type one higher
							sp = labels.floor(sp);
							sp.setOriginalIndex(sp.getOriginalIndex()+1);
						}
						else {
							// otherwise put the new label into the list
							labels.add(sp);
						}
					}
				}
			}
		}
		return labels;
	}
			
	public void setupBoxes(){
		this.loc2 = new int[lpc.length][(int)parameters[1]];
		double[][] regions_dist = new double[lpc.length*(int)parameters[1]][];
		double[][] regions_ang = new double[lpc.length*(int)parameters[1]][];
		this.labelset = new double[lpc.length*(int)parameters[1]];
		Random rng = new Random();
		// random sampling locations according to class
		for (int i=0; i<lpc.length; i++){
			for (int j=0; j<(int)parameters[1]; j++){
				loc2[i][j] = rng.nextInt(lpc[i]);
				labelset[i*(int)parameters[1]+j] = i;
			}
			Arrays.sort(loc2[i]);
		}
		// current label counter for all classes
		int[] counter = new int[lpc.length];
		// counter for the location in the sampling point lists
		int[] counter2 = new int[lpc.length];
		// go through all images
		for (int i=0; i<FI.getNames().length; i++){
			// find the regions
			double[][] keys = (double[][])ReadWrite.readObject(directories[1]+File.separator+FI.getBases()[i]+".key");
			double[][] keys2 = new double[0][0];
			if (directories[2]!=""){
				keys2 = (double[][])ReadWrite.readObject(directories[2]+File.separator+FI.getBases()[i]+".key");
			}
			int layers = keys.length;
			// for all substacks
			String filename = directories[3]+File.separator+FI.getBases()[i]+TypeReader.TIFF[0];
			String filename2 = directories[4]+File.separator+FI.getBases()[i]+TypeReader.TIFF[0];
			String filename3 = directories[5]+File.separator+FI.getBases()[i]+TypeReader.TIFF[0];
			for (int j=0; j<layers; j++){
				double[] im = VectorConv.float2double(ImageReader.tiffLayerArray(filename,j));
				double[] im2 = VectorConv.float2double(ImageReader.tiffLayerArray(filename2,j));
				double[] im3 = VectorConv.float2double(ImageReader.tiffLayerArray(filename3,j));
				int[][] reg = RegionFunctions.getRegions(im);
				// go through all regions
				for (int k=0; k<keys[j].length; k++){
					if (parameters[0]>0){
						if (keys[j][k]>=parameters[0]){
							if (directories[2]!=""){
								int a = (int)keys2[j][k];
								// if we found a region of interest
								if (loc2[a][counter2[a]]==counter[a]){
									while (loc2[a][counter2[a]]==counter[a]){
										int b = a*(int)parameters[1]+counter2[a];
										regions_dist[b] = new double[reg[k].length];
										regions_ang[b] = new double[reg[k].length];
										for (int m=0; m<reg[k].length; m++){
											regions_dist[b][m] = im2[reg[k][m]];
											regions_ang[b][m] = im3[reg[k][m]];
										}
										counter2[a]++;
										// make sure the counter isn't too large for final comparisons
										if (counter2[a]==loc2[a].length){
											counter2[a]--;
											break;
										}
									}
								}
								// adjust the counter for this class
								counter[a]++;
							}
							else {
								int a = 0;
								if (loc2[a][counter2[a]]==counter[a]){
									while (loc2[a][counter2[a]]==counter[a]){
										int b = a*(int)parameters[1]+counter2[a];
										regions_dist[b] = new double[reg[k].length];
										regions_ang[b] = new double[reg[k].length];
										for (int m=0; m<reg[k].length; m++){
											regions_dist[b][m] = im2[reg[k][m]];
											regions_ang[b][m] = im3[reg[k][m]];
										}
										counter2[a]++;
										// make sure the counter isn't too large for final comparisons
										if (counter2[a]==loc2[a].length){
											counter2[a]--;
											break;
										}
									}
								}
								// adjust the counter for this class
								counter[a]++;
							}
						}
					}
					else if (directories[2]!=""){
						int a = (int)keys2[j][k];
						// if we found a region of interest
						if (loc2[a][counter2[a]]==counter[a]){
							while (loc2[a][counter2[a]]==counter[a]){
								int b = a*(int)parameters[1]+counter2[a];
								regions_dist[b] = new double[reg[k].length];
								regions_ang[b] = new double[reg[k].length];
								for (int m=0; m<reg[k].length; m++){
									regions_dist[b][m] = im2[reg[k][m]];
									regions_ang[b][m] = im3[reg[k][m]];
								}
								counter2[a]++;
								// make sure the counter isn't too large for final comparisons
								if (counter2[a]==loc2[a].length){
									counter2[a]--;
									break;
								}
							}
						}
						// adjust the counter for this class
						counter[a]++;
					}
					else {
						int a = 0;
						if (loc2[a][counter2[a]]==counter[a]){
							while (loc2[a][counter2[a]]==counter[a]){
								int b = a*(int)parameters[1]+counter2[a];
								regions_dist[b] = new double[reg[k].length];
								regions_ang[b] = new double[reg[k].length];
								for (int m=0; m<reg[k].length; m++){
									regions_dist[b][m] = im2[reg[k][m]];
									regions_ang[b][m] = im3[reg[k][m]];
								}
								counter2[a]++;
								// make sure the counter isn't too large for final comparisons
								if (counter2[a]==loc2[a].length){
									counter2[a]--;
									break;
								}
							}
						}
						// adjust the counter for this class
						counter[a]++;
					}
				}
			}
		}
		// calculate the boxes for all selected regions
		this.boxes = RegionFunctions.boxConversion(regions_dist,regions_ang,(int)parameters[2],(int)parameters[3],parameters[4]);
		// add a bit of noise to help the dimension reduction
		/*
		double d = 0.01;
		for (int i=0; i<boxes.length; i++){
			for (int j=0; j<boxes[i].length; j++){
				boxes[i][j]+=rng.nextGaussian()*d;
			}
		}
		*/
		// set values outside of the p-quantile to values within the p-quantile of each dimension
		int p1 = (int)((int)parameters[1]*parameters[7]);
		int p2 = (int)((int)parameters[1]*(1-parameters[7])-1);
		for (int k=0; k<boxes[0].length; k++){
			for (int i=0; i<lpc.length; i++){
				double[] vec = new double[(int)parameters[1]];
				for (int j=0; j<vec.length; j++){
					int a = i*vec.length+j;
					vec[j] = boxes[a][k];
				}
				double[] vec2 = vec.clone();
				Arrays.sort(vec2);
				for (int j=0; j<vec.length; j++){
					int a = i*vec.length+j;
					if (boxes[a][k]<vec2[p1]||boxes[a][k]>vec2[p2]){
						boxes[a][k] = vec2[rng.nextInt(p2-p1+1)+p1];
					}
				}
			}
		}
	}
	
	public void applyProjections(String regdir, String reg_dist, String reg_ang, String target){
		// get all the relevant file information
		FileInterpreter FI = new FileInterpreter(regdir);
		int[][] info = FI.getInfo();
		// make sure the target folder exists
		new File(target).mkdirs();
		// make to store the key location file in the folder
		ReadWrite.writeObject(target+File.separator+FolderNames.KEYPARAM,regdir);
		// go through all the image files in the base directory
		for (int i=0; i<info[0].length; i++){
			if (!(new File(target+File.separator+FI.getBases()[i]+".key")).exists()){
				String filename = FI.getNames()[i];
				String filename2 = reg_dist+File.separator+FI.getBases()[i]+TypeReader.TIFF[0];
				String filename3 = reg_ang+File.separator+FI.getBases()[i]+TypeReader.TIFF[0];
				double[][] keys = new double[proj.length*FI.getInfo()[0][i]][];
				// go through all subimages
				for (int j=0; j<FI.getInfo()[0][i]; j++){
					double[] im = VectorConv.float2double(ImageReader.tiffLayerArray(filename,j));
					double[] im2 = VectorConv.float2double(ImageReader.tiffLayerArray(filename2,j));
					double[] im3 = VectorConv.float2double(ImageReader.tiffLayerArray(filename3,j));
					int[][] reg = RegionFunctions.getRegions(im);
					double[][] regions_dist = new double[reg.length][];
					double[][] regions_ang = new double[reg.length][];
					for (int k=0; k<reg.length; k++){
						regions_dist[k] = VectorAccess.access(im2,reg[k]);
						regions_ang[k] = VectorAccess.access(im3,reg[k]);
					}
					double[][] temp = RegionFunctions.applyBoxTemplate(proj,regions_dist,regions_ang,(int)parameters[2],(int)parameters[3],parameters[4]);
					for (int m=0; m<proj.length; m++){
						keys[j*proj.length+m] = new double[reg.length];
						for (int k=0; k<reg.length; k++){
							keys[j*proj.length+m][k] = temp[k][m];
						}
					}
				}
				String name = target+File.separator+FI.getBases()[i]+".key";
				// store the keys in a file
				ReadWrite.writeObject(name,keys);
			}
		}
	}
	
	public double[][] getProjections(){
		double[][] copy = new double[proj.length][proj[0].length];
		for (int i=0; i<proj.length; i++){
			copy[i] = proj[i].clone();
		}
		return copy;
	}
}