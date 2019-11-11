package flib.ij.segmentation;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.Math;
import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Arrays;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.VectorAccess;
import flib.math.SortPair2;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.featureextraction.FolderNames;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.splitfunctions.GiniClusterSplit;
import flib.ij.stack.StackTo32bit;
import flib.ij.stack.StackOperations;
import flib.datastructures.TypeConversion;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.io.Opener;
import ij.io.FileInfo;
import ij.io.TiffDecoder;

public class Classifier implements
java.io.Serializable {
	// folders used to get the feature vector
	// first folder is the base directory for the whole procedure
	// second folder is the location of the training data, 
	// third folder is the target storage location for the classifier
	// all further folders contain feature components
	private String[] folders;
	// the random forest classifier;
	private RandomForest RF;
	// start folder...
	private int sf = 4;
	// training type
	private int type = 0;
	
	public Classifier(final String[] folders, final int[][] traininglocations, final double[] labels, final double[] rf_parameters, final double[] splitpurity, int ntree){
		this.folders = new String[folders.length];
		for (int i=0; i<folders.length; i++){
			this.folders[i] = new String(folders[i]);
		}
		// the first folder contains the base directory
		// all other directories are in relative position to this directory
		String base = folders[0];
		String source = base+File.separator+folders[1];
		String target = base+File.separator+folders[2];
		// make the target directory
		new File(target).mkdirs();
		// list all files in the training directory
		File[] directoryListing = (new File(source)).listFiles();
		// get all relevant image files
		int[] file_number = TypeReader.fileNumber(directoryListing);
		int np = 0;
		for (int i=0; i<traininglocations.length; i++){
			np+=traininglocations[i].length;
		}
		// find all images
		// configuration files for the random forest
		// ... for prosperities sake?
		String configtxt = target+File.separator+"config.txt";
		writeConfigTXT(configtxt,rf_parameters,splitpurity,new int[]{np,ntree});
		// temporary training set
		ArrayList<ArrayList<Double>> temp = new ArrayList<ArrayList<Double>>();
		for (int i=0; i<np; i++){
			temp.add(new ArrayList<Double>());
		}
		int start = 0;
		// go through all images in the training folder
		for (int i=0; i<file_number.length; i++){
			String name = directoryListing[file_number[i]].getName();
			// according to image add training points to the training set
			addTrainingSet(name,traininglocations[i],start,temp);
			start+=traininglocations[i].length;
		}
		double[][] trainingset = (double[][])TypeConversion.cpArrayList(temp,2,"[D");
		boolean[] categorical = new boolean[trainingset[0].length];
		for (int i=0; i<categorical.length; i++){
			boolean m = true;
			for (int j=0; j<trainingset.length; j++){
				if (trainingset[j][i]!=(int)trainingset[j][i]){
					m = false;
				}
			}
			categorical[i] = m;
		}
		// create the random forest classifier
		this.RF = new RandomForest(trainingset,labels,VectorFun.add(new double[labels.length],1),categorical, VectorFun.add(new double[trainingset[0].length],1),rf_parameters,splitpurity,new GiniClusterSplit(),ntree);
	}
	
	public static void run(final String[] folders, int np, final double[] rf_parameters, final double[] splitpurity, int ntree){
		run(folders,np,rf_parameters,splitpurity,ntree,0);
	}
	
	// create and store a classifier object using specified folders
	public static void run(final String[] folders, int np, final double[] rf_parameters, final double[] splitpurity, int ntree, int type){
		String source = folders[0]+File.separator+folders[1];
		// get the label counts from the training files
		Storage st;
		int sf = 4;
		if (type==0){
			st = readLabels(source);
		}
		else {
			st = readLabels2(source,folders[0]+File.separator+folders[sf-1],type);
		}
		// training locations according to class
		int[][] locations = new int[st.getCounts().length][np];
		if (type==1){
			int c = st.getCounts().length/2;
			for (int i=0;i<c; i++){
				if (st.getCounts()[2*i+1]==0){
					locations[2*i] = new int[2*np];
					locations[2*i+1] = new int[0];
				}
			}
		}
		else if (type!=0) {
			int c = (int)Math.sqrt(st.getCounts().length);
			int[] temp = new int[c];
			for (int i=0; i<st.getCounts().length; i++){
				if (st.getCounts()[i]==0){
					temp[i%c]++;
				}
			}
			for (int i=0; i<st.getCounts().length; i++){
				if (st.getCounts()[i]==0){
					locations[i] = new int[0];
				}
				else {
					locations[i] = new int[np*(c-1)/(c-1-temp[i%c])];
				}
			}
			for (int i=0; i<c; i++){
				if (temp[i]!=(c-1)){
					locations[i*i] = new int[(c-1)*np];
				}
				else {
					locations[i*i] = new int[(c-1)*np*2];
				}
			}
		}
		Random rng = new Random();
		File[] directoryListing = (new File(source)).listFiles();
		int[] file_number = TypeReader.fileNumber(directoryListing);
		// create the locations according to class
		for (int i=0; i<locations.length; i++){
			for (int j=0; j<locations[i].length; j++){
				locations[i][j] = rng.nextInt(st.getCounts()[i]);
			}
			if (locations[i].length>0){
				Arrays.sort(locations[i]);
			}
		}
		int l = 0;
		for (int i=0; i<st.getCounts().length; i++){
			l+=locations[i].length;
		}
		double[] labels = new double[l];
		ArrayList<ArrayList<Integer>> traininglocations = new ArrayList<ArrayList<Integer>>();
		int count = 0;
		int[] counter = new int[labels.length];
		int[] current = new int[labels.length];
		for (int i=0; i<st.getImage().length; i++){
			String name = directoryListing[file_number[i]].getAbsolutePath();
			String name2 = directoryListing[file_number[i]].getName();
			name2 = folders[0]+File.separator+folders[sf-1]+File.separator+TypeReader.imageBase(name2)+TypeReader.TIFF[0];
			traininglocations.add(new ArrayList<Integer>());
			// function to run through images and find which 
			// training points are contained in this image
			count = trainingLocations(name,name2,type,locations,counter,current,count,traininglocations.get(i),labels,st.getLabels());
		}
		int[][] tl = (int[][])TypeConversion.cpArrayList(traininglocations,2,"[I");
		Classifier cl = new Classifier(folders,tl,labels,rf_parameters,splitpurity,ntree);
		// save the classifier in the target directory which should
		// have been created with the classifier
		String target = folders[0]+File.separator+folders[2];
		String RFCL = target+File.separator+"classifier.ser";
		ReadWrite.writeObject(RFCL,cl);
	}
	
	public void applyClassifier(int chunk_size){
		// application of the classifier
		// chunk size lets all regulate how many feature vectors are in 
		// memory at any point
		String base = folders[0];
		String target = base+File.separator+folders[2];
		int nclass = RF.getNumClasses();
		String[] targets = new String[nclass];
		// make the directories for the individual class votes
		for (int i=0; i<nclass; i++){
			targets[i] = base+File.separator+folders[2]+String.format("%d",i);
			new File(targets[i]).mkdirs();
		}
		File[] directoryListing = (new File(folders[0])).listFiles();
		int[] file_number = TypeReader.fileNumber(directoryListing);
		// for all images in the base directory
		for (int i=0; i<file_number.length; i++){
			String name = directoryListing[file_number[i]].getName();
			String target_name = target+File.separator+TypeReader.imageBase(name)+TypeReader.TIFF[0];
				if (!(new File(target_name)).exists()){
					apply2Image(name,chunk_size);
				}
		}
	}
	
	private static int trainingLocations(final String filename, final String filename2, int type, int[][] locations, int[] counter, int[] current, int count, ArrayList<Integer> traininglocations, double[] labels, final TreeSet<SortPair2> l){
		// open the original image file and convert it (if neccessary)
		ImagePlus imp = StackTo32bit.convert(IJ.openImage(filename));
		ImageStack stack = imp.getImageStack();
		// go through all images in the original (if there are more than one)
		// there should not be more than one image layer for training!!!
		double[] x = VectorConv.float2double((float[])(stack.getProcessor(1).convertToFloat().getPixels()));
		double[] y = new double[0];
		if (type>0){
			ImagePlus imp2 = StackTo32bit.convert(IJ.openImage(filename2));
			ImageStack stack2 = imp2.getImageStack();
			y = VectorConv.float2double((float[])(stack2.getProcessor(1).convertToFloat().getPixels()));
		}
		int c = l.size();
		for (int j=0; j<x.length; j++){
			// check if the pixel has a class label
			if (l.contains(new SortPair2(x[j],0))){
				// find the translation of the class label to an integer value
				int a = (int)l.floor(new SortPair2(x[j],0)).getOriginalIndex();
				if (type==1){
					if (a!=y[j]){
						a+=c;
					}
				}
				else if (type!=0){
					a+=c*y[j];
				}
				// in case the current marker for the class matches 
				// the next value among the random samples
				while (locations[a][current[a]]==counter[a]){
					// add the pixel to the list of training points for this image
					traininglocations.add(j);
					// add the pixel to the whole label list
					// that's why we require several different counters
					labels[count] = a%c;
					count++;
					current[a]++;
					if (current[a]==locations[a].length){
						current[a] = locations[a].length-1;
						break;
					}
				}
				// counter for the encountered number of labelled pixels
				counter[a]++;
			}
		}
		return count;
	}
	
	private static void writeConfigTXT(final String name, final double[] parameters, final double[] splitpurity, final int[] misc){
		try {
			File infotxt = new File(name);
			BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
			String newline = System.getProperty("line.separator");
			writer.write("Random forest parameters"+newline+
				"Number of training points: "+String.format("%d",misc[0])+newline+
				"Number of classes: "+String.format("%d",(int)parameters[0])+newline+
				"Number of trees: "+String.format("%d",misc[1])+newline+
				"Mtry: "+String.format("%d",(int)parameters[1])+newline+
				"Maximum tree depth: "+String.format("%d",(int)parameters[2])+newline+
				"Maximum leaf size: "+String.format("%d",(int)parameters[3])+newline+
				"Mtry shuffling type: "+String.format("%d",(int)parameters[4])+newline+
				"Fraction of zero crossing and gini: "+String.format("%.2f",parameters[5])+newline+
				"Transition matrix diagonal: "+String.format("%.2f",parameters[6])+newline+
				"Transition matrix noise injection: "+String.format("%.2f",parameters[7])+newline);
			for (int i=0; i<splitpurity.length; i++){
				writer.write("Split purity class "+String.format("%d",i)+": "+String.format("%.2f",splitpurity[i])+newline);
			}
			writer.close();
		}
		catch (Exception e){}
	}
	
	private void addTrainingSet(final String filename, final int[] locations, int start, ArrayList<ArrayList<Double>> temp){
		// again we assume there was only one layer per training image
		String base = folders[0];
		// feature files are always tiff files
		String tiffname = TypeReader.imageBase(filename)+TypeReader.TIFF[0];
		for (int i=sf; i<folders.length; i++){
			String name = base+File.separator+folders[i]+File.separator+tiffname;
			//IJ.log(name);
			ImagePlus imp = StackTo32bit.convert(IJ.openImage(name));
			int n = imp.getNSlices();
			ImageStack stack = imp.getImageStack();
			for (int j=0; j<n; j++){
				double[] x = VectorAccess.access(VectorConv.float2double((float[])(stack.getProcessor(1+j).convertToFloat().getPixels())),locations);
				for (int k=0; k<locations.length; k++){
					temp.get(start+k).add(x[k]);
				}
			}
		}
	}
	
	private void apply2Image(final String filename,int chunk_size){
		apply2Image(filename,chunk_size,Double.NaN);
	}
	
	private void apply2Image(final String filename,int chunk_size, double val){
		String base = folders[0];
		String rename = TypeReader.imageBase(filename)+TypeReader.TIFF[0];
		String orig = folders[0]+File.separator+filename;
		// unlike the previous training cases
		// images the classifier is applied to are allowed to have multile layers
		int n = IJ.openImage(orig).getNSlices();
		int w = IJ.openImage(orig).getWidth();
		int h = IJ.openImage(orig).getHeight();
		// number of random forest classes
		int nclass = RF.getNumClasses();
		// number of feature dimensions
		int d = RF.getCategorical().length;
		// stack for class predictions
		ImageStack stack2 = new ImageStack(w,h);
		// array of stacks for class votes
		ArrayList<ImageStack> stack3 = new ArrayList<ImageStack>();
		for (int i=0; i<nclass; i++){
			stack3.add(new ImageStack(w,h));
		}
		// for all layers of the original image
		for (int m=0; m<n; m++){
			// use the image to only classify certain locations
			float[] y = new float[0];
			if (!Double.isNaN(val)){
				String name = base+File.separator+folders[sf-1]+File.separator+rename;
				y = (float[])(new Opener()).openTiff(name,m+1).getImageStack().getPixels(1);
			}
			boolean check = true;
			// predicted classes
			double[] classes = new double[w*h];
			double[][] votes = new double[nclass][w*h];
			int l = 0;
			int counter = 0;
			// while different chunks are still being run through
			while (check){
				// points for the current chunk
				double[][] points = new double[chunk_size][d];
				int[] loc = VectorFun.add(new int[chunk_size],-1);
				int k = 0;
				while (k<chunk_size&&counter<w*h){
					if (!Double.isNaN(val)){
						if (y[counter]==val){
							loc[k] = counter;
							k++;
						}
					}
					else {
						loc[k] = counter;
						k++;
					}
					counter++;
				}
				int count = 0;
				for (int i=sf; i<folders.length; i++){
					String name = base+File.separator+folders[i]+File.separator+rename;
					//IJ.log(name);
					//ImagePlus imp = StackTo32bit.convert(IJ.openImage(name));
					// fairly complicated tiff file reading...
					TiffDecoder td = new TiffDecoder((new File(name)).getParent(),(new File(name)).getName());
					FileInfo[] info = null;
					try {
						info = td.getTiffInfo();
					} catch (IOException e) {
						String msg = e.getMessage();
						if (msg==null||msg.equals("")) msg = ""+e;
						IJ.error("Open TIFF", msg);
					}
					FileInfo fi = info[0];
					int n3 = 0;
					if (info.length==1 && fi.nImages>1) {
						n3 = fi.nImages/n;
					}
					else {
						n3 = info.length/n;
					}
					//ImagePlus imp = IJ.openImage(name);
					//int n2 = imp.getNSlices();
					// the actual number of chunks we run through depends
					// on the number of images in the original image
					//int n3 = n2/n;
					//ImageStack stack = imp.getImageStack();
					for (int j=n3*m; j<n3*(m+1); j++){
						//double[] x = VectorConv.float2double((float[])(stack.getProcessor(1+j).convertToFloat().getPixels()));
						ImageStack stack = (new Opener()).openTiff(name,j+1).getImageStack();
						k = 0;
						while (k<chunk_size){
							if (loc[k]!=-1){
								double x = ((float[])stack.getPixels(1))[loc[k]];
								//points[k-l][count] = x[k];
								points[k][count] = x;
							}
							k++;
						}
						count++;
					}
				}
				// apply the classifier
				for (int i=0; i<chunk_size; i++){
					/*if (i==0){
						IJ.log(Arrays.toString(points[0]));
					}*/
					if (loc[i]!=-1){
						double[] v = RF.applyForest(points[i]);
						for (int j=0; j<nclass; j++){
							votes[j][loc[i]] = v[j];
							classes[loc[i]] = VectorFun.max(v)[1];
						}
					}
				}
				if (counter==w*h){
					check = false;
				}
			}
			// add the current class and votes images to the stack
			stack2.addSlice("None", new FloatProcessor(w,h,classes));
			for (int i=0; i<nclass; i++){
				stack3.get(i).addSlice("None", new FloatProcessor(w,h,votes[i]));
			}
		}
		// store files in the appropriate locations
		ImagePlus imp2 = new ImagePlus("Classes",stack2);
		imp2.getProcessor().setMinAndMax(0,nclass-1);
		IJ.saveAsTiff(imp2,base+File.separator+folders[2]+File.separator+rename);
		for (int i=0; i<nclass; i++){
			ImagePlus imp3 = new ImagePlus("Votes",stack3.get(i));
			imp3.getProcessor().setMinAndMax(0,1);
			IJ.saveAsTiff(imp3,base+File.separator+folders[2]+String.format("%d",i)+File.separator+rename);
		}
	}
	
	// getter functions
	public RandomForest getForest(){
		return this.RF;
	}
	
	public String[] getFolders(){
		return this.folders;
	}
	
	// function to translate the labels in the image to sorted integer values
	private static Storage readLabels(final String foldername){
		// get all files in the target directory
		File[] directoryListing = (new File(foldername)).listFiles();
		int[] file_number = TypeReader.fileNumber(directoryListing);
		TreeSet<SortPair2> labels = new TreeSet<SortPair2>();
		for (int i=0; i<file_number.length; i++){
			String name = directoryListing[file_number[i]].getAbsolutePath();
			courseImage(name,labels);
		}
		Storage st = new Storage();
		st.setCounts(new int[labels.size()-1]);
		st.setImage(new int[file_number.length][labels.size()-1]);
		Iterator<SortPair2> it = labels.iterator();
		st.setLabels(new TreeSet<SortPair2>());
		int count = 0;
		while (it.hasNext()){
			SortPair2 sp = it.next();
			if (count!=labels.size()-1){
				st.getLabels().add(new SortPair2(sp.getValue(),count));
				st.getCounts()[count] = (int)sp.getOriginalIndex();
				count++;
			}
		}
		for (int i=0; i<file_number.length; i++){
			String name = directoryListing[file_number[i]].getAbsolutePath();
			recourseImage(name,st.getLabels(),st.getImage()[i]);
		}
		return st;
	}
	
	private static void courseImage(final String filename, TreeSet<SortPair2> labels){
		// open the original image file and convert it (if neccessary)
		ImagePlus imp = StackTo32bit.convert(IJ.openImage(filename));
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		// go through all images in the original (if there are more than one)
		for (int i=1; i<n+1; i++){
			double[] x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			for (int j=0; j<x.length; j++){
				SortPair2 sp = new SortPair2(x[j],1);
				if (labels.contains(sp)){
					sp = labels.floor(sp);
					sp.setOriginalIndex(sp.getOriginalIndex()+1);
				}
				else {
					labels.add(sp);
				}
			}
		}
	}
	
	private static void recourseImage(final String filename, TreeSet<SortPair2> labels, int[] counts){
		// open the original image file and convert it (if neccessary)
		ImagePlus imp = StackTo32bit.convert(IJ.openImage(filename));
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		// go through all images in the original (if there are more than one)
		for (int i=1; i<n+1; i++){
			double[] x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			for (int j=0; j<x.length; j++){
				if (labels.contains(new SortPair2(x[j],0))){
					counts[(int)labels.floor(new SortPair2(x[j],0)).getOriginalIndex()]++;
				}
			}
		}
	}
	
	// function to translate the labels in the image to sorted integer values
	private static Storage readLabels2(final String foldername, final String foldername2, int type){
		// get all files in the target directory
		File[] directoryListing = (new File(foldername)).listFiles();
		int[] file_number = TypeReader.fileNumber(directoryListing);
		TreeSet<SortPair2> labels = new TreeSet<SortPair2>();
		for (int i=0; i<file_number.length; i++){
			String name = directoryListing[file_number[i]].getAbsolutePath();
			courseImage(name,labels);
		}
		Storage st = new Storage();
		int l = 0;
		if (type==1){
			l = (labels.size()-1)*2;
		}
		else {
			l = (labels.size()-1)*(labels.size()-1);
		}
		st.setCounts(new int[l]);
		st.setImage(new int[file_number.length][l]);
		Iterator<SortPair2> it = labels.iterator();
		st.setLabels(new TreeSet<SortPair2>());
		int count = 0;
		while (it.hasNext()){
			SortPair2 sp = it.next();
			if (count!=labels.size()-1){
				st.getLabels().add(new SortPair2(sp.getValue(),count));
				st.getCounts()[count] = (int)sp.getOriginalIndex();
				count++;
			}
		}
		for (int i=0; i<file_number.length; i++){
			String name = directoryListing[file_number[i]].getAbsolutePath();
			String name2 = directoryListing[file_number[i]].getName();
			name2 = foldername2+File.separator+TypeReader.imageBase(name2)+TypeReader.TIFF[0];
			recourseImage2(name,name2,st.getLabels(),st.getImage()[i],type);
		}
		int[] counts = new int[l];
		for (int i=0; i<st.getImage().length; i++){
			for (int j=0; j<l; j++){
				counts[j]+=st.getImage()[i][j];
			}
		}
		st.setCounts(counts);
		return st;
	}
	
	private static void recourseImage2(final String filename, final String filename2, TreeSet<SortPair2> labels, int[] counts, int type){
		// open the original image file and convert it (if neccessary)
		ImagePlus imp = StackTo32bit.convert(IJ.openImage(filename));
		ImageStack stack = imp.getImageStack();
		ImagePlus imp2 = StackTo32bit.convert(IJ.openImage(filename2));
		ImageStack stack2 = imp2.getImageStack();
		int l = labels.size();
		// go through all images in the original (if there are more than one)
		double[] x = VectorConv.float2double((float[])(stack.getProcessor(1).convertToFloat().getPixels()));
		// classified image
		double[] y = VectorConv.float2double((float[])(stack2.getProcessor(1).convertToFloat().getPixels()));
		for (int j=0; j<x.length; j++){
			if (labels.contains(new SortPair2(x[j],0))){
				int a = (int)labels.floor(new SortPair2(x[j],0)).getOriginalIndex();
				if (type==1){
					if (a==y[j]){
						counts[a]++;
					}
					else {
						counts[a+l]++;
					}
				}
				else {
					counts[a+l*(int)y[j]]++;
				}
			}
		}
	}
	
	// class for variable storage
	// and yes... it is really bad pratice to make this subclass static...
	// this means that the run function of this classifier class should
	// never be run simultaneously!!!
	private static class Storage {
		private TreeSet<SortPair2> label_conversion;
		private int[] counts;
		private int[][] image_label;
		
		public TreeSet<SortPair2>getLabels(){
			return label_conversion;
		}
		
		public int[] getCounts(){
			return counts;
		}
		
		public int[][] getImage(){
			return image_label;
		}
		
		public void setLabels(TreeSet<SortPair2> label_conversion){
			this.label_conversion = label_conversion;
		}
		
		public void setCounts(int[] counts){
			this.counts = counts;
		}
		
		public void setImage(final int[][] image_label){
			this.image_label = new int[image_label.length][];
			for (int i=0; i<image_label.length; i++){
				this.image_label[i] = image_label[i].clone();
			}
		}
	}
}