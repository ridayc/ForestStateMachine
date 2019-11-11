package flib.ij.segmentation;

import java.io.File;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Arrays;
import flib.math.SortPair2;
import flib.math.RankSort;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.io.TypeReader;
import flib.io.ReadWrite;
import flib.ij.io.ImageReader;
import flib.ij.featureextraction.FolderNames;
import flib.ij.featureextraction.FileInterpreter;
import flib.ij.featureextraction.MultiFileExtraction;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.ForestFunctions;
import flib.algorithms.randomforest.splitfunctions.MixedSplit;
import flib.algorithms.randomforest.splitfunctions.VarianceSplit;
import flib.algorithms.clustering.RFC;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class Classifier implements
java.io.Serializable {
	// folder containing all label files
	private String labeldir;
	// all folder used for classification
	private String[] folders;
	// folder containing classified versions of the image
	// this is used for second and deeper level classifiers
	private String classdir;
	// unique labels
	private TreeSet<SortPair2> labels;
	// label translations
	private TreeSet<SortPair2> lt;
	// number of pixels per label class
	private int[] lpc;
	// initial training point locations
	private int[] loc;
	// the random forest classifier
	private RandomForest RF;
	// in case we want RF clustering afterwards
	private RFC[] cluster;
	// set up parameters
	private double[] parameters;
	// parameters for the randomforest
	private double[] rf_param;
	// clustering oobe
	private double[] oobe;
	
	public Classifier(String labeldir, final String[] folders, String classdir, final double[] parameters, final double[] rf_param, final double[] splitpurity){
		this.labeldir = labeldir;
		this.folders = folders.clone();
		this.classdir = classdir;
		this.parameters = parameters.clone();
		this.rf_param = rf_param.clone();
		// File interpreter for the label directory
		FileInterpreter FI = new FileInterpreter(labeldir);
		// find all unique labels
		// create the base label treeset
		TreeSet<SortPair2> labels = new TreeSet<SortPair2>();
		this.lpc = new int[0];
		// labels for the random forest or clustering
		double[] labelset = new double[0];
		// if not in the clustering case
		if (parameters[0]>0||!classdir.equals("")){
			this.labels = readLabels(FI);
			// give a renaming of the labels
			this.lt = new TreeSet<SortPair2>();
			this.lpc = translateLabels(parameters[0]==2||parameters[0]<0,this.lt,this.labels);
			// also covers one of the clustering cases
			if (parameters[0]<=2){
				this.loc = new int[lpc.length*(int)parameters[2]];
				labelset = simpleTraining(FI,(int)parameters[2],this.lpc,this.loc,this.lt);
			}
			else {
				// otherwise we are reclassifying misclassified data
				this.loc = new int[lpc.length*(int)parameters[2]*2];
				labelset = correctionTraining(FI,classdir,(int)parameters[2],lpc.length,lpc,loc,lt);
			}
		}
		// clustering cases
		else {
			this.loc = new int[(int)parameters[2]];
			clusterTraining(FI,(int)parameters[2],this.loc,countPixels(FI));
		}
		// extract the training vector based on the training locations
		double[][] trainingset = MultiFileExtraction.trainingExtraction(labeldir,this.folders,FI,loc);
		// dimensions which only contain integers during training should be
		// considered categorical
		boolean[] categorical = new boolean[trainingset[0].length];
		for (int i=0; i<categorical.length; i++){
			categorical[i] = true;
		}
		for (int i=0; i<trainingset.length; i++){
			for (int j=0; j<categorical.length; j++){
				if (trainingset[i][j]!=(int)trainingset[i][j]){
					categorical[j] = false;
				}
			}
		}
		// create the explicit random forest
		if (parameters[0]>0){
			this.RF = new RandomForest(trainingset,labelset,VectorFun.add(new double[labelset.length],1),categorical,VectorFun.add(new double[categorical.length],1),rf_param,splitpurity,new MixedSplit(),(int)parameters[1]);
		}
		// clustering cases
		else{
			// supervised clustering
			if (!classdir.equals("")){
				RandomForest temp = ((Classifier)ReadWrite.readObject(classdir+File.separator+"classifier.ser")).getForest();
				this.oobe = temp.outOfBagError();
				int[][] leaves = temp.getLeafIndices(trainingset);
				RFC rfc = new RFC(leaves,(int)rf_param[0],temp.getTreeSizes());
				labelset = VectorConv.int2double(rfc.assignCluster(leaves));
				this.RF = new RandomForest(trainingset,labelset,VectorFun.add(new double[labelset.length],1),categorical,VectorFun.add(new double[categorical.length],1),rf_param,splitpurity,new MixedSplit(),(int)parameters[1]);
			}
			// artificial split
			else if (parameters[0]==0){
				double[][] set2 = ForestFunctions.randomizedSet(trainingset,categorical,1,parameters[3]);
				double[][] set = new double[(int)parameters[2]*2][trainingset[0].length];
				labelset = new double[(int)parameters[2]*2];
				for (int i=0; i<(int)parameters[2]; i++){
					set[i] = trainingset[i].clone();
					set[i+(int)parameters[2]] = set2[i].clone();
					labelset[i+(int)parameters[2]] = 1;
				}
				double nc = this.rf_param[0];
				this.rf_param[0] = 2;
				RandomForest temp = new RandomForest(set,labelset,VectorFun.add(new double[labelset.length],1),categorical,VectorFun.add(new double[categorical.length],1),this.rf_param,new double[]{1,1},new MixedSplit(),(int)parameters[1]);
				this.oobe = temp.outOfBagError();
				int[][] leaves = temp.getLeafIndices(trainingset);
				RFC rfc = new RFC(leaves,(int)rf_param[0],temp.getTreeSizes());
				labelset = VectorConv.int2double(rfc.assignCluster(leaves));
				this.rf_param[0] = nc;
				this.RF = new RandomForest(trainingset,labelset,VectorFun.add(new double[labelset.length],1),categorical,VectorFun.add(new double[categorical.length],1),this.rf_param,splitpurity,new MixedSplit(),(int)parameters[1]);
			}
			// variance split
			else {
				labelset = new double[(int)parameters[2]];
				double nc = this.rf_param[0];
				this.rf_param[0] = 1;
				RandomForest temp = new RandomForest(trainingset,labelset,VectorFun.add(new double[trainingset.length],1),new boolean[categorical.length],VectorFun.add(new double[categorical.length],1),rf_param,splitpurity,new VarianceSplit(),(int)parameters[1]);
				int[][] leaves = temp.getLeafIndices(trainingset);
				RFC rfc = new RFC(leaves,(int)rf_param[0],temp.getTreeSizes());
				labelset = VectorConv.int2double(rfc.assignCluster(leaves));
				this.rf_param[0] = nc;
				this.RF = new RandomForest(trainingset,labelset,VectorFun.add(new double[labelset.length],1),categorical,VectorFun.add(new double[categorical.length],1),rf_param,splitpurity,new MixedSplit(),(int)parameters[1]);
			}
		}
	}
	
	public static TreeSet<SortPair2> readLabels(final FileInterpreter FI){
		// create the base label treeset
		TreeSet<SortPair2> labels = new TreeSet<SortPair2>();
		// go through all images in the label directory
		for (int i=0; i<FI.getNames().length; i++){
			// for all substacks
			for (int j=0; j<FI.getInfo()[0][i]; j++){
				float[] im;
				String filename = FI.getNames()[i];
				if (TypeReader.isTiff(filename)){
					im = ImageReader.tiffLayerArray(filename,j);
				}
				else {
					im = ImageReader.imageArray(filename);
				}
				// go through all image pixels
				for (int k=0; k<im.length; k++){
					SortPair2 sp = new SortPair2((double)im[k],1);
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
		return labels;
	}
	
	public static int countPixels(final FileInterpreter FI){
		int count = 0;
		// go through all images in the label directory
		for (int i=0; i<FI.getNames().length; i++){
			// for all substacks
			for (int j=0; j<FI.getInfo()[0][i]; j++){
				String filename = FI.getNames()[i];
				if (TypeReader.isTiff(filename)){
					count+=ImageReader.tiffLayerArray(filename,j).length;
				}
				else {
					count+=ImageReader.imageArray(filename).length;
				}
			}
		}
		return count;
	}
	
	public static int[] translateLabels(boolean comp, TreeSet<SortPair2> lt, TreeSet<SortPair2> labels){
		int top = labels.size()-1;
		if (comp){
			top++;
		}
		int[] lpc = new int[top];
		Iterator<SortPair2> it = labels.iterator();
		int count =0; 
		while(it.hasNext()){
			SortPair2 sp = it.next();
			if (count!=top){
				lt.add(new SortPair2(sp.getValue(),count));
				lpc[count] = (int)sp.getOriginalIndex();
			}
			count++;
		}
		return lpc;
	}
	
	public static int[][] classTransitions(final FileInterpreter FI, String classdir, int lp, TreeSet<SortPair2> lt){
		int[][] ct = new int[lp][lp];
		// go through all images in the label directory
		// as well as all images in the classification directory
		for (int i=0; i<FI.getNames().length; i++){
			// for all substacks
			for (int j=0; j<FI.getInfo()[0][i]; j++){
				float[] im;
				float[] im2;
				String filename = FI.getNames()[i];
				if (TypeReader.isTiff(filename)){
					im = ImageReader.tiffLayerArray(filename,j);
				}
				else {
					im = ImageReader.imageArray(filename);
				}
				filename = classdir+File.separator+FI.getBases()[i]+TypeReader.TIFF[0];
				im2 = ImageReader.tiffLayerArray(filename,j);
				// go through all pixels and see which were labeled according to which class
				for (int k=0; k<im.length; k++){
					SortPair2 sp = new SortPair2(im[k],0);
					if (sp.compareTo(lt.last())<=0){
						ct[(int)im2[k]][(int)lt.floor(sp).getOriginalIndex()]++;
					}
				}
			}
		}
		return ct;
	}
	
	public static double[] simpleTraining(final FileInterpreter FI, int np, final int[] lpc, int[] loc, final TreeSet<SortPair2> lt){
		// in this case we are simply sampling an equal number of points from each class
		int[][] loc2 = new int[lpc.length][np];
		double[] labelset = new double[lpc.length*np];
		// random number generator for sampling
		Random rng = new Random();
		// random sampling locations according to class
		for (int i=0; i<lpc.length; i++){
			for (int j=0; j<np; j++){
				loc2[i][j] = rng.nextInt(lpc[i]);
				labelset[i*np+j] = i;
			}
			Arrays.sort(loc2[i]);
		}
		// current label counter for all classes
		int[] counter = new int[lpc.length];
		// counter for the location in the sampling point lists
		int[] counter2 = new int[lpc.length];
		// go through all images
		for (int i=0; i<FI.getNames().length; i++){
			// for all substacks
			for (int j=0; j<FI.getInfo()[0][i]; j++){
				float[] im;
				String filename = FI.getNames()[i];
				if (TypeReader.isTiff(filename)){
					im = ImageReader.tiffLayerArray(filename,j);
				}
				else {
					im = ImageReader.imageArray(filename);
				}
				// current image offset
				int offset = FI.getInfo()[2][i]-FI.getInfo()[1][i]*(FI.getInfo()[0][i]-j);
				// go through all image pixels
				for (int k=0; k<im.length; k++){
					// check if the current pixel is a label pixel
					SortPair2 sp = new SortPair2(im[k],0);
					if (sp.compareTo(lt.last())<=0){
						int a = (int)lt.floor(sp).getOriginalIndex();
						// here is where the magic goes...
						if (loc2[a][counter2[a]]==counter[a]){
							// the while loop handles the case where the 
							// same label location is found multiple times
							// within the training locations list
							while (loc2[a][counter2[a]]==counter[a]){
								loc[a*np+counter2[a]] = k+offset;
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
		// reorder the locations and labels
		RankSort rs = new RankSort(VectorConv.int2double(loc),labelset);
		double[] temp = rs.getSorted();
		for (int i=0; i<temp.length; i++){
			loc[i] = (int)temp[i];
		}
		labelset = rs.getDRank();
		return labelset;
	}
	
	public static void clusterTraining(final FileInterpreter FI, int np, int[] loc, int count){
		// random number generator for sampling
		Random rng = new Random();
		// random sampling locations according to class
		for (int i=0; i<np; i++){
			loc[i] = rng.nextInt(count);
		}
		Arrays.sort(loc);
	}
	
	public static double[] correctionTraining(final FileInterpreter FI, String classdir, int np, int lp, final int[] lpc, int[] loc, final TreeSet<SortPair2> lt){
		// we sample an equal number of samples from each class
		// one portion correctly classified, one portion incorrectly classified
		int[][] loc2 = new int[lp*2][np];
		double[] labelset = new double[lp*np*2];
		// random number generator for sampling
		Random rng = new Random();
		// class transition matrix
		int[][] mat = classTransitions(FI,classdir,lp,lt);
		int[] lpc2 = new int[lp];
		for (int i=0; i<lp; i++){
			lpc2[i] = mat[i][i];
		}
		// count the number of points
		// random sampling locations according to class
		for (int i=0; i<lp; i++){
			for (int j=0; j<np; j++){
				// correctly labelled pixels
				if (lpc2[i]>0){
					loc2[2*i][j] = rng.nextInt(lpc2[i]);
				}
				else {
					loc2[2*i][j] = rng.nextInt(lpc[i]-lpc2[i]);
				}
				// incorrectly labelled pixels
				if (lpc[i]-lpc2[i]>0){
					loc2[2*i+1][j] = rng.nextInt(lpc[i]-lpc2[i]);
				}
				else {
					loc2[2*i+1][j] = rng.nextInt(lpc2[i]);
				}
				labelset[2*i*np+j] = i;
				labelset[(2*i+1)*np+j] = i;
			}
			Arrays.sort(loc2[2*i]);
			Arrays.sort(loc2[2*i+1]);
		}
		
		// current label counter for all classes
		int[] counter = new int[loc2.length];
		// counter for the location in the sampling point lists
		int[] counter2 = new int[loc2.length];
		// go through all images
		for (int i=0; i<FI.getNames().length; i++){
			// for all substacks
			for (int j=0; j<FI.getInfo()[0][i]; j++){
				float[] im;
				float[] im2;
				String filename = FI.getNames()[i];
				String filename2 = classdir+File.separator+FI.getBases()[i]+TypeReader.TIFF[0];
				if (TypeReader.isTiff(filename)){
					im = ImageReader.tiffLayerArray(filename,j);
					im2 = ImageReader.tiffLayerArray(filename2,j);
				}
				else {
					im = ImageReader.imageArray(filename);
					im2 = ImageReader.tiffLayerArray(filename2,j);
				}
				// current image offset
				int offset = FI.getInfo()[2][i]-FI.getInfo()[1][i]*(FI.getInfo()[0][i]-j);
				// go through all image pixels
				for (int k=0; k<im.length; k++){
					// check if the current pixel is a label pixel
					SortPair2 sp = new SortPair2(im[k],0);
					if (sp.compareTo(lt.last())<=0){
						int a = (int)lt.floor(sp).getOriginalIndex();
						// here is where the magic goes...
						int s;
						if (a==(int)im2[k]){
							// correctly labelled
							s = 0;
						}
						else {
							// incorrectly labelled
							s = 1;
						}
						int b = 2*a+s;
						if (loc2[b][counter2[b]]==counter[b]){
							// the while loop handles the case where the 
							// same label location is found multiple times
							// within the training locations list
							while (loc2[b][counter2[b]]==counter[b]){
								loc[b*np+counter2[b]] = k+offset;
								counter2[b]++;
								// make sure the counter isn't too large for final comparisons
								if (counter2[b]==loc2[b].length){
									counter2[b]--;
									break;
								}
							}
							
						}
						if (lpc2[a]<=0){
							int b2 = 2*a;
							if (loc2[b2][counter2[b2]]==counter[b]){
								// the while loop handles the case where the 
								// same label location is found multiple times
								// within the training locations list
								while (loc2[b2][counter2[b2]]==counter[b]){
									loc[b2*np+counter2[b2]] = k+offset;
									counter2[b2]++;
									// make sure the counter isn't too large for final comparisons
									if (counter2[b2]==loc2[b2].length){
										counter2[b2]--;
										break;
									}
								}
								
							}
						}
						else if (lpc[a]-lpc2[a]<=0){
							int b2 = 2*a+1;
							if (loc2[b2][counter2[b2]]==counter[b]){
								// the while loop handles the case where the 
								// same label location is found multiple times
								// within the training locations list
								while (loc2[b2][counter2[b2]]==counter[b]){
									loc[b2*np+counter2[b2]] = k+offset;
									counter2[b2]++;
									// make sure the counter isn't too large for final comparisons
									if (counter2[b2]==loc2[b2].length){
										counter2[b2]--;
										break;
									}
								}
								
							}
						}
						// adjust the counter for this class
						counter[b]++;
					}
				}
			}
		}
		// reorder the locations and labels
		RankSort rs = new RankSort(VectorConv.int2double(loc),labelset);
		double[] temp = rs.getSorted();
		for (int i=0; i<temp.length; i++){
			loc[i] = (int)temp[i];
		}
		labelset = rs.getDRank();
		return labelset;
	}
	
	public void classifyImage(String basename, final String[] folders, String target, int chunk_size, int n, int w, int h){
		// create a stack for the random forest votes
		ImageStack stack = new ImageStack(w,h);
		// a stack for the clusters
		ImageStack stack2 = new ImageStack(w,h);
		int dim = this.RF.getCategorical().length;
		int nclass = RF.getNumClasses();
		int wh = w*h;
		for (int i=0; i<n; i++){
			int start = 0;
			int stop = 0;
			double[][] st = new double[nclass][w*h];
			double[] z = new double[w*h];
			double[] ind = new double[w*h];
			long a, b;
			while (start<wh){
				stop = start+chunk_size;
				if (stop>wh){
					stop = wh;
				}
				a = System.nanoTime();
				double[][] chunk = MultiFileExtraction.imageChunk(basename,folders,start,chunk_size,n,i,wh,dim);
				b = System.nanoTime();
				System.out.println(Double.toString((b-a)/1e6));
				// classify the chunk
				a = System.nanoTime();
				double[][] y = this.RF.applyForest(chunk);
				b = System.nanoTime();
				System.out.println(Double.toString((b-a)/1e6));
				System.out.println("Next chunk");
				// put the chunk into the image holder
				for (int j=0; j<nclass; j++){
					for (int k=0; k<y.length; k++){
						st[j][start+k] = y[k][j];
						// get the maximum vote to find the predicted class
						if (z[start+k]<st[j][start+k]){
							z[start+k] = st[j][start+k];
							ind[start+k] = j;
						}
					}
				}
				start = stop;
			}
			for (int j=0; j<nclass; j++){
				stack.addSlice("None", new FloatProcessor(w,h,st[j]));
			}
			stack2.addSlice("None", new FloatProcessor(w,h,ind));
		}
		// save the stacks
		ImagePlus imp = new ImagePlus("Class Votes",stack);
		imp.getProcessor().setMinAndMax(0,1);
		IJ.saveAsTiff(imp,target+File.separator+FolderNames.VOTES+File.separator+basename+TypeReader.TIFF[0]);
		ImagePlus imp2 = new ImagePlus("Classes",stack2);
		imp2.getProcessor().setMinAndMax(0,nclass-1);
		IJ.saveAsTiff(imp2,target+File.separator+basename+TypeReader.TIFF[0]);
	}
	
	public void classifyFolder(String basefolder, final String[] folders, String target, int chunk_size){
		// get all the relevant file information
		FileInterpreter FI = new FileInterpreter(basefolder);
		int[][] info = FI.getInfo();
		// make sure the target folder exists
		new File(target).mkdirs();
		new File(target+File.separator+FolderNames.VOTES).mkdirs();
		// go through all the image files in the base directory
		for (int i=0; i<info[0].length; i++){
			int[] size = ImageReader.getImageSize(FI.getNames()[i]);
			if (!(new File(target+File.separator+FI.getBases()[i]+TypeReader.TIFF[0])).exists()){
				classifyImage(FI.getBases()[i],folders,target,chunk_size,size[0],size[1],size[2]);
			}
		}
	}
	
	public RandomForest getForest(){
		return this.RF;
	}
	
	public double[] clusterOobe(){
		return this.oobe;
	}
	
	public void clusterSetup(final int[] nclust, double balance, int maxit){
		this.cluster = new RFC[nclust.length];
		for (int i=0; i<nclust.length; i++){
			this.cluster[i] = new RFC(RF.getLeafIndices(RF.getTrainingset()),nclust[i],RF.getTreeSizes(),balance,maxit);
		}
	}
	
	public void clusterImage(String basename, final String[] folders, String target, int chunk_size, int n, int w, int h){
		// create a stack for the cluster indices
		ImageStack stack = new ImageStack(w,h);
		int dim = this.RF.getCategorical().length;
		int wh = w*h;
		double max = 0;
		for (int i=0; i<cluster.length; i++){
			if (cluster[i].getSizes().length>max){
				max = cluster[i].getSizes().length;
			}
		}
		max--;
		for (int i=0; i<n; i++){
			int start = 0;
			int stop = 0;
			double[][] z = new double[cluster.length][w*h];
			double[][] ind = new double[cluster.length][w*h];
			long a, b;
			while (start<wh){
				stop = start+chunk_size;
				if (stop>wh){
					stop = wh;
				}
				a = System.nanoTime();
				double[][] chunk = MultiFileExtraction.imageChunk(basename,folders,start,chunk_size,n,i,wh,dim);
				b = System.nanoTime();
				System.out.println(Double.toString((b-a)/1e6));
				// classify the chunk
				a = System.nanoTime();
				int[][] indices = this.RF.getLeafIndices(chunk);
				for (int j=0; j<cluster.length; j++){
					double[][] y = this.cluster[j].getDist(indices);
					// put the chunk into the image holder
					for (int k=0; k<cluster[j].getSizes().length; k++){
						for (int l=0; l<y.length; l++){
							// get the maximum distance to find the cluster
							if (z[j][start+l]<y[l][k]){
								z[j][start+l] = y[l][k];
								ind[j][start+l] = k;
							}
						}
					}
				}
				start = stop;
				b = System.nanoTime();
				System.out.println(Double.toString((b-a)/1e6));
				System.out.println("Next chunk");
			}
			for (int j=0; j<cluster.length; j++){
				stack.addSlice("None", new FloatProcessor(w,h,ind[j]));
			}
		}
		// save the stacks
		ImagePlus imp = new ImagePlus("Clusters",stack);
		imp.getProcessor().setMinAndMax(0,max);
		IJ.saveAsTiff(imp,target+File.separator+basename+TypeReader.TIFF[0]);
	}
	
	public void clusterFolder(String basefolder, final String[] folders, String target, int chunk_size){
		// check that the cluster has been initialized
		if (cluster==null){
			IJ.log("No can do. You need to first initialize the cluster.");
			return;
		}
		// get all the relevant file information
		FileInterpreter FI = new FileInterpreter(basefolder);
		int[][] info = FI.getInfo();
		// make sure the target folder exists
		new File(target).mkdirs();
		// go through all the image files in the base directory
		for (int i=0; i<info[0].length; i++){
			int[] size = ImageReader.getImageSize(FI.getNames()[i]);
			if (!(new File(target+File.separator+FI.getBases()[i]+TypeReader.TIFF[0])).exists()){
				clusterImage(FI.getBases()[i],folders,target,chunk_size,size[0],size[1],size[2]);
			}
		}
	}
	
	public RFC[] getCluster(){
		return this.cluster;
	}
	
}