package flib.ij.featureextraction;

import java.io.File;
import java.util.Random;
import java.util.Arrays;
import java.util.TreeSet;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.math.RankSort;
import flib.math.SortPair2;
import flib.math.random.Shuffle;
import flib.io.TypeReader;
import flib.ij.io.ImageReader;
import flib.ij.featureextraction.FolderNames;
import flib.ij.featureextraction.FileInterpreter;
import flib.ij.featureextraction.MultiFileExtraction;
import flib.ij.segmentation.Classifier;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.clustering.RFC3;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class Clustering2 implements
java.io.Serializable {
	// all folder used for classification
	private String[] folders;
	// folder containing classified versions of the image
	// this is used for second and deeper level classifiers
	private String rawdir;
	// label folders, if any
	private String[] labeldir;
	// initial training point locations
	private int[] loc;
	// the random forest classifier
	private RFC3 cluster;

	
	public Clustering2(String rawdir, final String[] labeldir, final String[] folders, int np, final double[] parameters, final double[] rf_param1, final double[] rf_param2){
		this.rawdir = rawdir;
		this.labeldir = labeldir.clone();
		this.folders = folders.clone();
		// File interpreter for the label directory
		double[] labelset = new double[0];
		double[][] trainingset;
		this.loc = new int[0];
		if (!this.labeldir[0].equals("")){
			FileInterpreter FI = new FileInterpreter(this.labeldir[0]);
			TreeSet<SortPair2> lt = new TreeSet<SortPair2>();
			TreeSet<SortPair2> labels = Classifier.readLabels(FI);
			int[] lpc = Classifier.translateLabels(labeldir[1].equals("threshold"),lt,labels);
			if (labeldir[1].equals("")||labeldir[1].equals("threshold")){
				this.loc = new int[lpc.length*np];
				labelset = Classifier.simpleTraining(FI,np,lpc,loc,lt);
			}
			else {
				this.loc = new int[lpc.length*np*2];
				labelset = Classifier.correctionTraining(FI,labeldir[1],np,lpc.length,lpc,this.loc,lt);
			} 
			// extract the training vector based on the training locations
			trainingset = MultiFileExtraction.trainingExtraction(this.labeldir[0],this.folders,FI,this.loc);
		}
		else {
			FileInterpreter FI = new FileInterpreter(rawdir);
			loc = new int[np];
			// get the training point locations
			samplePoints(FI,this.loc);
			// extract the training vector based on the training locations
			trainingset = MultiFileExtraction.trainingExtraction(this.rawdir,this.folders,FI,this.loc);
			int nclass = (int)rf_param1[0];
			int[] temp = Shuffle.randPerm(trainingset.length);
			labelset = new double[temp.length];
			for (int i=0; i<labelset.length; i++){
				labelset[i] = temp[i]%nclass;
			}
		}
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
		this.cluster = new RFC3(parameters,rf_param1,rf_param2,trainingset,labelset,VectorFun.add(new double[trainingset.length],1),categorical);
	}
	
	public static void samplePoints(final FileInterpreter FI, int[] loc){
		// in this case we are simply sampling an equal number of points from each class
		int np = loc.length;
		// random number generator for sampling
		Random rng = new Random();
		// random sampling locations according to class
		int tot = FI.getInfo()[2][FI.getInfo()[0].length-1];
		for (int i=0; i<np; i++){
			loc[i] = rng.nextInt(tot);
		}
		Arrays.sort(loc);
	}
	
	public void classifyImage(String basename, final String[] folders, String target, int chunk_size, int n, int w, int h){
		// create a stack for the random forest votes
		ImageStack stack = new ImageStack(w,h);
		int dim = cluster.getForest().getCategorical().length;
		int nclass = cluster.getForest().getNumClasses();
		int nclust = cluster.getPredictors().length;
		int wh = w*h;
		double[] max2 = {Double.MAX_VALUE,Double.MIN_VALUE};
		for (int i=0; i<n; i++){
			double[][] st = new double[nclust][w*h];
			long a, b;
			int start = 0;
			int stop = 0;
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
				double[][] y = new double[1][0];
				if (this.cluster.getRegtype()==0){
					y = new double[chunk.length][chunk[0].length];
					for (int j=0; j<chunk.length; j++){
						y[j] = chunk[j].clone();
					}
				}
				else {
					int[][] tmp = this.cluster.getForest().pointToSetProximities(chunk,this.cluster.getPP());
					y = new double[tmp.length][tmp[0].length];
					for (int j=0; j<tmp.length; j++){
						for (int k=0; k<tmp[0].length; k++){
							y[j][k] = tmp[j][k];
						}
					}
				}
				for (int j=0; j<nclust; j++){
					// get the regressor values
					double[][] y2 = this.cluster.getPredictors()[j].applyForest(y);
					for (int k=0; k<y.length; k++){
						st[j][start+k] = y2[k][0];
					}
				}
				start+=chunk_size;
				b = System.nanoTime();
				System.out.println(Double.toString((b-a)/1e6));
				System.out.println("Next chunk");
			}
			for (int j=0; j<nclust; j++){
				double t = VectorFun.min(st[j])[0];
				if (t<max2[0]){
				max2[0] = t;
				}
				t = VectorFun.max(st[j])[0];
				if (t>max2[1]){
					max2[1] = t;
				}
				stack.addSlice("None", new FloatProcessor(w,h,st[j]));
			}
		}
		// save the stacks
		ImagePlus imp = new ImagePlus("Projections",stack);
		imp.getProcessor().setMinAndMax(max2[0],max2[1]);
		IJ.saveAsTiff(imp,target+File.separator+basename+TypeReader.TIFF[0]);
	}
	
	public void classifyFolder(String basefolder, final String[] folders, String target, int chunk_size){
		// get all the relevant file information
		FileInterpreter FI = new FileInterpreter(basefolder);
		int[][] info = FI.getInfo();
		// make sure the target folder exists
		new File(target).mkdirs();
		// go through all the image files in the base directory
		for (int i=0; i<info[0].length; i++){
			int[] size = ImageReader.getImageSize(FI.getNames()[i]);
			if (!(new File(target+File.separator+FI.getBases()[i]+TypeReader.TIFF[0])).exists()){
				classifyImage(FI.getBases()[i],folders,target,chunk_size,size[0],size[1],size[2]);
			}
		}
	}
	
	public RandomForest getForest(){
		return this.cluster.getForest();
	}
	
	public RFC3 getCluster(){
		return this.cluster;
	}
}