package flib.ij.featureextraction;

import java.util.Arrays;
import java.util.Random;
import java.io.File;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.math.VectorFun;
import flib.math.RankSort;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.RewriteForest;
import flib.algorithms.randomforest.splitfunctions.VarianceSplit;
import flib.algorithms.clustering.RFC;
import flib.algorithms.regions.RegionFunctions;
import flib.ij.featureextraction.FileInterpreter;
import flib.ij.featureextraction.FolderNames;
import flib.ij.featureextraction.MultiFileExtraction;
import flib.ij.io.ImageReader;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class RFCluster implements
java.io.Serializable {
	// folder for file origin and for file storage
	private String folderIO;
	// folders used for feature extraction
	private String[] folders;
	// initial training location
	private int[] loc;
	// trainingset
	private double[][] trainingset;
	// Variance split random forest classifier for clustering
	private RandomForest RF;
	// cluster interpreter
	private RFC cluster;
	// settings for the clustering
	private double[] parameters;
	// paramters[0]: number of training points
	// parameters[1]: rf mtry
	// parameters[2]: rf max depth
	// parameters[3]: rf max leafsize
	// parameters[4]: rf ntree
	// parameters[5]: number of clusters
	// parameters[6]: cluster size balancing
	// parameters[7]: max number of clustering iterations
	
	public RFCluster(String folderIO, final String[] folders, final double[] parameters){
		// copy folder names and setup parameters
		this.folderIO = folderIO;
		this.folders = folders.clone();
		this.parameters = parameters.clone();
		// count the number of images and of pixels per images in all image files
		// in the origin folder
		// sadly, this might take some time... but by far not as long as the 
		// process which follows after
		FileInterpreter FI = new FileInterpreter(this.folderIO);
		int[][] info = FI.getInfo();
		// random sampling time! set up the random number generator
		Random rng = new Random();
		// sample from all potential locations
		this.loc = new int[(int)parameters[0]];
		for (int i=0; i<loc.length; i++){
			loc[i] = rng.nextInt(info[2][info[2].length-1]);
		}
		Arrays.sort(loc);
		// extract the training data for the clustering
		this.trainingset = MultiFileExtraction.trainingExtraction(this.folderIO,this.folders,FI,loc);
		// create a converted training set based on rank ordering
		double[][] trs = new double[trainingset.length][trainingset[0].length];
		double[][] sorted_ts = new double[trainingset.length][trainingset[0].length];
		// go through all dimensions
		for (int i=0; i<trs[0].length; i++){
			double[] temp = new double[trs.length];
			for (int j=0; j<trs.length; j++){
				temp[j] = trainingset[j][i];
			}
			// sort the values according to dimension
			// and then scales the values to the range [0...1]
			RankSort rs = new RankSort(temp);
			int[] r = rs.getRank();
			double[] s = rs.getSorted();
			for (int j=0; j<r.length; j++){
				trs[r[j]][i] = (double)j/r.length;
				sorted_ts[j][i] = s[j];
			}
		}
		// rf parameters
		double[] rf_param = new double[5];
		rf_param[1] = parameters[1];
		rf_param[2] = parameters[2];
		rf_param[3] = parameters[3];
		// 0: randomly choose any dimensions for splitting
		// 2: choose dimensions according to a weighted shuffling
		rf_param[4] = 0;
		// set up the random forest for the clustering
		// labels don't matter for this
		this.RF = new RandomForest(trs,new double[trs.length],VectorFun.add(new double[trs.length],1),new boolean[trs[0].length],VectorFun.add(new double[trs[0].length],1),rf_param, new double[1],new VarianceSplit(),(int)parameters[4]);
		// rewrite the random forest to work with the original training data
		this.RF = RewriteForest.rewriteForest(this.RF,sorted_ts);
		// set up the RF clustering
		this.cluster = new RFC(RF.getLeafIndices(trainingset),(int)parameters[5],RF.getTreeSizes(),parameters[6],(int)parameters[7]);
	}
	
	public String[] getFolders(){
		return this.folders;
	}
	
	public void apply2Image(String basename, final String[] folders, String target, int chunk_size, int n, int w, int h){
		// create a stack for the cluster center distances
		ImageStack stack = new ImageStack(w,h);
		// a stack for the clusters
		ImageStack stack2 = new ImageStack(w,h);
		double max = Double.MIN_VALUE;
		int dim = this.RF.getCategorical().length;
		for (int i=0; i<n; i++){
			int start = 0;
			double[][] st = new double[(int)parameters[5]][w*h];
			double[] z = new double[w*h];
			double[] ind = new double[w*h];
			long a, b;
			while (start<w*h){
				a = System.nanoTime();
				double[][] chunk = MultiFileExtraction.imageChunk(basename,folders,start,chunk_size,n,i,w*h,dim);
				b = System.nanoTime();
				System.out.println(Double.toString((b-a)/1e6));
				// classify the chunk
				a = System.nanoTime();
				double[][] y = this.cluster.getDist(this.RF.getLeafIndices(chunk));
				b = System.nanoTime();
				System.out.println(Double.toString((b-a)/1e6));
				System.out.println("Next chunk");
				// put the chunk into the image holder
				for (int j=0; j<(int)parameters[5]; j++){
					for (int k=0; k<y.length; k++){
						st[j][start+k] = y[k][j];
						// get the maximum distance to find the cluster
						if (z[start+k]<st[j][start+k]){
							z[start+k] = st[j][start+k];
							ind[start+k] = j;
						}
					}
				}
				start+=chunk_size;
			}
			for (int j=0; j<(int)parameters[5]; j++){
				stack.addSlice("None", new FloatProcessor(w,h,st[j]));
				double t = VectorFun.max(st[j])[0];
				if (t>max){
					max = t;
				}
			}
			stack2.addSlice("None", new FloatProcessor(w,h,ind));
		}
		// save the stacks
		ImagePlus imp = new ImagePlus("Cluster Projections",stack);
		imp.getProcessor().setMinAndMax(0,max);
		IJ.saveAsTiff(imp,target+File.separator+basename+TypeReader.TIFF[0]);
		ImagePlus imp2 = new ImagePlus("Clusters",stack2);
		imp2.getProcessor().setMinAndMax(0,(int)parameters[5]-1);
		IJ.saveAsTiff(imp2,target+File.separator+FolderNames.CLUSTER+File.separator+basename+TypeReader.TIFF[0]);
	}
	
	public void apply2Folder(String basefolder, final String[] folders, String target, int chunk_size){
		// get all the relevant file information
		FileInterpreter FI = new FileInterpreter(basefolder);
		int[][] info = FI.getInfo();
		// make sure the target folder exists
		new File(target).mkdirs();
		new File(target+File.separator+FolderNames.CLUSTER).mkdirs();
		// go through all the image files in the base directory
		for (int i=0; i<info[0].length; i++){
			int[] size = ImageReader.getImageSize(FI.getNames()[i]);
			if (!(new File(target+File.separator+FI.getBases()[i]+TypeReader.TIFF[0])).exists()){
				apply2Image(FI.getBases()[i],folders,target,chunk_size,size[0],size[1],size[2]);
			}
		}
	}
	
	public RandomForest getForest(){
		return RF;
	}
	
	public RFC getCluster(){
		return cluster;
	}
}