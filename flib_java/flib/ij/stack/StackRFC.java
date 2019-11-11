package flib.ij.stack;

import java.util.Random;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.random.Shuffle;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.clustering.RFC;

public class StackRFC {
	private RFC rfc;
	private RandomForest RF;
	
	public static double[][] randomTrainingSet(final ImagePlus imp, final int[] randind){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		int numpoints = randind.length;
		ImageStack stack = imp.getImageStack();
		double[] x;
		double[][] trainingset = new double[n][numpoints];
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			trainingset[i-1] = VectorAccess.access(x,randind);
		}
		return VectorAccess.flip(trainingset);
	}
	
	public static double[][] randomTrainingSet(final ImagePlus imp, int n){
		int[] randind = Shuffle.randPerm(imp.getWidth()*imp.getHeight());
		return randomTrainingSet(imp,VectorAccess.access(randind,0,n));
	}
	
	public StackRFC(final double[][] trainingset, int ntree, int mtry, int nc, double balance, int maxit, final boolean[] categorical, final double[] dimweights, int maxdepth, int maxleafsize, double splitpurity){
		int n = trainingset.length;
		int d = trainingset[0].length;
		double[][] new_set = new double[n*2][d];
		double[] weights = VectorFun.add(new double[2*n],1);
		double[] labels = new double[2*n];
		for (int i=0; i<n; i++){
			new_set[i] = trainingset[i].clone();
		}
		Random rng = new Random();
		for (int i=n; i<2*n; i++){
			labels[i] = 1;
			for (int j=0; j<d; j++){
				new_set[i][j] = trainingset[rng.nextInt(n)][j];
			}
		}
		this.RF = new RandomForest(new_set,2,labels,weights,categorical,dimweights,mtry,maxdepth,maxleafsize,splitpurity,1,ntree);
		int[][] lfi = this.RF.getLeafIndices(trainingset);
		int[] ts = this.RF.getTreeSizes();
		this.rfc = new RFC(lfi,nc,ts,balance,maxit);
	}
	
	public StackRFC(final double[][] trainingset, final double[] labels, int numclasses, int ntree, int mtry, int nc, double balance, int maxit, final boolean[] categorical, final double[] dimweights, int maxdepth, int maxleafsize, double splitpurity, int splittype){
		int n = trainingset.length;
		int d = trainingset[0].length;
		double[] weights = VectorFun.add(new double[n],1);
		this.RF = new RandomForest(trainingset,numclasses,labels,weights,categorical,dimweights,mtry,maxdepth,maxleafsize,splitpurity,splittype,ntree);
		int[][] lfi = this.RF.getLeafIndices(trainingset);
		int[] ts = this.RF.getTreeSizes();
		this.rfc = new RFC(lfi,nc,ts,balance,maxit);
	}
	
	public static double[][] dataSet(final ImagePlus imp){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		double[][] dataset = new double[n][w*h];
		for (int i=1; i<n+1; i++){
			dataset[i-1] = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
		}
		return VectorAccess.flip(dataset);
	}
	
	public int[] applyRFC(final double[][] dataset){
		return this.rfc.assignCluster(this.RF.getLeafIndices(dataset));
	}
	
	public int[] getDistances(){
		return this.rfc.getDistances();
	}
	
	public int[] getSizes(){
		return this.rfc.getSizes();
	}
	
	public RandomForest getForest(){
		return this.RF;
	}
	
	public RFC getRFC(){
		return this.rfc;
	}
}