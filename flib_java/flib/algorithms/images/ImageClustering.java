package flib.algorithms.images;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Random;
import java.lang.Math;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.VectorConv;
import flib.math.random.Sample;
import flib.math.random.Shuffle;
import flib.math.random.Shuffler;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.splitfunctions.GiniClusterSplit;
import flib.algorithms.clustering.SpectralMatrices;
import flib.algorithms.clustering.RFC;

public class ImageClustering {
	private RandomForest RF;
	private int r;
	private int type;
	private int dim = 0;
	private int num_class;
	
	public ImageClustering(final double[][] im, final double[][] mask, final int[][] w, int np, int np2, int num_class, int type, int rad, double[] rf_param, double[] rf_param2, int ntree, int ntree2, double sigma, double[] sig){
		this.r = rad;
		this.type = type;
		this.num_class = num_class;
		rf_param2[0] = num_class;
		int n = im.length;
		int[] cs = new int[n];
		for (int i=0; i<n; i++){
			for (int j=0; j<mask[i].length; j++){
				if (mask[i][j]>0){
					cs[i]++;
				}
			}
		}
		int[][] mask2 = new int[n][];
		for (int i=0; i<n; i++){
			mask2[i] = new int[cs[i]];
			int count = 0;
			for (int j=0; j<mask[i].length; j++){
				if (mask[i][j]>0){
					mask2[i][count] = j;
					count++;
				}
			}
		}
				
		cs = VectorFun.cumsum(cs);
		int[] points = Arrays.copyOfRange(Shuffle.randPerm(cs[n-1]),0,np2);
		Arrays.sort(points);
		if (type%2==0){
			dim = NeighborhoodSample.circleCoord(rad).length;
		}
		else {
			dim = (2*rad+1)*(2*rad+1);
		}
		int[][] shape = new int[1][2];
		if (type%2==0){
			shape = NeighborhoodSample.circleCoord(r);
		}
		else {
			shape = new int[(2*r+1)*(2*r+1)][2];
			for (int i=0; i<2*r+1; i++){
				for (int j=0; j<2*r+1; j++){
					shape[i+(2*r+1)*j][0] = (i-r);
					shape[i+(2*r+1)*j][1] = (j-r);
				}
			}
		}
		double[] weights = new double[shape.length];
		double c = -Math.signum(sigma);
		for (int i=0; i<shape.length; i++){
			weights[i] = Math.exp(c*(shape[i][0]*shape[i][0]+shape[i][1]*shape[i][1])/(2*sigma*sigma));
		}
		double fill = Double.MIN_VALUE;
		double[][] trainingset = new double[np2][dim];
		// in case we choose to use a regression forest
		double[] labels = new double[np2];
		// current lower bound for the current image pixel location
		int lb = 0;
		for (int i=0; i<n; i++){
			int ub = VectorFun.binarySearch(points,cs[i]);
			if (ub>lb){
				double[][] samp = new double[1][0];
				if (type%2==0){
					samp = NeighborhoodSample.sampleDisk(VectorAccess.access(mask2[i],VectorFun.add(Arrays.copyOfRange(points,lb,ub),-lb)),w[i][0],w[i][1],rad,0,fill,im[i]);
				}
				else {
					samp = NeighborhoodSample.sampleRectangle(VectorAccess.access(mask2[i],VectorFun.add(Arrays.copyOfRange(points,lb,ub),-lb)),w[i][0],w[i][1],rad,rad,1,0,fill,im[i]);
				}
				for (int j=lb; j<ub; j++){
					trainingset[lb+j] = samp[j].clone();
				}
				lb = ub;
			}
		}
		if (type/2==0){
			// create a new randomized trainingset
			Random rng = new Random();
			double[][] temp = new double[2*np][dim];
			labels = new double[2*np];
			int[] rp = Shuffle.randPerm(np2);
			for (int i=0; i<np; i++){
				temp[i] = trainingset[rp[i]].clone();
				int[] samp = Sample.sample(np,dim,rng);
				for (int j=0; j<dim; j++){
					temp[i+np][j] = trainingset[rp[samp[j]]][j];
				}
				labels[i+np] = 1;
			}
			rf_param[0] = 2;
			double[] splitpurity = VectorFun.add(new double[2],1);
			RandomForest RFtemp = new RandomForest(temp,labels,VectorFun.add(new double[2*np],1),new boolean[dim],weights,rf_param,splitpurity,new GiniClusterSplit(),ntree);
			//System.out.print(Arrays.toString(RFtemp.outOfBagError()));
			labels = VectorConv.int2double(SpectralMatrices.forestProximity(RFtemp.getLeafIndices(trainingset),num_class,sig[0],sig[1],(int)sig[2],(int)sig[3]));
			rf_param[0] = num_class;
		}
		else if(type/2==1){
			labels = VectorConv.int2double(SpectralMatrices.minmaxRankDist(trainingset,num_class,np,sig[0]));
		}
		else if(type/2==2){
			labels = VectorConv.int2double(SpectralMatrices.medRankDist(trainingset,num_class,np,sig[0]));
		}
		else if(type/2==3){
			labels = VectorConv.int2double(SpectralMatrices.diffRankDist(trainingset,num_class,np,sig[0]));
		}
		else if(type/2==4){
			labels = VectorConv.int2double(SpectralMatrices.minRankDist(trainingset,num_class,np,sig[0]));
		}
		else if(type/2==5){
			labels = VectorConv.int2double(SpectralMatrices.maxRankDist(trainingset,num_class,np,sig[0]));
		}
		else {
			// create a new randomized trainingset
			Random rng = new Random();
			double[][] temp = new double[2*np][dim];
			labels = new double[2*np];
			int[] rp = Shuffle.randPerm(np2);
			for (int i=0; i<np; i++){
				temp[i] = trainingset[rp[i]].clone();
				int[] samp = Sample.sample(np,dim,rng);
				for (int j=0; j<dim; j++){
					temp[i+np][j] = trainingset[rp[samp[j]]][j];
				}
				labels[i+np] = 1;
			}
			rf_param[0] = 2;
			double[] splitpurity = VectorFun.add(new double[2],1);
			RandomForest RFtemp = new RandomForest(temp,labels,VectorFun.add(new double[2*np],1),new boolean[dim],weights,rf_param,splitpurity,new GiniClusterSplit(),ntree);
			labels = VectorConv.int2double((new RFC(RFtemp.getLeafIndices(trainingset),num_class,RFtemp.getTreeSizes(),sig[0],(int)sig[2])).assignCluster(RFtemp.getLeafIndices(trainingset)));
		}
		double[] splitpurity = VectorFun.add(new double[num_class],1);
		RF = new RandomForest(trainingset,labels,VectorFun.add(new double[np2],1),new boolean[dim],weights,rf_param2,splitpurity,new GiniClusterSplit(),ntree2);
	}

	public double[] apply(final double[] x, int w, int h){
		double[] im = new double[x.length];
		double fill = Double.MIN_VALUE;
		double[] samp = new double[0];
		for (int i=0; i<x.length; i++){
			int[] p = new int[1];
			p[0] = i;
			if (type%2==0){
				samp = NeighborhoodSample.sampleDisk(p,w,h,r,0,fill,x)[0];
			}
			else {
				samp = NeighborhoodSample.sampleRectangle(p,w,h,r,r,1,0,fill,x)[0];
			}
			int counter = 0;
			for (int j=0; j<dim; j++){
				if (samp[j]==fill){
					counter++;
				}
			}
			if (counter>0){
				int[] missing = new int[counter];
				counter = 0;
				for (int j=0; j<dim; j++){
					if (samp[j]==fill){
						missing[counter] = j;
						counter++;
					}
				}
				im[i] = VectorFun.maxind(RF.applyForest(samp,missing));
			}
			else {
				im[i]  = RF.predict(samp);
			}
		}
		return im;
	}
	
	public RandomForest getForest(){
		return RF;
	}
}		