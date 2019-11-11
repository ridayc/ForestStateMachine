package flib.algorithms.sampling;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Random;
import java.lang.Math;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.VectorConv;
import flib.math.random.Shuffle;
import flib.math.random.Shuffler;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.splitfunctions.GiniClusterSplit;
import flib.algorithms.randomforest.splitfunctions.CrossingRegression;
import flib.algorithms.clustering.SpectralMatrices;

public class ImageSampler {
	private RandomForest RF;
	private ArrayList<RandomForest> samplers;
	private int r;
	private int type;
	private int dim = 0;
	
	public ImageSampler(final double[][] im, final double[][] mask, final int[][] w, int np, int ns, int nsp, int type, int rad, double[] rf_param, double sigma){
		this.r = rad;
		this.type = type;
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
		int[] points = Arrays.copyOfRange(Shuffle.randPerm(cs[n-1]),0,np);
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
		double[][] trainingset = new double[np][dim];
		// in case we choose to use a regression forest
		double[] labels = new double[np];
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
					labels[lb+j] = im[i][mask2[i][points[lb+j]]];
				}
				lb = ub;
			}
		}
		// use a regression forest for separation
		if (type/2==0){
			double[] parameters = new double[6];
			parameters[0] = 1;
			parameters[1] = rf_param[0];
			parameters[2] = rf_param[1];
			parameters[3] = rf_param[2];
			parameters[4] = 2;
			parameters[5] = rf_param[3];
			double[] splitpurity = new double[1];
			this.RF = new RandomForest(trainingset,labels,VectorFun.add(new double[np],1),new boolean[dim],weights,parameters,splitpurity,new CrossingRegression(),(int)rf_param[4]);
		}
		else {
			double[] parameters = new double[8];
			parameters[0] = 2;
			parameters[1] = rf_param[0];
			parameters[2] = rf_param[1];
			parameters[3] = rf_param[2];
			parameters[4] = 2;
			parameters[5] = rf_param[3];
			parameters[6] = 0.5;
			parameters[7] = 1;
			double[] splitpurity = VectorFun.add(new double[2],1);
			labels = VectorConv.int2double(SpectralMatrices.minRankDist(trainingset,2,(int)rf_param[6],rf_param[7]));
			this.RF = new RandomForest(trainingset,labels,VectorFun.add(new double[np],1),new boolean[dim],weights,parameters,splitpurity,new GiniClusterSplit(),(int)rf_param[4]);
		}
		this.samplers = new ArrayList<RandomForest>();
		for (int i=0; i<ns; i++){
			points = Arrays.copyOfRange(Shuffle.randPerm(cs[n-1]),0,nsp);
			Arrays.sort(points);
			trainingset = new double[nsp][dim];
			// in case we choose to use a regression forest
			labels = new double[nsp];
			// current lower bound for the current image pixel location
			lb = 0;
			for (int j=0; j<n; j++){
				int ub = VectorFun.binarySearch(points,cs[j]);
				if (ub>lb){
					double[][] samp = new double[1][0];
					if (type%2==0){
						samp = NeighborhoodSample.sampleDisk(VectorAccess.access(mask2[j],VectorFun.add(Arrays.copyOfRange(points,lb,ub),-lb)),w[j][0],w[j][1],rad,0,fill,im[j]);
					}
					else {
						samp = NeighborhoodSample.sampleRectangle(VectorAccess.access(mask2[j],VectorFun.add(Arrays.copyOfRange(points,lb,ub),-lb)),w[j][0],w[j][1],rad,rad,1,0,fill,im[j]);
					}
					for (int k=lb; k<ub; k++){
						trainingset[lb+k] = samp[k].clone();
						labels[lb+k] = im[j][mask2[j][points[lb+k]]];
					}
				}
				lb = ub;
			}
			// use a regression forest for separation
			if (type/2==0){
				double[] parameters = new double[6];
				parameters[0] = 1;
				parameters[1] = rf_param[0];
				parameters[2] = rf_param[1];
				parameters[3] = rf_param[5];
				parameters[4] = 1;
				parameters[5] = rf_param[3];
				double[] splitpurity = new double[1];
				this.samplers.add(new RandomForest(trainingset,labels,VectorFun.add(new double[nsp],1),new boolean[dim],VectorFun.add(new double[dim],1),parameters,splitpurity,new CrossingRegression(),(int)rf_param[4]));
			}
			else {
				double[] parameters = new double[8];
				parameters[0] = 2;
				parameters[1] = rf_param[0];
				parameters[2] = rf_param[1];
				parameters[3] = rf_param[5];
				parameters[4] = 1;
				parameters[5] = rf_param[3];
				parameters[6] = 0.5;
				parameters[7] = 1;
				double[] splitpurity = VectorFun.add(new double[2],1);
				labels = VectorConv.int2double(SpectralMatrices.minRankDist(trainingset,2,(int)rf_param[6],rf_param[7]));
				this.RF = new RandomForest(trainingset,labels,VectorFun.add(new double[np],1),new boolean[dim],VectorFun.add(new double[dim],1),parameters,splitpurity,new GiniClusterSplit(),(int)rf_param[4]);
			}
		}
	}
	
	public double[][] sampleImage(int w, int h, final int[] rep, int trans, boolean rev, int met){
		int counter = 0;
		for (int i=0; i<dim; i++){
			counter+=rep[i];
		}
		double[][] im = new double[counter+1][w*h];
		int l = samplers.size();
		double[][] temp = RF.variableImportance();
		double[] vi = new double[dim];
		for (int i=0; i<dim; i++){
			vi[i] = temp[i][0];
		}
		Shuffler sh = new Shuffler(vi);
		Random rng = new Random();
		counter = 0;
		for (int i=0; i<dim; i++){
			for (int j=0; j<rep[i]; j++){
				// choose an order to go through all image points
				int[] rp = Shuffle.randPerm(im[0].length);
				for (int k=0; k<im[0].length; k++){
					if (j<=trans){
						sample(w,h,im[counter],rp[k],samplers.get(rng.nextInt(samplers.size())),RF,i,rev,sh,met);
					}
					else {
						sample(w,h,im[counter],rp[k],RF,RF,i,rev,sh,met);
					}
				}
				im[counter+1] = im[counter].clone();
				counter++;
			}
		}
		return im;
	}
	
	public double[][] sampleImage(int w, int h, int rep, int trans, boolean rev, int met){
		int[] s = VectorFun.add(new int[dim],rep);
		return sampleImage(w,h,s,trans,rev,met);
	}
	
	public void sample(int w, int h, double[] im, int i, final RandomForest RF1, final RandomForest RF2, int level, boolean rev, Shuffler sh, int met){
		int l = w*h;
		double fill = Double.MIN_VALUE;
		TreeSet<Integer> t = new TreeSet<Integer>();
		double[] point = new double[0];
		if (rev){
			for (int j=0; j<dim; j++){
				t.add(j);
			}
			int[] list = sh.randPerm(level);
			for (int j=0; j<list.length; j++){
				t.remove(list[j]);
			}
			int[] p = new int[1];
			p[0] = i;
			if (type%2==0){
				point = NeighborhoodSample.sampleDisk(p,w,h,r,met,fill,im)[0];
			}
			else {
				point = NeighborhoodSample.sampleRectangle(p,w,h,r,r,1,met,fill,im)[0];
			}
			for (int j=0; j<point.length; j++){
				if (point[j]==fill){
					t.add(j);
				}
			}
		}
		else {
			int[] list = sh.randPerm(dim-level);
			for (int j=0; j<list.length; j++){
				t.add(list[j]);
			}
			int[] p = new int[1];
			p[0] = i;
			if (type%2==0){
				point = NeighborhoodSample.sampleDisk(p,w,h,r,met,fill,im)[0];
			}
			else {
				point = NeighborhoodSample.sampleRectangle(p,w,h,r,r,1,met,fill,im)[0];
			}
			for (int j=0; j<point.length; j++){
				if (point[j]==fill){
					t.add(j);
				}
			}
		}
		int[] missing = new int[t.size()];
		Iterator<Integer> it = t.iterator();
		int counter = 0;
		while (it.hasNext()){
			missing[counter] = it.next();
			counter++;
		}
		if (type/2==0){
			double[] votes = new double[0];
			if (met%2==0){
				votes = RF2.applyForest(RF1.nearestNeighbor(point,missing));
			}
			else {
				votes = RF2.applyForest(RF1.sample(point,missing));
			}
			Arrays.sort(votes);
			if (met/2==0){
				im[i] = votes[(new Random()).nextInt(votes.length)];
			}
			else {
				im[i] = votes[votes.length/2];
			}
		}
	}

	public double[][] completion(final double[] x, final double[] mask, int w, int h, int trans, int it, int met, double sigma, int type, double prob){
		int counter = 0;
		for (int i=0; i<w*h; i++){
			if (mask[i]<=0){
				counter++;
			}
		}
		int[] loc = new int[counter];
		counter = 0;
		for (int i=0; i<w*h; i++){
			if (mask[i]<=0){
				loc[counter] = i;
				counter++;
			}
		}
		double[][] im = new double[it+1][w*h];
		im[0] = x.clone();
		Random rng = new Random();
		double fill = Double.MAX_VALUE;
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
		for (int i=0; i<shape.length; i++){
			weights[i] = Math.exp(-(shape[i][0]*shape[i][0]+shape[i][1]*shape[i][1])/(2*sigma*sigma));
		}
		for (int i=0; i<it; i++){
			// choose an order to go through all image points
			int[] rp = Shuffle.randPerm(loc.length);
			for (int j=0; j<loc.length; j++){
				int[] p = new int[1];
				double[] point = new double[0];
				double[] votes = new double[0];
				p[0] = loc[j];
				if (type%2==0){
					point = NeighborhoodSample.sampleDisk(p,w,h,r,met,fill,im[i])[0];
				}
				else {
					point = NeighborhoodSample.sampleRectangle(p,w,h,r,r,1,met,fill,im[i])[0];
				}
				counter = 0; 
				for (int k=0; k<point.length; k++){
					if (point[k]==fill){
						counter++;
					}
				}
				int[] missing = new int[counter];
				counter = 0;
				for (int k=0; k<point.length; k++){
					if (point[k]==fill){
						missing[counter] = k;
						counter++;
					}
				}
				if (i>=trans){
					//votes = RF.applyForest(RF.nearestNeighbor(point,missing));
					if (met/2==0){
						votes = RF.nearestNeighbor(point,missing);
					}
					else {
						votes = RF.sampledNeighbor(point,missing,type/2,prob);
					}
				}
				else {
					//votes = RF.applyForest(samplers.get(rng.nextInt(samplers.size())).nearestNeighbor(point,missing));
					if (met/2==0){
						votes = samplers.get(rng.nextInt(samplers.size())).nearestNeighbor(point,missing);
					}
					else {
						votes = samplers.get(rng.nextInt(samplers.size())).sampledNeighbor(point,missing,type/2,prob);
					}
				}
				/*
				if (met%2==0){
					//im[i][loc[j]] = votes[rng.nextInt(votes.length)];
				}
				else {
					//im[i][loc[j]] = votes[votes.length/2];
				}
				*/
				int[] loc2 = NeighborhoodSample.shapeNeighbor2d(shape,w,h,loc[j]%w,loc[j]/w,met%2);
				for (int k=0; k<loc2.length; k++){
					if(loc2[k]!=-1){
						if (Arrays.binarySearch(loc,loc2[k])>=0){
							im[i][loc2[k]] = votes[k]*weights[k]+im[i][loc2[k]]*(1-weights[k]);
						}
					}
				}
			}
			im[i+1] = im[i].clone();
		}
		return im;
	}
	
	public RandomForest getForest(){
		return RF;
	}
	
	public ArrayList<RandomForest> getSamplers(){
		return samplers;
	}
}		