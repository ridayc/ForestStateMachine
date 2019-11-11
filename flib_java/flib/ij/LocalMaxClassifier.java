package flib.ij;

import java.util.ArrayList;
import java.lang.Math;
import flib.ij.stack.StackOperations;
import flib.ij.stack.StackConvolve;
import flib.math.BV;
import flib.math.VectorConv;
import flib.math.VectorAccess;
import flib.math.VectorFun;
import flib.complexkernels.Gaussian;
import flib.algorithms.DistanceTransform;
import flib.algorithms.SeededWatershed;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.ij.featureextraction.PatchRFC;
import flib.ij.segmentation.ApplyPatchRFC;
//import flib.algorithms.randomforest.RandomForest;

public class LocalMaxClassifier {
	// w: image width
	// h: image height
	// n: number of images
	// nc: number of classes in the initial images
	// tc: target number of clusters per class
	private int w,h,nc, tc;
	// r: patch radius
	private double r;
	private double[][] im;
	private int[][] minpixels, minclasses;
	private int[] ind;
	private ArrayList<PatchRFC> PRFC;
	
	public LocalMaxClassifier(final double[] im, int w, int h, int nc, int tc, double r, int numpoints, int mtry, double splitpurity, int ntree, double balance, double rad, int maxit, int foresttype){
		this.w = w;
		this.h = h;
		this.nc = nc;
		this.tc = tc;
		this.r = r;
		this.minpixels = new int[nc][];
		this.minclasses = new int[nc][];
		double fill = 1e5;
		// Gaussian smoothing kernel preparation
		double[] c = (new Gaussian(new int[]{w,h},0.3)).getKernel();
		StackConvolve cK = new StackConvolve(w,h);
		for (int i=0; i<this.nc; i++){
			// get the distance transform of the individual classes
			// get the ridges of the slightly smoothed distance transform
			double[] temp = VectorFun.mult(cK.convolveComplexKernel(DistanceTransform.dt2d(this.w,this.h,VectorConv.bool2double(BV.eq(im,i)),0),c),-1);
			double[] temploc = new double[this.w*this.h];
			int[] tempint = SeededWatershed.connectedMinima2(this.w,this.h,temp,8);
			VectorAccess.write(temploc,tempint,1);
			temp = VectorFun.mult(temploc,VectorConv.bool2double(BV.lt(temp,-1)));
			// ignore ridge lines below this distance transform value
			double comp = 1;
			this.minpixels[i] = VectorAccess.subset(temp);
		}
		// split the individual ridge pixel sets into clusters
		PRFC = new ArrayList<PatchRFC>();
		int s = (NeighborhoodSample.spiralCoord(1,r,1,1,rad,4)).length;
		int m;
		if (mtry<1){
			m = (int)Math.sqrt(s);
		}
		else {
			m = (int)((float)s/mtry);
		}
		double[] ones;
		this.ind = new int[this.w*this.h];
		double[] minval = new double[this.w*this.h];
		int[][] origclass = VectorAccess.labels2Indices(VectorConv.double2int(im),this.nc);
		for (int i=0; i<this.nc; i++){
			// generate the patch RFC
			PRFC.add(new PatchRFC(this.w, this.h,this.r,fill,this.minpixels[i],numpoints,this.tc,true,m,100,3,splitpurity, ntree,0,balance,rad,maxit,foresttype,im));
			// apply the patch RFC to obtain a clustering
			this.minclasses[i] = ApplyPatchRFC.getIndices(PRFC.get(i).getForest(),PRFC.get(i).getRFC(),this.w,this.h,this.r,rad,fill,0,this.minpixels[i],im);
			int[][] labels = VectorAccess.labels2Indices(this.minclasses[i],this.tc);
			ones = VectorFun.add(new double[w*h],1);
			VectorAccess.write(ones,VectorAccess.access(this.minpixels[i],labels[0]),new double[labels[0].length]);
			VectorAccess.write(this.ind,origclass[i],VectorFun.add(new int[origclass[i].length],i*this.tc));
			double[] temp = cK.convolveComplexKernel(DistanceTransform.dt2d(this.w,this.h,ones,0),c);
			VectorAccess.write(minval,origclass[i],VectorAccess.access(temp,origclass[i]));
			for (int j=1; j<this.tc; j++){
				ones = VectorFun.add(new double[w*h],1);
				VectorAccess.write(ones,VectorAccess.access(this.minpixels[i],labels[j]),new double[labels[j].length]);
				temp = cK.convolveComplexKernel(DistanceTransform.dt2d(this.w,this.h,ones,0),c);
				for (int k=0; k<origclass[i].length; k++){
					if (temp[origclass[i][k]]<minval[origclass[i][k]]){
						this.ind[origclass[i][k]] = j+i*this.tc;
						minval[origclass[i][k]] = temp[origclass[i][k]];
					}
				}
			}
		}
	}
	
	public int[] getIndices(){
		return this.ind.clone();
	}
}