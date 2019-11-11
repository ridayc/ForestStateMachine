package flib.ij.celldetection;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorAccess;
import flib.math.VectorConv;
import flib.math.BV;
import flib.algorithms.BLabel;
import flib.algorithms.blobanalysis.BlobAnalysis2D;
import flib.ij.stack.StackKmeans;

public class RoundnessLevels {
	private ImagePlus indices;
	private int n,nclust,w,h;
	private int[][] blobsize;
	private double[][] rg;
	
	public RoundnessLevels(final ImagePlus imp, int pps, int nclust, int maxit){
		this.indices = (new StackKmeans(imp,pps,nclust,maxit)).getIndices();
		this.n = this.indices.getNSlices();
		this.w = this.indices.getWidth();
		this.h = this.indices.getHeight();
		this.nclust = nclust;
		this.blobsize = new int[this.nclust-1][];
		this.rg = new double[this.nclust-1][];
		int[][] tempsize = new int[this.n][];
		double[][] temprg = new double[this.n][];
		ImageStack stack = this.indices.getImageStack();
		double[] x;
		BLabel bl;
		BlobAnalysis2D bla2;
		for (int i=1; i<this.nclust; i++){
			for (int j=0; j<this.n; j++){
				x = VectorConv.float2double((float[])(stack.getProcessor(j+1).convertToFloat().getPixels()));
				x = VectorConv.bool2double(BV.gte(x,i));
				bl = new BLabel(w,h,x,4);
				bla2 = new BlobAnalysis2D(w,bl.getBlobList());
				tempsize[j] = bla2.getBlobSizes();
				temprg[j] = bla2.radiusOfGyration();
			}
			this.blobsize[i-1] = VectorAccess.vertCat(tempsize);
			this.rg[i-1] = VectorAccess.vertCat(temprg);
		}
	}
	
	public int[][] getSizes(){
		int[][] blobsize = new int[this.nclust-1][];
		for (int i=0; i<this.nclust-1; i++){
			blobsize[i] = this.blobsize[i].clone();
		}
		return blobsize;
	}
	
	public double[][] getRG(){
		double[][] rg = new double[this.nclust-1][];
		for (int i=0; i<this.nclust-1; i++){
			rg[i] = this.rg[i].clone();
		}
		return rg;
	}
	
	public ImagePlus getIndices(){
		return this.indices;
	}
	
	public ImagePlus getBinaryStack(int level){
		ImageStack stack = indices.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		for (int i=0; i<this.n; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i+1).convertToFloat().getPixels()));
			x = VectorConv.bool2double(BV.gte(x,level));
			stack2.addSlice("None",new FloatProcessor(w,h,x));
		}
		return (new ImagePlus("Thresholded",stack2));
	}
}