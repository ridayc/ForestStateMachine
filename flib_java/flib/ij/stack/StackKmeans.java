package flib.ij.stack;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorConv;
import flib.math.VectorAccess;
import flib.math.random.Shuffle;
import flib.algorithms.clustering.Kmeans;
import flib.math.RankSort;

public class StackKmeans {
	private ImagePlus imp;
	private ImagePlus kmeanstack;
	private Kmeans km;
	private int nclust;
	
	public StackKmeans(final ImagePlus imp, int pps, int nclust, int maxit){
		this.imp = imp;
		//pps: pixels per slice for the kmeans
		int w = imp.getWidth();
		int h = imp.getHeight();
		int s = w*h;
		int n = imp.getNSlices();
		int l = n*pps;
		this.nclust = nclust;
		// pixels to use for the kmeans
		double[] kmp = new double[l];
		// random permutation vector
		int[] a;
		ImageStack stack = imp.getImageStack();
		double[] x;
		for (int i=1; i<n+1; i++){
			a = Shuffle.randPerm(s);
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			VectorAccess.write(kmp,VectorAccess.access(x,VectorAccess.access(a,0,pps)),(i-1)*pps);
		}
		this.km = new Kmeans(VectorAccess.horzCat(kmp),nclust,maxit);
		double[][] cent = this.km.getCenters();
		double[] cent1D = new double[nclust];
		for (int i=0; i<nclust; i++){
			cent1D[i] = cent[i][0];
		}
		RankSort r = new RankSort(cent1D);
		this.km.sortCenters(r.getRank());
	}
	
	public StackKmeans(final ImagePlus imp, int pps, int nclust){
		this(imp,pps,nclust,200);
	}
	
	public StackKmeans(final ImagePlus imp, int pps){
		this(imp,pps,2,200);
	}
	
	public ImagePlus getIndices(){
		int w = this.imp.getWidth();
		int h = this.imp.getHeight();
		int n = this.imp.getNSlices();
		ImageStack stack = this.imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			stack2.addSlice("None", new FloatProcessor(w,h,VectorConv.int2double(this.km.assignCluster(VectorAccess.horzCat(x)))));
		}
		ImagePlus impi = new ImagePlus("kmeans indices",stack2);
		impi.getProcessor().setMinAndMax(0,this.nclust-1);
		return impi;
	}
	
	public Kmeans getKmeans(){
		return this.km;
	}
}	