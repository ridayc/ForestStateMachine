package flib.ij.stack;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import java.lang.Math;
import java.util.Arrays;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.random.Shuffle;
import flib.complexkernels.SymmetricLogGabor;
import flib.fftfunctions.FFTWrapper;
import flib.math.ComplexMath;
import flib.ij.stack.StackResize;
import flib.ij.stack.StackPixelArray;
import flib.ij.stack.StackConvolve;
import flib.algorithms.randomforest.RandomForest;

public class ScaleImportance {
	private ImagePlus imp;
	// number of points used for the forest
	private int np;
	private int nclust;
	private RandomForest RF;
	private double[][] trainingset;
	private int[] traininglabels;
	
	public ScaleImportance(final ImagePlus imp, final ImagePlus ind, double l, double factor, int numstep, int numpoints, int ntree){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int s = w*h;
		int n = imp.getNSlices();
		this.imp = imp;
		ImageStack stack = imp.getStack();
		int m = 0;
		int a;
		double[] pixels;
		// find the maximum index value
		for (int i=1; i<n+1; i++){
			pixels = VectorConv.float2double((float[])ind.getStack().getProcessor(i).convertToFloat().getPixels());
			a = (int)(VectorFun.max(pixels)[0]);
			if (a>m){
				m = a;
			}
		}
		this.nclust = m+1;
		int[] lengths = new int[this.nclust];
		// find the number of pixels in each label category
		for (int i=1; i<n+1; i++){
			pixels = VectorConv.float2double((float[])ind.getStack().getProcessor(i).convertToFloat().getPixels());
			for (int j=0; j<pixels.length; j++){
				lengths[(int)pixels[j]]++;
			}
		}
		// find the numer of pixels for the training in each category
		// and the total number of training points to use
		// either predefined or limited by the smallest category
		m = (int)VectorFun.min(VectorConv.int2double(lengths))[0];
		if (m>(int)(numpoints/this.nclust)){
			m = (int)(numpoints/this.nclust);
		}
		this.np = m*this.nclust;
		// generate the training point locations
		int[][] rloc = new int[this.nclust][m];
		int[] r;
		for (int i=0; i<this.nclust; i++){
			r = Shuffle.randPerm(lengths[i]);
			rloc[i] = VectorAccess.access(r,0,m);
			Arrays.sort(rloc[i]);			
		}
		// get the stack positions of the training point locations
		int[] counter = new int[this.nclust];
		int[] counter2 = new int[this.nclust];
		int[][] loc = new int[this.nclust][m];
		for (int i=1; i<n+1; i++){
			pixels = VectorConv.float2double((float[])ind.getStack().getProcessor(i).convertToFloat().getPixels());
			for (int j=0; j<pixels.length; j++){
				a = (int)pixels[j];
				if(rloc[a][counter[a]]==counter2[a]){
					loc[a][counter[a]] = (i-1)*pixels.length+j;
					counter[a]++;
					if (counter[a]>=m){
						counter[a] = m-1;
					}
				}
				counter2[a]++;
			}
		}
		// create a training
		int[] loc1d = VectorAccess.vertCat(loc);
		Arrays.sort(loc1d);
		// prepare the trainingset
		this.trainingset = new double[this.np][numstep];
		int w2, h2;
		ImagePlus temp;
		SymmetricLogGabor LG;
		double[] c,cr,ci;
		double l2;
		for (int i=0; i<numstep; i++){
			//w2 = (int)(w/Math.pow(factor,i));
			//h2 = (int)(h/Math.pow(factor,i));
			l2 = l*Math.pow(factor,i);
			w2 = w;
			h2 = h;
			LG = new SymmetricLogGabor(new int[]{w,h},l2,2);
			c = LG.getKernel();
			cr = FFTWrapper.fftshift2(w2,h2,ComplexMath.getReal(c));
			//ci = FFTWrapper.fftshift2(w2,h2,ComplexMath.getComplex(c));
			//c = ComplexMath.complexVector(cr,ci);
			c = ComplexMath.complexVector(cr);
			//temp = (new StackResize(imp,w2,h2)).getImage();
			temp = imp;
			temp = (new StackConvolve(w2,h2)).convolveComplexKernel(temp,c);
			temp = (new StackResize(temp,w,h)).getImage();
			a = 0;
			for (int j=1; j<n+1; j++){
				pixels = VectorConv.float2double((float[])temp.getStack().getProcessor(j).convertToFloat().getPixels());
				for (int k=0; k<pixels.length; k++){
					if(loc1d[a]==(j-1)*pixels.length+k){
						this.trainingset[a][i] = pixels[k];
						a++;
						if (a>=this.np){
							a = this.np-1;
						}
					}
				}
			}
		}
		// generate the training labels
		this.traininglabels = new int[this.np];
		a = 0;
		for (int i=1; i<n+1; i++){
			pixels = VectorConv.float2double((float[])ind.getStack().getProcessor(i).convertToFloat().getPixels());
			for (int j=0; j<pixels.length; j++){
				if(loc1d[a]==(i-1)*pixels.length+j){
					this.traininglabels[a] = (int)pixels[j];
					a++;
					if (a>=this.np){
						a = this.np-1;
					}
				}
			}
		}
		// generate other variables which are necessary for the random forest
		double[] weights = new double[this.np];
		Arrays.fill(weights, 1);
		boolean[] categorical = new boolean[numstep];
		this.RF = new RandomForest(trainingset,this.nclust,VectorConv.int2double(traininglabels),weights,categorical,new double[1], (int)Math.sqrt(numstep),Integer.MAX_VALUE,5,1,0,ntree);
	}
	
	public RandomForest getForest(){
		return this.RF;
	}
	
	public int[] getTrainingLabels(){
		return this.traininglabels.clone();
	}
}		