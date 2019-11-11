package flib.ij.stack;

import java.util.Arrays;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.BV;
import flib.fftfunctions.FFTWrapper;
import flib.math.random.Shuffle;
import flib.algorithms.Watershed;
import flib.algorithms.SeededWatershed;
import flib.algorithms.DistanceTransform;
import flib.algorithms.AssignToRandomNeighbor;
import flib.algorithms.BLabel;
import flib.algorithms.regions.RegionPartitioning;
import flib.algorithms.regions.RegionFunctions;

public class StackOperations {
	
	public static ImagePlus assignToRandomNeighbor(final ImagePlus imp, int connectivity){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		double[] z;
		AssignToRandomNeighbor AtRN;
		double l2,t;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			AtRN = new AssignToRandomNeighbor(w,h,x,connectivity);
			z = AtRN.getImage();
			t = VectorFun.min(z)[0];
			if (t<min){
				min = t;
			}
			t = VectorFun.max(z)[0];
			if (t>max){
				max = t;
			}
			stack2.addSlice("None", new FloatProcessor(w,h,z));
		}
		ImagePlus imp2 = new ImagePlus("Reassigned",stack2);
		imp2.getProcessor().setMinAndMax(min,max);
		return imp2;
	}
	
	public static ImagePlus watershed(final ImagePlus imp, int connectivity, int direction){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		double[] z;
		SeededWatershed WS;
		double l2,t;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			if (direction==1){
				WS = new SeededWatershed(w,h,x,connectivity);
			}
			else {
				WS = new SeededWatershed(w,h,VectorFun.mult(x,-1),connectivity);
			}
			z = VectorConv.int2double(WS.getRegionNumber());
			t = VectorFun.min(z)[0];
			if (t<min){
				min = t;
			}
			t = VectorFun.max(z)[0];
			if (t>max){
				max = t;
			}
			stack2.addSlice("None", new FloatProcessor(w,h,z));
		}
		ImagePlus imp2 = new ImagePlus("Watershed",stack2);
		imp2.getProcessor().setMinAndMax(min,max);
		return imp2;
	}
	
	public static ImagePlus watershedBoundaries2Sided(final ImagePlus imp, int connectivity){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		double[] z;
		Watershed WS;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			WS = new Watershed(w,h,x,connectivity);
			z = VectorConv.bool2double(BV.gt(VectorConv.int2double(WS.getRegionNumber()),0));
			stack2.addSlice("None", new FloatProcessor(w,h,z));
			WS = new Watershed(w,h,VectorFun.mult(x,-1),connectivity);
			z = VectorConv.bool2double(BV.gt(VectorConv.int2double(WS.getRegionNumber()),0));
			stack2.addSlice("None", new FloatProcessor(w,h,z));
		}
		ImagePlus imp2 = new ImagePlus("Watershed",stack2);
		return imp2;
	}
	
	public static ImagePlus distanceTransform(final ImagePlus imp, int type){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		double[] z;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		double t;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			z = DistanceTransform.dt2d(w,h,x,type);
			t = VectorFun.min(z)[0];
			if (t<min){
				min = t;
			}
			t = VectorFun.max(z)[0];
			if (t>max){
				max = t;
			}
			stack2.addSlice("None", new FloatProcessor(w,h,z));
		}
		ImagePlus imp2 = new ImagePlus("DistanceTransform",stack2);
		imp2.getProcessor().setMinAndMax(min,max);
		return imp2;
	}
	
	public static ImagePlus regions(final ImagePlus imp){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		double[] z;
		double l2,t;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		for (int i=1; i<n+1; i++){
			x = VectorConv.float2double((float[])(stack.getProcessor(i).convertToFloat().getPixels()));
			z = (new RegionPartitioning(w,h,x)).getImage();
			t = VectorFun.min(z)[0];
			if (t<min){
				min = t;
			}
			t = VectorFun.max(z)[0];
			if (t>max){
				max = t;
			}
			stack2.addSlice("None", new FloatProcessor(w,h,z));
		}
		ImagePlus imp2 = new ImagePlus("Regions",stack2);
		imp2.getProcessor().setMinAndMax(min,max);
		return imp2;
	}
	
	public static ImagePlus regionMean(final ImagePlus regions, final ImagePlus imp){
		int w = regions.getWidth();
		int h = regions.getHeight();
		int nr = regions.getNSlices();
		int ni = imp.getNSlices();
		ImageStack stackr = regions.getImageStack();
		ImageStack stacki = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		double[] y;
		double[] z;
		double[] m;
		int[][] reg;
		double l2,t;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		y = VectorConv.float2double((float[])(stacki.getProcessor(1).convertToFloat().getPixels()));
		for (int i=1; i<nr+1; i++){
			x = VectorConv.float2double((float[])(stackr.getProcessor(i).convertToFloat().getPixels()));
			if (nr==ni){
				y = VectorConv.float2double((float[])(stacki.getProcessor(i).convertToFloat().getPixels()));
			}
			reg = RegionFunctions.getRegions(x);
			m = RegionFunctions.regionMean(reg,y);
			z = RegionFunctions.regionFill(w,h,reg,m);
			t = VectorFun.min(z)[0];
			if (t<min){
				min = t;
			}
			t = VectorFun.max(z)[0];
			if (t>max){
				max = t;
			}
			stack2.addSlice("None", new FloatProcessor(w,h,z));
		}
		ImagePlus imp2 = new ImagePlus("Region Mean",stack2);
		imp2.getProcessor().setMinAndMax(min,max);
		return imp2;
	}
	
	public static ImagePlus regionMaxFreq(final ImagePlus regions, final ImagePlus imp, int numval){
		int w = regions.getWidth();
		int h = regions.getHeight();
		int nr = regions.getNSlices();
		int ni = imp.getNSlices();
		ImageStack stackr = regions.getImageStack();
		ImageStack stacki = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		double[] x;
		int[] y;
		double[] z;
		double[] m;
		double[][] valF;
		int[][] reg;
		double l2,t;
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		y = VectorConv.float2int((float[])(stacki.getProcessor(1).convertToFloat().getPixels()));
		for (int i=1; i<nr+1; i++){
			x = VectorConv.float2double((float[])(stackr.getProcessor(i).convertToFloat().getPixels()));
			if (nr==ni){
				y = VectorConv.float2int((float[])(stacki.getProcessor(i).convertToFloat().getPixels()));
			}
			reg = RegionFunctions.getRegions(x);
			valF = RegionFunctions.valueFrequencies(reg,numval,y);
			m = VectorConv.int2double(RegionFunctions.maxFreqInd(reg,valF));
			z = RegionFunctions.regionFill(w,h,reg,m);
			t = VectorFun.min(z)[0];
			if (t<min){
				min = t;
			}
			t = VectorFun.max(z)[0];
			if (t>max){
				max = t;
			}
			stack2.addSlice("None", new FloatProcessor(w,h,z));
		}
		ImagePlus imp2 = new ImagePlus("Region Mean",stack2);
		imp2.getProcessor().setMinAndMax(min,max);
		return imp2;
	}
	
	public static double[][] imageCorrelationMatrix(final ImagePlus imp, int nc){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		int[] x,y;
		double[][] CM = new double[n*n][nc*nc];
		int[] a,b;
		for (int i=0; i<n; i++){
			x = VectorConv.float2int((float[])(stack.getProcessor(i+1).convertToFloat().getPixels()));
			for (int j=0; j<i+1; j++){
				a = new int[nc];
				b = new int[nc];
				y = VectorConv.float2int((float[])(stack.getProcessor(j+1).convertToFloat().getPixels()));
				for (int k=0; k<w*h; k++){
					if (x[k]>=0&&x[k]<nc&&y[k]>=0&&y[k]<nc){
						CM[j+i*n][y[k]+nc*x[k]]+=0.5;
						CM[j+i*n][x[k]+nc*y[k]]+=0.5;
						a[x[k]]++;
						b[y[k]]++;
					}
				}
				int c;
				for (int k=0; k<nc; k++){
					for (int l=0; l<nc; l++){
						c = (a[k]+b[l])/2;
						if (c>0){
							CM[j+i*n][l+k*nc]/=c;
						}
					}
				}
				//CM[j+i*n]=VectorFun.div(CM[j+i*n],w*h);
			}
		}
		// update all symmetric location
		for (int i=0; i<n; i++){
			for (int j=i+1; j<n; j++){
				CM[j+i*n] = CM[i+j*n].clone();
			}
		}
		return CM;
	}
	
	public static double[][] correlationClusters(final int[] fitness, final ImagePlus imp){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		int nc = fitness.length/n;
		double[][] x = new double[nc][w*h];
		int[] y;
		int[] fitness2 = new int[fitness.length];
		ImageStack stack = imp.getImageStack();
		for (int i=0; i<n; i++){
			for (int j=0; j<nc; j++){
				fitness2[i*nc+fitness[i*nc+j]] = j;
			}
		}
		for (int i=0; i<n; i++){
			y = VectorConv.float2int((float[])(stack.getProcessor(i+1).convertToFloat().getPixels()));
			for (int j=0; j<w*h; j++){
				x[fitness2[y[j]+i*nc]][j]++;
			}
		}
		for (int i=0; i<nc; i++){
			x[i] = VectorFun.div(x[i],n);
		}
		return x;
	}
	
	public static ImagePlus convert2Stack(final double[][] x, int w, int h){
		int n= x.length;
		ImageStack stack = new ImageStack(w,h,n);
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		double t;
		for (int i=0; i<n; i++){
			stack.setPixels((new FloatProcessor(w,h,x[i])).getPixels(),i+1);
			t = VectorFun.min(x[i])[0];
			if (t<min){
				min = t;
			}
			t = VectorFun.max(x[i])[0];
			if (t>max){
				max = t;
			}
		}
		ImagePlus imp = new ImagePlus("Converted to Stack",stack);
		imp.getProcessor().setMinAndMax(min,max);
		return imp;
	}
	
	public static ImagePlus convert2Stack(final double[] x, int w, int h, int b){
		ImageStack stack = new ImageStack(w,h,b);
		double min = VectorFun.min(x)[0];
		double max = VectorFun.max(x)[0];
		double t;
		for (int i=0; i<b; i++){
			stack.setPixels((new FloatProcessor(w,h,VectorAccess.access(x,i*w*h,(i+1)*w*h))).getPixels(),i+1);
		}
		ImagePlus imp = new ImagePlus("Converted to Stack",stack);
		imp.getProcessor().setMinAndMax(min,max);
		return imp;
	}
	
	public static double[] stackPixelArray(ImagePlus imp){
		int n = imp.getNSlices();
		int w= imp.getWidth();
		int h= imp.getHeight();
		ImageStack stack = imp.getImageStack();
		double[] pixels = new double[w*h*n];
		for (int i=1; i<n+1; i++){
			VectorAccess.write(pixels,VectorConv.float2double((float[])stack.getProcessor(i).convertToFloat().getPixels()),(i-1)*w*h);
		}
		return pixels;
	}
	
	public static double[][] stack2PixelArrays(final ImagePlus imp){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		double[][] x = new double[n][w*h];
		for (int i=0; i<n; i++){
			x[i] = VectorConv.float2double((float[])(stack.getProcessor(i+1).convertToFloat().getPixels()));
		}
		return x;
	}
	
	public static double[] stack2PixelArray(final ImagePlus imp,int i){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		ImageStack stack = imp.getImageStack();
		double[] x = new double[w*h];
		x = VectorConv.float2double((float[])(stack.getProcessor(i+1).convertToFloat().getPixels()));
		return x;
	}
	
	public static double[] maxIndex(final double[][] x){
		double[][] temp = VectorAccess.flip(x);
		double[] y = new double[temp.length];
		for (int i=0; i<temp.length; i++){
			y[i] = VectorFun.max(temp[i])[1];
		}
		return y;
	}
	
	public static ImagePlus meanImage(final ImagePlus imp){
		int w = imp.getWidth();
		int h = imp.getHeight();
		int n = imp.getNSlices();
		double[][] x = stack2PixelArrays(imp);
		double[] y = new double[w*h];
		for (int i=0; i<n; i++){
			for (int j=0; j<w*h; j++){
				y[j]+=x[i][j];
			}
		}
		for (int i=0; i<w*h; i++){
			y[i]/=n;
		}
		ImageStack stack = new ImageStack(w,h);
		stack.addSlice(new FloatProcessor(w,h,y));
		ImagePlus meanimp = new ImagePlus("Mean",stack);
		return meanimp;
	}
		
	
	public static ImagePlus splitImage(final ImagePlus imp, int numclasses){
		int w = imp.getWidth();
		int h = imp.getHeight();
		double[] labelim = VectorConv.float2double((float[])(imp.getProcessor().convertToFloat().getPixels()));
		double[][] freqim = new double[numclasses][w*h];
		for (int i=0; i<numclasses; i++){
			freqim[i] = VectorConv.bool2double(BV.eq(labelim,i));
		}
		return convert2Stack(freqim,w,h);
	}
	
	public static ImagePlus merge(final ImagePlus... imp){
		int w= imp[0].getWidth();
		int h= imp[0].getHeight();
		ImageStack stack = new ImageStack(w,h);
		double min = Double.MAX_VALUE;
		double max = -Double.MAX_VALUE;
		double t;
		for (int j=0; j<imp.length; j++){
			ImageStack stack2 = imp[j].getImageStack();
			for (int i=0; i<imp[j].getNSlices(); i++){
				double[] x = VectorConv.float2double((float[])(stack2.getProcessor(i+1).convertToFloat().getPixels()));
				stack.addSlice("None", new FloatProcessor(w,h,x));
				t = VectorFun.min(x)[0];
				if (t<min){
					min = t;
				}
				t = VectorFun.max(x)[0];
				if (t>max){
					max = t;
				}
			}
		}
		ImagePlus impm = new ImagePlus("Merged Stack",stack);
		impm.getProcessor().setMinAndMax(min,max);
		return impm;
	}
	
	public static void mergei(ImagePlus... imp){
		int w= imp[0].getWidth();
		int h= imp[0].getHeight();
		double min = imp[0].getProcessor().getMin();
		double max = imp[0].getProcessor().getMax();
		double t;
		for (int j=1; j<imp.length; j++){
			ImageStack stack = imp[j].getImageStack();
			for (int i=0; i<imp[j].getNSlices(); i++){
				double[] x = VectorConv.float2double((float[])(stack.getProcessor(i+1).convertToFloat().getPixels()));
				imp[0].getImageStack().addSlice("None", new FloatProcessor(w,h,x));
				t = VectorFun.min(x)[0];
				if (t<min){
					min = t;
				}
				t = VectorFun.max(x)[0];
				if (t>max){
					max = t;
				}
			}
		}
		imp[0].getProcessor().setMinAndMax(min,max);
	}
		
	
	public static ImagePlus locateMinima3D(final ImagePlus imp, double fill, int connectivity){
		int w= imp.getWidth();
		int w2 = w+2;
		int h= imp.getHeight();
		int h2 = h+2;
		int n = imp.getNSlices();
		int a,b;
		ImageStack stack = imp.getImageStack();
		ImageStack stack2 = new ImageStack(w,h);
		int[] d = new int[9];
		d[0] = 0;
		d[1] = 1;
		d[2] = -1;
		d[3] = w2;
		d[4] = -w2;
		d[5] = 1+w2;
		d[6] = -1+w2;
		d[7] = 1-w2;
		d[8] = -1-w2;
		double[] st;
		double[] x1 = VectorFun.add(new double[w2*h2],fill);
		double[] x2 = FFTWrapper.pad2(new int[]{w,h},VectorConv.float2double((float[])(stack.getProcessor(1).convertToFloat().getPixels())),fill);
		double[] x3;
		boolean ymax, ymin;
		for (int i=0; i<n; i++){
			if (i<n-1){
				x3 = FFTWrapper.pad2(new int[]{w,h},VectorConv.float2double((float[])(stack.getProcessor(i+2).convertToFloat().getPixels())),fill);
			}
			else {
				x3 = VectorFun.add(new double[w2*h2],fill);
			}
			st = new double[w*h];
			for (int j=0; j<w*h; j++){
				a = ((int)(j/w)+1)*w2+(j%w)+1;
				ymin = false;
				ymax = false;
				// 6 way connectivity needs to be checked for all pixels anyway...
				if (x1[a]>x2[a]){
					ymax = true;
				}
				else {
					ymin = true;
				}
				if (x2[a]>x3[a]){
					ymin = true;
				}
				else {
					ymax = true;
				}
				// the standard six way case
				for (int k=1; k<5; k++){
					if (x2[a]>x2[a+d[k]]){
						ymin = true;
					}
					else {
						ymax = true;
					}
				}
				// 18 and 26 connectivity case
				if (connectivity==18||connectivity==26){
					// same level
					for (int k=5; k<9; k++){
						if (x2[a]>x2[a+d[k]]){
							ymin = true;
						}
						else {
							ymax = true;
						}
					}
					// different levels
					for (int k=1; k<5; k++){
						if (x1[a+d[k]]>x2[a]){
							ymax = true;
						}
						else {
							ymin = true;
						}
						if (x3[a+d[k]]>x2[a]){
							ymax = true;
						}
						else {
							ymin = true;
						}
					}
					
				}
				if (connectivity==26){
					// different levels
					for (int k=5; k<9; k++){
						if (x1[a+d[k]]>x2[a]){
							ymax = true;
						}
						else {
							ymin = true;
						}
						if (x3[a+d[k]]>x2[a]){
							ymax = true;
						}
						else {
							ymin = true;
						}
					}
				}
				if (!ymin){
					st[j] = -1;
				}
				if (!ymax){
					st[j] = 1;
				}
			}
			stack2.addSlice("None", new FloatProcessor(w,h,st));
			x1 = x2.clone();
			x2 = x3.clone();
		}
		ImagePlus imp2 = new ImagePlus("Converted to Stack",stack2);
		imp2.getProcessor().setMinAndMax(-1,1);
		return imp2;
	}
	
	public static double[] randError(final ImagePlus labels, final ImagePlus predictions, double value1, double value2){
		int w = labels.getWidth();
		int h = labels.getHeight();
		int n = labels.getNSlices();
		ImageStack stack = labels.getImageStack();
		ImageStack stack2 = predictions.getImageStack();
		double[] result = new double[n];
		double w2 = w;
		double h2 = h;
		double t = (w2*h2)*(w2*h2-1)*0.5;
		for (int l=1; l<n+1; l++){
			boolean[] xbin = BV.eq(VectorConv.float2double((float[])(stack.getProcessor(l).convertToFloat().getPixels())),value1);
			boolean[] ybin = BV.eq(VectorConv.float2double((float[])(stack2.getProcessor(l).convertToFloat().getPixels())),value2);
			double[] x = (new BLabel(w,h,VectorFun.add(VectorConv.bool2double(xbin),1),4)).getBlobNumber();
			double[] y = (new BLabel(w,h,VectorFun.add(VectorConv.bool2double(ybin),1),4)).getBlobNumber();
			int[][] regx = RegionFunctions.getRegions(x);
			int[][] regy = RegionFunctions.getRegions(y);
			// counts for pairs in same region in both images
			double c = 0;
			for (int i=0; i<regx.length; i++){
				double t2 = (double)regx[i].length*(regx[i].length-1)*0.5;
				double[] temp = VectorAccess.access(y,regx[i]);
				Arrays.sort(temp);
				double a = temp[0];
				double count = 1;
				for (int j=1; j<temp.length; j++){
					if (temp[j]>a){
						t2-=count*(count-1)*0.5;
						count = 1;
						a = temp[j];
					}
					else {
						count++;
					}
				}
				t2-=count*(count-1)*0.5;
				c+=t2;
			}
			double d = 0;
			for (int i=0; i<regy.length; i++){
				double t2 = (double)regy[i].length*(regy[i].length-1)*0.5;
				double[] temp = VectorAccess.access(x,regy[i]);
				Arrays.sort(temp);
				double a = temp[0];
				double count = 1;
				for (int j=1; j<temp.length; j++){
					if (temp[j]>a){
						t2-=count*(count-1)*0.5;
						count = 1;
						a = temp[j];
					}
					else {
						count++;
					}
				}
				t2-=count*(count-1)*0.5;
				d+=t2;
			}
			
			result[l-1] = (t-c-d)/t;
		}
		return result;
	}	

	public static double[] adaptedRandError(final ImagePlus labels, final ImagePlus predictions, double value1, double value2){
		int w = labels.getWidth();
		int h = labels.getHeight();
		int n = labels.getNSlices();
		ImageStack stack = labels.getImageStack();
		ImageStack stack2 = predictions.getImageStack();
		double[] result = new double[n];
		for (int l=1; l<n+1; l++){
			boolean[] xbin = BV.eq(VectorConv.float2double((float[])(stack.getProcessor(l).convertToFloat().getPixels())),value1);
			boolean[] ybin = BV.eq(VectorConv.float2double((float[])(stack2.getProcessor(l).convertToFloat().getPixels())),value2);
			boolean[] bin = BV.and(xbin,ybin);
			double[] x = (new BLabel(w,h,VectorConv.bool2double(xbin),4)).getBlobNumber();
			double[] y = (new BLabel(w,h,VectorConv.bool2double(ybin),4)).getBlobNumber();
			int[][] regx = RegionFunctions.getRegions(x);
			int[][] regy = RegionFunctions.getRegions(y);
			double s = VectorFun.sum(VectorConv.bool2double(xbin));
			double t = s*(s-1)*0.5;
			// counts for pairs in same region in both images
			double c = 0;
			for (int i=0; i<regx.length; i++){
				double t2 = (double)regx[i].length*(regx[i].length-1)*0.5;
				double[] temp = VectorAccess.access(y,regx[i]);
				Arrays.sort(temp);
				double a = temp[0];
				double count = 1;
				for (int j=1; j<temp.length; j++){
					if (temp[j]>a){
						if (a!=-1){
							t2-=count*(count-1)*0.5;
						}
						count = 1;
						a = temp[j];
					}
					else {
						count++;
					}
				}
				if (a!=-1){
					t2-=count*(count-1)*0.5;
				}
				c+=t2;
			}
			double d = 0;
			for (int i=0; i<regy.length; i++){
				//double t2 = (double)regy[i].length*(regy[i].length-1)*0.5;
				double[] temp = VectorAccess.access(x,regy[i]);
				Arrays.sort(temp);
				double b = 0;
				for (int j=0; j<temp.length; j++){
					if (temp[j]>-1){
						b++;
					}
				}
				double t2 = b*(b-1)*0.5;
				double a = temp[0];
				double count = 1;
				for (int j=1; j<temp.length; j++){
					if (temp[j]>a){
						if (a!=-1){
							t2-=count*(count-1)*0.5;
						}
						count = 1;
						a = temp[j];
					}
					else {
						count++;
					}
				}
				if (a!=-1){
					t2-=count*(count-1)*0.5;
				}
				d+=t2;
			}
			
			result[l-1] = (t-c-d)/t;
		}
		return result;
	}	
}