package flib.algorithms.geometry;

import java.util.ArrayList;
import java.util.TreeSet;
import java.lang.Math;
import flib.fftfunctions.FFTWrapper;
import flib.algorithms.DistanceTransform;
import flib.math.BV;
import flib.math.VectorConv;
import flib.math.VectorFun;

public class ShapeAnalysis {
	// calculate the center of mass of a 2d region
	public static double[] centerOfMass2D(int w, final ArrayList<Integer> region){
		// center of mass initialization
		double[] com = new double[2];
		// number of pixels in the region
		int l = region.size();
		for (int i=0; i<l; i++){
			com[0]+=(region.get(i)%w);
			com[1]+=(region.get(i)/w);
		}
		com[0]/=l;
		com[1]/=l;
		return com;
	}
	
	public static double radiusOfGyration2D(int w, final ArrayList<Integer> region, final double[] com){
		// number of pixels in the region
		int l = region.size();
		double x,y;
		// center of mass initialization
		double rg = 0;
		for (int i=0; i<l; i++){
			x = region.get(i)%w-com[0];
			y = region.get(i)/w-com[1];
			rg+=x*x+y*y;
		}
		rg/=l;
		return rg;
	}
	
	// separates regions found in the image. Regions are pixel groups with a value higher than -1
	public double[] separateRegions2D(int w, int h, final double[] im){
		int w2 = w+2;
		int h2 = h+2;
		double[] im_new = FFTWrapper.pad2(w,h,im,0,1,-1);
		im_new = FFTWrapper.pad2(w,h,im_new,1,1,-1);
		double[] x = im_new.clone();
		// 4-way connectivity
		int[] d = new int[4];
		d[0] = -1;
		d[1] = 1;
		d[2] = -w2;
		d[3] = w2;
		int a,b;
		int l = d.length;
		int len = im.length;
		// go through all pixels
		for (int i=0; i<len; i++){
			if (x[i]>-1){ 
				a = ((int)(i/w)+1)*w2+(i%w)+1;
				TreeSet<Double> s = new TreeSet<Double>();
				for (int k=0; k<l; k++){
					b = a+d[k];
					if (x[b]!=-1){
						s.add(x[b]);
					}
				}
				// check if pixels touch two different regions
				if (s.size()>1){
					im_new[a] = -1;
				}
			}
		}
		return im_new;
	}
	
	// assume that the regions have be separated already
	public double[] distanceField2D(int w, int h, final double[] im){
		return DistanceTransform.dt2d(w,h,VectorConv.bool2double(BV.gt(im,-1)),0);
	}
	
	public double sumOverRegion(final ArrayList<Integer> region, final double[] im){
		double sum = 0;
		for (int i=0; i<region.size(); i++){
			sum+=im[region.get(i)];
		}
		return sum;
	}
	
	public double[] regionCompactness2D(int w, final ArrayList<ArrayList<Integer>> regions){
		int l = regions.size();
		double[] rC = new double[l];
		double[] com = new double[2];
		for (int i=0; i<l; i++){
			com = centerOfMass2D(w,regions.get(i));
			// plus 4 is to correct very small pixel clusterings
			rC[i] = regions.get(i).size()/(2*Math.PI*radiusOfGyration2D(w,regions.get(i),com)+4);
		}
		return rC;
	}
	
	public double[] regionCorrosionCompactness2D(int w, int h, final ArrayList<ArrayList<Integer>> regions, final double[] im){
		int l = regions.size();
		double[] cC = new double[l];
		double[] temp = distanceField2D(w,h,separateRegions2D(w,h,im));
		double c = Math.pow(3,2.0/3)*Math.pow(Math.PI,1.0/3);
		double s;
		for (int i=0; i<l; i++){
			cC[i] = Math.pow(sumOverRegion(regions.get(i),temp),2.0/3)*c/regions.get(i).size();
		}
		return cC;
	}
	
	public double[][] distantPoints2D(int w, final ArrayList<Integer> region, double[] com, int numpoints){
		double[][] dp = new double[numpoints][2];
		int l = region.size();
		double[][] loc = new double[l][2];
		for (int i=0; i<l; i++){
			loc[i][0] = region.get(i)%w-com[0];
			loc[i][1] = region.get(i)/w-com[1];
		}
		double[] dist = new double[l];
		for (int i=0; i<l; i++){
			dist[i] = loc[i][0]*loc[i][0]+loc[i][1]*loc[i][1];
		}
		double[] m = new double[2];
		double a,b;
		for (int i=0; i<numpoints; i++){
			m = VectorFun.max(dist);
			// check that there are still points which have not yet been selected
			if (m[0]>0){
				dp[i] = loc[(int)m[1]].clone();
			}
			// go through all points in the region
			for (int j=0; j<l; j++){
				a = loc[(int)m[1]][0]-loc[j][0];
				b = loc[(int)m[1]][1]-loc[j][1];
				a = a*a+b*b;
				if (a<dist[j]){
					dist[j] = a;
				}
			}
		}
		return dp;
	}
	
	public double[][][] distantPoints2D(int w, final ArrayList<ArrayList<Integer>> regions, int numpoints){
		int l = regions.size();
		double[][][] dp = new double[l][numpoints][2];
		for (int i=0; i<l; i++){
			dp[i] = distantPoints2D(w,regions.get(i),centerOfMass2D(w,regions.get(i)),numpoints);
		}
		return dp;
	}
}