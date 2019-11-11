package flib.algorithms.sampling;

import java.lang.Math;
import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Iterator;
import flib.math.SortPair;

public class NeighborhoodSample {
	// this function takes a single point in 2d and finds all coordinates in its neighborhood
	// according to the coordinates given in shape. Non-existing points are dealt with 
	// according to type
	public static int[] shapeNeighbor2d(final int[][] shape, int w, int h, int x, int y, int type){
		// w: image width
		// h: image height
		int l = shape.length;
		// periodic case
		int[] coord = new int[l];
		int a,b;
		if (type==1){
			for (int i=0; i<l; i++){
				// hopefully not too overkill...
				a = x+shape[i][0];
				b = y+shape[i][1];
				coord[i] = (int)((Math.signum(a)*(Math.abs(a)%w)+w)%w+w*((Math.signum(b)*(Math.abs(b)%h)+h)%h));
			}
		}
		// non-periodic fill case
		else {
			for (int i=0; i<l; i++){
				a = x+shape[i][0];
				b = y+shape[i][1];
				if (a>=0&&a<w&&b>=0&&b<h){
					coord[i] = a+b*w;
				}
				else {
					// indicator that we will need the fill value
					coord[i] = -1;
				}
			}
		}
		return coord;
	}

	public static double[][] sample2d(final int[] points, int w, int h, final int[][] shape, int type, double fill, final double[]... im){
		int n = points.length;
		int l = shape.length;
		int d = im.length;
		double[][] samplepoints = new double[n][l*d];
		//int[] coord = new int[l];
		/*
		for (int i=0; i<n; i++){
			coord = shapeNeighbor2d(shape, w, h, points[i]%w, points[i]/w, type);
			for (int j=0; j<d; j++){
				for (int k=0; k<l; k++){
					if (coord[k]==-1){
						samplepoints[i][j*l+k] = fill;
					}
					else {
						samplepoints[i][j*l+k] = im[j][coord[k]];
					}
				}
			}
		}
		*/
		for (int i=0; i<d; i++){
			for (int k=0; k<n; k++){
				for (int j=0; j<l; j++){
					int coord = -1;
					if (type==1){
						int a =  points[k]%w+shape[j][0];
						int b = points[k]/w+shape[j][1];
						coord = (int)((Math.signum(a)*(Math.abs(a)%w)+w)%w+w*((Math.signum(b)*(Math.abs(b)%h)+h)%h));
					}
					// non-periodic fill case
					else {
						int a =  points[k]%w+shape[j][0];
						int b = points[k]/w+shape[j][1];
						if (a>=0&&a<w&&b>=0&&b<h){
							coord = a+b*w;
						}
					}
					if (coord==-1){
						samplepoints[k][i*l+j] = fill;
					}
					else {
						samplepoints[k][i*l+j] = im[i][coord];
					}
				}
			}
		}
		return samplepoints;
	}
	
	// rectangle case
	public static double[][] sampleRectangle(final int[] points, int w, int h, int rx, int ry, int dist, int type, double fill, final double[]... im){
		int[][] shape = new int[(2*rx+1)*(2*ry+1)][2];
		for (int i=0; i<2*rx+1; i++){
			for (int j=0; j<2*ry+1; j++){
				shape[i+(2*rx+1)*j][0] = (i-rx)*dist;
				shape[i+(2*rx+1)*j][1] = (j-ry)*dist;
			}
		}
		return sample2d(points,w,h,shape,type,fill,im);
	}
	
	public static int[][] circleCoord(int r){
		return circleCoord((double)r);
	}
	
	public static int[][] circleCoord(double r){
		ArrayList<int[]> c = new ArrayList<int[]>();
		for (int i=-(int)r; i<=(int)r; i++){
			for (int j=-(int)r; j<=(int)r; j++){
				if (i*i+j*j<=r*r){
					c.add(new int[]{i,j});
				}
			}
		}
		int l = c.size();
		int[][]shape = new int[l][2];
		for (int i=0; i<l; i++){
			shape[i] = c.get(i).clone();
		}
		return shape;
	}
	
	// sampling discrete points among multiple spiral arms
	public static int[][] spiralCoord(double phi, double rad, double b, double m, double n, int arms){
		// variables:
		// a: angular stretching constant
		// phi: angular increment
		// rad: maximal radius to go up to
		// b: radial stretching constant
		// m: angular exponent
		// n: radial exponent
		// arms: number of arms to use for the spiral
		TreeSet<SortPair> coord = new TreeSet<SortPair>();
		double r, p, o;
		int x, y, j;
		for (int i=0; i<arms; i++){
			r = 0;
			o = 2*Math.PI/arms*i;
			p = 0;
			j = 0;
			while (r<=rad){
				x = (int)Math.round(Math.cos(o+p)*r);
				y = (int)Math.round(Math.sin(o+p)*r);
				coord.add(new SortPair(x,y));
				r = b*Math.pow(j,n);
				p = Math.pow(j*phi,m);
				j++;
			}
		}
		int l = coord.size();
		int[][] shape = new int[l][2];
		Iterator<SortPair> itr = coord.iterator();
		SortPair temp;
		int count = 0;
		while(itr.hasNext()){
			temp = itr.next();
			shape[count][0] = (int)temp.getValue();
			shape[count][1] = (int)temp.getOriginalIndex();
			count++;
		}
		return shape;
	}		
	
	public static double[][] sampleDisk(final int[] points, int w, int h, int r, int type, double fill, final double[]... im){
		int[][] shape = circleCoord(r);
		return sample2d(points,w,h,shape,type,fill,im);
	}
	
	// square case... still kept for the old format
	public static double[][] sample2d(final int[] points, final double[] im, int w, int h, int dist, int n, double fill){
		return sampleRectangle(points,w,h,n,n,dist,0,fill,im);
	}
	
	public static double[][] sample2d(final int[] points, final double[][] im, int w, int h, int dist, int n, double fill){
		return sampleRectangle(points,w,h,n,n,dist,0,fill,im);
	}
		
	/*public static double[][] sample2d(final int[] points, final double[] im, int w, int h, int dist, int n, double fill){
		// points: the list of the point location of which we sample the neighborhood
		// im: the image which we sample the pixels from
		// dist: the distance in pixels between sampling points
		// n: the number of points to travel from the center point
		// fill: the fill value we use in case a pixel isn't in the image frame
		// w: image width
		// h: image height
		// the locations of all neighborhood pixels relative to the center point
		int[][] neighborhood = new int[(2*n+1)*(2*n+1)][2];
		for (int i=0; i<2*n+1; i++){
			for (int j=0; j<2*n+1; j++){
				neighborhood[i+j*(2*n+1)][0] = (i-n)*dist;
				neighborhood[i+j*(2*n+1)][1] = (j-n)*dist;
			}
		}
		int x,y,a,b;
		// list of sampled points
		double[][] samplepoints = new double[points.length][neighborhood.length];
		// go through all points
		for (int i=0; i<points.length; i++){
			x = points[i]%w;
			y = points[i]/w;
			// go through each neighborhoodpoint
			for (int j=0; j<neighborhood.length; j++){
				a = x+neighborhood[j][0];
				b = y+neighborhood[j][1];
				if (a>=0&&a<w&&b>=0&&b<h){
					samplepoints[i][j] = im[points[i]+neighborhood[j][0]+neighborhood[j][1]*w];
				}
				else {
					samplepoints[i][j] = fill;
				}
			}
		}
		return samplepoints;
	}
	
	// here comes the case where we have multiple images which corresponding pixels
	public static double[][] sample2d(final int[] points, final double[][] im, int w, int h, int dist, int n, double fill){
		// points: the list of the point location of which we sample the neighborhood
		// im: the image which we sample the pixels from
		// w: image width
		// h: image height
		// dist: the distance in pixels between sampling points
		// n: the number of points to travel from the center point
		// fill: the fill value we use in case a pixel isn't in the image frame
		
		// the locations of all neighborhood pixels relative to the center point
		int[][] neighborhood = new int[(2*n+1)*(2*n+1)][2];
		for (int i=0; i<2*n+1; i++){
			for (int j=0; j<2*n+1; j++){
				neighborhood[i+j*(2*n+1)][0] = (i-n)*dist;
				neighborhood[i+j*(2*n+1)][1] = (j-n)*dist;
			}
		}
		int x,y,a,b;
		int d = im[0].length;
		// list of sampled points
		double[][] samplepoints = new double[points.length][neighborhood.length*d];
		// go through all points
		for (int i=0; i<points.length; i++){
			x = points[i]%w;
			y = points[i]/w;
			// go through each neighborhoodpoint
			for (int j=0; j<neighborhood.length; j++){
				a = x+neighborhood[j][0];
				b = y+neighborhood[j][1];
				if (a>=0&&a<w&&b>=0&&b<h){
					for (int k=0; k<d; k++){
						samplepoints[i][j*d+k] = im[points[i]+neighborhood[j][0]+neighborhood[j][1]*w][k];
					}
				}
				else {
					for (int k=0; k<d; k++){
						samplepoints[i][j*d+k] = fill;
					}
				}
			}
		}
		return samplepoints;
	}*/
}