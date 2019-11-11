package flib.algorithms.regions;

import java.lang.Math;
import java.util.Arrays;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.RankSort;

public class RegionFunctions {
	
	static double IP = Math.sqrt(2)-1;
	
	public static int[][] getRegions(final double[] x){
		int m = (int)VectorFun.max(x)[0]+1;
		int[][] regions = new int[m][];
		int[] count = new int[m];
		for (int i=0; i<x.length; i++){
			if (x[i]>-1){
				count[(int)x[i]]++;
			}
		}
		for (int i=0; i<m; i++){
			regions[i] = new int[count[i]];
		}
		count = new int[m];
		for (int i=0; i<x.length; i++){
			if (x[i]>-1){
				regions[(int)x[i]][count[(int)x[i]]] = i;
				count[(int)x[i]]++;
			}
		}
		return regions;
	}
	
	public static int[] getKeys(final int[][] regions, final double[] x){
		int[] key_points = new int[regions.length];
		for (int i=0; i<key_points.length; i++){
			if (x[regions[i][0]]>0){
				key_points[i] = (new RankSort(VectorAccess.access(x,regions[i]),regions[i])).getRank()[regions[i].length-1];
			}
			else {
				key_points[i] = (new RankSort(VectorAccess.access(x,regions[i]),regions[i])).getRank()[0];
			}
		}
		return key_points;
	}
	
	public static double[][] keyDistances(int w, int h, final int[][] regions, final int[] keys){
		double[][] dist = new double[3][w*h];
		double[] orient = regionOrientation(w,h,regions);
		for (int i=0; i<regions.length; i++){
			int xk = keys[i]%w;
			int yk = keys[i]/w;
			double v = 1./Math.sqrt(regions[i].length);
			for (int j=0; j<regions[i].length; j++){
				int x = regions[i][j]%w-xk;
				int y = regions[i][j]/w-yk;
				dist[0][regions[i][j]] = Math.sqrt(x*x+y*y)*v;
				dist[1][regions[i][j]] = Math.atan2(y,x);
				dist[2][regions[i][j]] = dist[1][regions[i][j]]-orient[i]+Math.PI;
				while (dist[2][regions[i][j]]<0){
					dist[2][regions[i][j]]+=2*Math.PI;
				}
				while (dist[2][regions[i][j]]>=2*Math.PI){
					dist[2][regions[i][j]]-=2*Math.PI;
				}
			}
		}
		return dist;
	}
	
	public static double[] regionFill(int w, int h, final int[][] regions, final double[] val){
		double[] im = new double[w*h];
		for (int i=0; i<regions.length; i++){
			for (int j=0; j<regions[i].length; j++){
				im[regions[i][j]] = val[i];
			}
		}
		return im;
	}
	
	public static double[] regionMean(final int[][] regions, final double[] x){
		double[] val = new double[regions.length];
		for (int i=0; i<regions.length; i++){
			val[i] = VectorFun.sum(VectorAccess.access(x,regions[i]))/regions[i].length;
		}
		return val;
	}
	
	public static double[] regionVar(final int[][] regions, final double[] x){
		double[] val = new double[regions.length];
		for (int i=0; i<regions.length; i++){
			int s = regions[i].length;
			double[] y = VectorAccess.access(x,regions[i]);
			double m = VectorFun.sum(y)/s;
			val[i] = VectorFun.sum(VectorFun.mult(y,y))/s-m*m;
		}
		return val;
	}
	
	public static double[] regionMax(final int[][] regions, final double[] x){
		double[] val = new double[regions.length];
		for (int i=0; i<regions.length; i++){
			val[i] = VectorFun.max(VectorAccess.access(x,regions[i]))[0];
		}
		return val;
	}
	
	public static double[] regionMin(final int[][] regions, final double[] x){
		double[] val = new double[regions.length];
		for (int i=0; i<regions.length; i++){
			val[i] = VectorFun.max(VectorAccess.access(x,regions[i]))[0];
		}
		return val;
	}
	
	public static double[] regionSignedMaxAbs(final int[][] regions, final double[] x){
		double[] val = new double[regions.length];
		for (int i=0; i<regions.length; i++){
			double[] y = VectorAccess.access(x,regions[i]);
			val[i] =  y[(int)VectorFun.max(VectorFun.abs(y))[1]];
		}
		return val;
	}
	
	public static int[] regionCentroid(int w, int h, final int[][] regions){
		int[] cent = new int[regions.length];
		for (int i=0; i<cent.length; i++){
			int[] x = regions[i].clone();
			int[] y = regions[i].clone();
			for (int j=0; j<x.length; j++){
				x[j] = x[j]%w;
				y[j] = y[j]/w;
			}
			Arrays.sort(x);
			Arrays.sort(y);
			cent[i] = x[x.length/2]+y[x.length/2]*w;
		}
		return cent;
	}
	
	public static double[][] regionCovMat(int w, int h, final int[][] regions){
		double[][] covmat = new double[regions.length][3];
		for (int i=0; i<regions.length; i++){
			double x2=0, x=0, y2=0, y=0, xy=0;
			for (int j=0; j<regions[i].length; j++){
				int a = regions[i][j]%w;
				int b =regions[i][j]/w;
				x2+=a*a;
				x+=a;
				y2+=b*b;
				y+=b;
				xy+=a*b;
			}
			x/=regions[i].length;
			y/=regions[i].length;
			covmat[i][0] = x2/regions[i].length-x*x;
			covmat[i][1] = y2/regions[i].length-y*y;
			covmat[i][2] = xy/regions[i].length-x*y;
		}
		return covmat;
	}
	
	public static double[][] regionGeometry(final double[][] covmat){
		double[][] rpca = new double[covmat.length][3];
		for (int i=0; i<covmat.length; i++){
			// b^2-4ac has already been bunched together
			double lambda1 = Math.sqrt((covmat[i][0]-covmat[i][1])*(covmat[i][0]-covmat[i][1])+4*covmat[i][2]*covmat[i][2]);
			double lambda2 = (covmat[i][0]+covmat[i][1]-lambda1)*0.5;
			lambda1 = (covmat[i][0]+covmat[i][1]+lambda1)*0.5;
			double v = covmat[i][2]/(lambda1-covmat[i][0]);
			// pca direction angle
			rpca[i][0] = Math.atan2(1,v);
			if (lambda1-covmat[i][0]==0){
				rpca[i][0] = 0;
			}
			// elongation
			rpca[i][1] = Math.sqrt(lambda2/lambda1);
			if (lambda1==0){
				rpca[i][1] = 1;
			}
			// radius of gyration
			rpca[i][2] = covmat[i][0]+covmat[i][1];
		}
		return rpca;
	}
	
	public static double[] regionOrientation(int w, int h, final int[][] regions){
		double[][] covmat = regionCovMat(w,h,regions);
		double[] orient = new double[covmat.length];
		for (int i=0; i<covmat.length; i++){
			// b^2-4ac has already been bunched together
			double lambda1 = Math.sqrt((covmat[i][0]-covmat[i][1])*(covmat[i][0]-covmat[i][1])+4*covmat[i][2]*covmat[i][2]);
			double lambda2 = (covmat[i][0]+covmat[i][1]-lambda1)*0.5;
			lambda1 = (covmat[i][0]+covmat[i][1]+lambda1)*0.5;
			double v = covmat[i][2]/(lambda1-covmat[i][0]);
			// pca direction angle
			orient[i] = Math.atan2(1,v);
			if (lambda1-covmat[i][0]==0){
				orient[i] = 0;
			}
		}
		return orient;
	}
	
	// this is used to created a downscaled binary mask of a region shape for all regions in the list
	public static double[][] boxConversion(final double[][] regions_dist, final double[][] regions_ang ,int rx, int ry, double s){
		int l = regions_dist.length;
		double[][] box = new double[l][rx*ry];
		double u = rx/(2.*s);
		double ox = (rx-1)*0.5;
		double oy = (ry-1)*0.5;
		for (int i=0; i<l; i++){
			double t = (double)(rx*ry)/regions_dist[i].length;
			for (int j=0; j<regions_dist[i].length; j++){
				int x = (int)Math.round(regions_dist[i][j]*Math.cos(regions_ang[i][j])*u+ox);
				int y = (int)Math.round(regions_dist[i][j]*Math.sin(regions_ang[i][j])*u+oy);
				if (x>=0&&x<rx&&y>=0&&y<ry){
					//box[i][x+y*rx] = 1;
					box[i][x+y*rx]+=t;
				}
			}
		}
		return box;
	}
	
	// apply a set of filters to the binary mask of the points of a region
	public static double[][] applyBoxTemplate(final double[][] proj, final double[][] regions_dist, final double[][] regions_ang, int rx, int ry, double s){
		// number of dimensions of the filters
		int n = proj.length;
		int l = regions_dist.length;
		double u = rx/(2.*s);
		double ox = (rx-1)*0.5;
		double oy = (ry-1)*0.5;
		double[][] res = new double[l][n];
		// temporary list to remember which box locations have already
		// been visited
		//boolean[] visited = new boolean[rx*ry];
		for (int i=0; i<l; i++){
			double t = 1./(Math.sqrt(regions_dist[i].length));
			for (int j=0; j<regions_dist[i].length; j++){
				int x = (int)Math.round(regions_dist[i][j]*Math.cos(regions_ang[i][j])*u+ox);
				int y = (int)Math.round(regions_dist[i][j]*Math.sin(regions_ang[i][j])*u+oy);
				if (x>=0&&x<rx&&y>=0&&y<ry){
					int o = x+y*rx;
					/*
					if (!visited[o]){
						visited[o] = true;
						for (int k=0; k<n; k++){
							res[i][k]+=proj[k][o];
						}
					}
					*/
					for (int k=0; k<n; k++){
						res[i][k]+=proj[k][o]*t;
					}
				}	
			}
			// reset the visited values
			/*
			for (int j=0; j<regions_dist[i].length; j++){
				int x = (int)Math.round(regions_dist[i][j]*Math.cos(regions_ang[i][j])*u+ox);
				int y = (int)Math.round(regions_dist[i][j]*Math.sin(regions_ang[i][j])*u+oy);
				if (x>=0&&x<rx&&y>=0&&y<ry){
					visited[x+y*rx] = false;
				}
			}
			*/
		}
		return res;
	}
			
	
	public static double[][] valueFrequencies(final int[][] regions, int numval, final int[] x){
		double[][] val = new double[regions.length][numval];
		for (int i=0; i<regions.length; i++){
			for (int j=0; j<regions[i].length; j++){
				val[i][x[regions[i][j]]]++;
			}
			for (int j=0; j<numval; j++){
				val[i][j]/=regions[i].length;
			}
		}
		return val;
	}
	
	public static double[][] valueFrequencies(final int[][] regions, int numval, final double[] x){
		double[][] val = new double[numval][regions.length];
		for (int i=0; i<regions.length; i++){
			for (int j=0; j<regions[i].length; j++){
				int y = (int)x[regions[i][j]];
				if (y>=0&&y<numval){
					val[y][i]++;
				}
			}
			for (int j=0; j<numval; j++){
				val[j][i]/=regions[i].length;
			}
		}
		return val;
	}
	
	public static int[] maxFreqInd(final int[][] regions, final double[][] freq){
		int[] val = new int[regions.length];
		for (int i=0; i<regions.length; i++){
			val[i] = VectorFun.maxind(freq[i]);
		}
		return val;
	}
	
	// this function doesn't follow the standard region protocol
	public static double[] boundaryLength(int w, int h, final double[] z){
		int l = (int)VectorFun.max(z)[0]+1;
		double[] iq = new double[l];
		int[] d1 = {1,-1,w,-w};
		int[] d2 = {1+w,-1+w,1-w,-1-w};
		for (int i=0; i<z.length; i++){
			int x = i%w;
			int y = i/w;
			// if the image boundary is touched, the pixel is a straight boundary pixel
			if (x==0||x==w-1||y==0||y==h-1){
				iq[(int)z[i]]++;
			}
			else {
				boolean b = false;
				for (int j=0; j<d1.length; j++){
					if (z[i+d1[j]]!=z[i]){
						b = true;
						iq[(int)z[i]]++;
						break;
					}
				}
				if (!b){
					for (int j=0; j<d2.length; j++){
						if (z[i+d2[j]]!=z[i]){
							b = true;
							iq[(int)z[i]]+=IP;
							break;
						}
					}
				}
			}
		}
		return iq;
	}
}