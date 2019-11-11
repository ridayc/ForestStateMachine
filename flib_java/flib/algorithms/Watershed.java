package flib.algorithms;

import flib.fftfunctions.FFTWrapper;
import flib.math.RankSort;
import java.util.ArrayList;

/* This class contains an implementation of the 2D watershed algorithm. The function
doesn't deal well with discrete/equal pixel values (noise regions will be introduced),
and therefore it is left up to the user of the function to preprocess their image that it becomes smooth.
The function requires the following variables:
w: width of the binary image
h: height of the image
x: the binary image
connectivity: an integer indicating if we're looking at 4 or 8 pixels connected neighborhoods
*/

public class Watershed {
	// counter: number of watershed regions
	// zero pixels are waterrshed boundaries
	private int w, h, counter;
	// watershed regions and their contained pixels
	private ArrayList<ArrayList<Integer>> regions;
	// image with the pixels labelled according to region
	// location of the minima in the image
	private int[] regionnumber, minima;
	private double[] im;
	// offset vector of the connectivity neighborhood
	private int [] d;
	
	public Watershed(int w, int h, final double[] x, int connectivity){
		this.w = w;
		this.h = h;
		this.watershed2(x,connectivity);
	}

	private void watershed2(final double[] x, int connectivity){
		// w2, h2 are needed for boundary checking
		int w2= this.w+2;
		int h2 = this.h+2;
		// image size
		int len = x.length;
		// image containing the region number of the corresponding pixel
		this.regionnumber = new int[w2*h2];
		this.minima = new int[len];
		// the following initialization of an array of lists causes a warning
		this.regions = new ArrayList<ArrayList<Integer>>();
		for (int i=0; i<w2*h2; i++){
			// initiate all blob numbers to the watershed pixel value
			regionnumber[i] = 0;
		}
		// pad the image on both sides and both dimensions with the largest possible double value
		this.im = FFTWrapper.pad2(new int[]{w,h},x,Double.MAX_VALUE);
		// start off with 0 blobs
		this.counter = 0;
		//
		// sort all pixels according to value and store their rank order
		int [] rank = (new RankSort(x)).getRank();
		int a, b, c, n1, n2;
		boolean m;
		// prepare the connectivity vector
		if (connectivity==4){
			this.d = new int[4];
			d[0] = -1;
			d[1] = 1;
			d[2] = -w2;
			d[3] = w2;
		}
		else {
			this.d = new int[8];
			d[0] = -1;
			d[1] = 1;
			d[2] = -w2;
			d[3] = w2;
			d[4] = -w2-1;
			d[5] = -w2+1;
			d[6] = w2-1;
			d[7] = w2+1;
		}
		int l = d.length;
		// go through all pixels
		// implicitly avoid the edges of the padded image
		// go through the pixels in rank order!
		for (int i=0; i<len; i++){
			// m states if the current point is a local minimum (smaller/equal to its neighborhood)
			m = true;
			// the following variables are used to determine if the current pixel is a watershed boundary
			n1 = 0;
			n2 = 0;
			// get the current pixel location
			c = rank[i];
			// recalculate to the zero padded image
			a = ((int)(c/w)+1)*w2+(c%w)+1;
			for (int k=0; k<l; k++){
				b = a+d[k];
				if (im[a]>im[b]){
					m = false;
				}
				if (regionnumber[b]!=0){
					if (n1==0){
						n1 = regionnumber[b];
					}
					else if (n1!=regionnumber[b]){
						n2 = regionnumber[b];
					}
				}
			}
			if (m) {
				regions.add(new ArrayList<Integer>());
				minima[counter] = c;
				regions.get(counter).add(c);
				counter ++;
				regionnumber[a] = counter;
			}
			else if (n2==0&&n1!=0){
				regionnumber[a] = n1;
				regions.get(n1-1).add(c);
			}
		}
	}
	
	public ArrayList<ArrayList<Integer>> getRegions(){
		return regions;
	}
	
	public int[] getMinima(){
		int[] m = new int[counter];
		for (int i=0; i<counter; i++){
			m[i] = minima[i];
		}
		return m;
	}
	
	public int[] getRegionNumber(){
		int[] bn = new int[w*h];
		int w2 = w+2;
		int h2 = h+2;
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				bn[(j-1)*w+i-1] = this.regionnumber[j*w2+i];
			}
		}
		return bn;
	}
	
	public static int[] findMinima2(int w, int h, final double[] x, int connectivity){
		int w2= w+2;
		int h2 = h+2;
		double[] im = FFTWrapper.pad2(new int[]{w,h},x,Double.MAX_VALUE);
		int counter = 0;
		int[] mmax = new int[w*h];
		int a;
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				a = j*w2+i;
				if (connectivity==4) {
					if (im[a]<=im[a-1]&&im[a]<=im[a+1]&&im[a]<=im[a-w2]&&im[a]<=im[a+w2]){
						mmax[counter] = (j-1)*w+i-1;
						counter++;
					}
				}
				else {
					if (im[a]<=im[a-1]&&im[a]<=im[a+1]&&im[a]<=im[a-w2]&&im[a]<=im[a+w2]&&im[a]<=im[a-w2-1]&&im[a]<=im[a-w2+1]&&im[a]<=im[a+w2-1]&&im[a]<=im[a+w2+1]){
						mmax[counter] = (j-1)*w+i-1;
						counter++;
					}
				}
			}
		}
		int[] b = new int[counter];
		for (int i=0; i<counter; i++){
			b[i] = mmax[i];
		}
		return b;
	}
}