package flib.algorithms;

import java.util.TreeSet;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.LinkedList;
import flib.fftfunctions.FFTWrapper;
import flib.math.SortPair;
import flib.math.VectorConv;
import flib.math.BV;
import flib.math.RankSort;

public class SeededWatershed {
	// w: width of the image
	// h: height of the image
	private int w, h;
	// current watershed level
	private double level;
	// list of all watershed levels
	private TreeSet<Double> levels;
	// pixels making up the current border evaluation
	private TreeSet<SortPair> borderpixels;
	private ArrayList<ArrayList<Integer>> regions;
	// for each pixel the region it belongs to
	private int[] regionnumber;
	// initial seed points. These will be the local minima if not otherwise indicated
	private int[] seeds;
	// neighbor offset vector
	private int[] d;
	// the image copy used for the neighborhood search
	private double[] im;
	// a copy of the image which states if the current pixel has or belongs to a border
	private boolean[] border;
	
	public SeededWatershed(int w, int h, final double[] x, int connectivity, int[] seeds){
		this.w = w;
		this.h = h;
		this.seeds = seeds.clone();
		this.watershed2(x, connectivity);
	}
	
	public SeededWatershed(int w, int h, final double[] x, int connectivity){
		this.w = w;
		this.h = h;
		this.seeds = connectedMinima2(this.w,this.h,x,connectivity);
		this.watershed2(x, connectivity);
	}
	
	private void watershed2(final double[] x, int connectivity){
		int w2 = this.w+2;
		int h2 = this.h+2;
		this.regionnumber = new int[w2*h2];
		this.regions = new ArrayList<ArrayList<Integer>>();
		for (int i=0; i<this.seeds.length; i++){
			// initiate all regions
			this.regions.add(new ArrayList<Integer>());
		}
		// pad the image on both sides and both dimensions with the largest possible double value
		this.im = FFTWrapper.pad2(new int[]{w,h},x,Double.MAX_VALUE);
		// all max value pixels should not be considered
		this.border = BV.eq(im,Double.MAX_VALUE);
		// prepare the connectivity vector
		if (connectivity==4){
			this.d = new int[4];
			d[0] = -1;
			d[1] = 1;
			d[2] = -w2;
			d[3] = w2;
		}
		// or 8 way connectivity
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
		int a,b,c,loc,n1,n2;
		SortPair sp;
		// find all border pixels of the initial seed points
		this.borderpixels = new TreeSet<SortPair>();
		// the seed points don't belong to any boundary
		for (int i=0; i<this.seeds.length; i++){
			// recalculate to the zero padded image
			a = ((int)(this.seeds[i]/w)+1)*w2+(this.seeds[i]%w)+1;
			this.border[a] = true;
			this.regionnumber[a] = i+1;
			this.regions.get(i).add(this.seeds[i]);
		}
		for (int i=0; i<this.seeds.length; i++){
			// recalculate to the zero padded image
			a = ((int)(this.seeds[i]/w)+1)*w2+(this.seeds[i]%w)+1;
			// go through seeds' neighborhoods and mark boundary pixels
			for (int j=0; j<l; j++){
				b = a+d[j];
				if (!this.border[b]){
					this.border[b] = true;
					this.borderpixels.add(new SortPair(im[b],(double)b));
				}
			}
		}
		// check through the border pixels and assign according to their 
		// neighbor region values
		// neighbor evaluation
		boolean m;
		while (this.borderpixels.size()>0){
			// get the current lowest border pixel
			sp = this.borderpixels.first();
			// current location
			a = (int)sp.getOriginalIndex();
			//System.out.println(Integer.toString(a));
			// first neighbor region number
			c = this.regionnumber[a];
			// the current pixel only touch one region?
			m = true;
			// look at the region numbers of all neighbors
			for (int j=0; j<l; j++){
				// we don't care about the case where the neighboring value
				// is a watershed pixel, or hasn't been visited yet
				b = a+d[j];
				if (this.regionnumber[b]!=0){
					// has a label been assigned yet?
					if (c!=0){
						// compare to the current label estimate
						if (c!=this.regionnumber[b]){
							// this means we have found a watershed pixel
							// we're finished here
							m = false;
							break;
						}
					}
					else {
						// assign the current region label
						c = this.regionnumber[b];
					}
				}
			}
			// cleaning up
			if (m){
				// case of a normal region pixel
				this.regionnumber[a] = c;
				// only add border pixels to pixels belonging to regions
				for (int j=0; j<l; j++){
					b = a+d[j];
					// if the neighbors haven't been visited yet
					// add them to the border list
					if (!this.border[b]){
						this.border[b] = true;
						this.borderpixels.add(new SortPair(im[b],(double)b));
					}
				}
				// otherwise regionnumber[a] stays zero
			}
			// remove the current pixel from the border list
			this.borderpixels.remove(sp);
			// add new border pixels to the list
		}
	}
	/*
	public ArrayList<ArrayList<Integer>> getRegions(){
		return regions;
	}
	*/
	
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
	
	public static int[] connectedMinima2(int w, int h, final double[] x, int connectivity){
		int w2= w+2;
		int h2 = h+2;
		double[] im = FFTWrapper.pad2(new int[]{w,h},x,Double.MAX_VALUE);
		double[] im2 = new double[w*h];
		im2 = FFTWrapper.pad2(new int[]{w,h},im2,1);
		int a;
		int[] d;
		if (connectivity==4){
			d = new int[4];
			d[0] = -1;
			d[1] = 1;
			d[2] = -w2;
			d[3] = w2;
		}
		else {
			d = new int[8];
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
		int counter = 0;
		int[] mmax = new int[w*h];
		boolean m;
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				a = j*w2+i;
				m = true;
				// check if the pixel has been visited before
				if (im2[a]==0){
					// simple case: pixel is lower than its neighbors
					for (int k=0; k<l; k++){
						if (im[a+d[k]]<=im[a]){
							m = false;
						}
					}
					if (m){
						mmax[counter] = (j-1)*w+i-1;
						counter++;
					}
					// problematic case: pixel is lower or equal to all of its neighbors
					else{
						
						m = true;
						for (int k=0; k<l; k++){
							if(im[a]>im[a+d[k]]){
								m = false;
							}
						}
						if (m){
							im2[a] = 1;
							// now things become tedious
							// we have to go all pixels at equal height
							// if there was one neighbor among all neighbor pixels which had a lower value the first value won't count as a minimum
							LinkedList<Integer> neighborlist = new LinkedList<Integer>();
							neighborlist.add(a);
							while (!neighborlist.isEmpty()){
								int b = neighborlist.removeLast();
								im2[b] = 1;
								for (int k=0; k<l; k++){
									if (im[b]>im[b+d[k]]){
										m = false;
									}
									if (im2[b+d[k]]==0&&im[b]==im[b+d[k]]){
										neighborlist.add(b+d[k]);
									}
								}
							}
							if (m){
								mmax[counter] = (j-1)*w+i-1;
								counter++;
							}
						}
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