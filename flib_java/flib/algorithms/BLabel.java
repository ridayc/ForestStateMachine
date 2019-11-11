package flib.algorithms;

import flib.fftfunctions.FFTWrapper;
import java.util.ArrayList;
import java.util.LinkedList;

/* This function is reminiscent of matlab's bwabel which labels connected regions
in a binary image. This function also works for images with discrete pre label values.
The algorithm will connect regions from pixels with the same larger than zero values.
The function requires the following variables:
w: width of the binary image
h: height of the image
x: the binary image
connectivity: an integer indicating if we're looking at 4 or 8 pixels connected neighborhoods
*/

public class BLabel {
	// w2, h2 are needed for boundary checking
	// l: number of pixels in a the local neighborhood
	// counter: number of disconnected regions
	private int w, h, b, w2, h2, b2, l, counter;
	// pixels contained in the blobs
	private ArrayList<ArrayList<Integer>> bloblist;
	// image containing the blob number of the corresponding pixel
	private int[] blobnumber;
	private double[] im;
	// offset vector of the connectivity neighborhood
	private int [] d;
	
	public BLabel(int w, int h, final double[] x, int connectivity){
		this.w = w;
		this.h = h;
		this.blabel2(w,h,x,connectivity);
	}
	
	public BLabel(int w, int h, int b, final double[] x, int connectivity){
		this.w = w;
		this.h = h;
		this.b = b;
		this.blabel3(w,h, b, x,connectivity);
	}
	
	private void blabel2(int w, int h, final double[] x, int connectivity){
		this.w2 = this.w+2;
		this.h2 = this.h+2;
		// the following initialization of an array of lists causes a warning
		this.bloblist = new ArrayList<ArrayList<Integer>>();
		this.blobnumber = new int[w2*h2];
		for (int i=0; i<w2*h2; i++){
			// initiate all blob numbers to a dummy value
			blobnumber[i] = -1;
		}
		// start off with 0 blobs
		counter = 0;
		// zero pad the image on both sides and both dimensions
		this.im = FFTWrapper.pad2(new int[]{w,h},x);
		int a,b;
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
		this.l = d.length;
		// go through all pixels
		// avoid the edges of the padded image
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				a = j*w2+i;
				// check that the pixels haven't been visited and are in the foreground
				if (blobnumber[a]==-1){
					if (im[a]>0){
						// If the pixel hasn't been labelled yet and is in the foreground
						// we need to create a new blob to put it into.
						blobnumber[a] = counter;
						bloblist.add(new ArrayList<Integer>());
						// The blob stack contains all connected pixels
						// which we still need to investigate
						LinkedList<Integer> BlobStack = new LinkedList<Integer>();
						BlobStack.add(a);
						// we need to adjust the pixel values 
						//when we store them into the blob list (because of padding)
						bloblist.get(counter).add((int)(a/w2-1)*w+(a%w2)-1);
						// keep evaluating pixels on the stack until it's empty
						while (!BlobStack.isEmpty()){
							a = BlobStack.removeLast();
							// visit the neighbor of each pixel in the stack
							for (int k=0; k<l; k++){
								b = a+d[k];
								// if the pixel neighbor is in the foreground and unlabelled:
								if(im[b]==im[a]&&blobnumber[b]==-1){
									// add it to the stack
									BlobStack.add(b);
									// assign it the current label
									blobnumber[b] = counter;
									// add the pixel to the current blob list
									bloblist.get(counter).add((int)(b/w2-1)*w+(b%w2)-1);
								}
							}
						}
						// when we've gone through all the connected pixels of the initial point
						// we increase the counter
						counter++;
					}
				}
			}
		}
	}
	
	private void blabel3(int w, int h, int b, final double[] x, int connectivity){
		this.w2 = this.w+2;
		this.h2 = this.h+2;
		this.b2 = this.b+2;
		// the following initialization of an array of lists causes a warning
		this.bloblist = new ArrayList<ArrayList<Integer>>();
		this.blobnumber = new int[w2*h2*b2];
		for (int i=0; i<w2*h2*b2; i++){
			// initiate all blob numbers to a dummy value
			blobnumber[i] = -1;
		}
		// start off with 0 blobs
		counter = 0;
		// zero pad the image on both sides and both dimensions
		this.im = FFTWrapper.pad3(new int[]{w,h,b},x);
		int a,c;
		// prepare the connectivity vector
		if (connectivity==6){
			this.d = new int[6];
			d[0] = -1;
			d[1] = 1;
			d[2] = -w2;
			d[3] = w2;
			d[4] = w2*h2;
			d[5] = -w2*h2;
		}
		if (connectivity==18){
			this.d = new int[18];
			d[0] = -1;
			d[1] = 1;
			d[2] = -w2;
			d[3] = w2;
			d[4] = w2*h2;
			d[5] = -w2*h2;
			d[6] = -1-w2;
			d[7] = 1-w2;
			d[8] = -1+w2;
			d[9] = 1+w2;
			d[10] = -1-w2*h2;
			d[11] = 1-w2*h2;
			d[12] = -1+w2*h2;
			d[13] = 1+w2*h2;
			d[14] = -w2-w2*h2;
			d[15] = w2-w2*h2;
			d[16] = -w2+w2*h2;
			d[17] = w2+w2*h2;
		}
		else {
			this.d = new int[26];
			d[0] = -1;
			d[1] = 1;
			d[2] = -w2;
			d[3] = w2;
			d[4] = w2*h2;
			d[5] = -w2*h2;
			d[6] = -1-w2;
			d[7] = 1-w2;
			d[8] = -1+w2;
			d[9] = 1+w2;
			d[10] = -1-w2*h2;
			d[11] = 1-w2*h2;
			d[12] = -1+w2*h2;
			d[13] = 1+w2*h2;
			d[14] = -w2-w2*h2;
			d[15] = w2-w2*h2;
			d[16] = -w2+w2*h2;
			d[17] = w2+w2*h2;
			d[18] = -1-w2-w2*h2;
			d[19] = 1-w2-w2*h2;
			d[20] = -1+w2-w2*h2;
			d[21] = 1+w2-w2*h2;
			d[22] = -1-w2+w2*h2;
			d[23] = 1-w2+w2*h2;
			d[24] = -1+w2+w2*h2;
			d[25] = 1+w2+w2*h2;
		}
		this.l = d.length;
		// go through all pixels
		// avoid the edges of the padded image
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				for (int k=1; k<b2-1; k++){ 
					a = k*w2*h2+j*w2+i;
					// check that the pixels haven't been visited and are in the foreground
					if (blobnumber[a]==-1){
						if (im[a]>0){
							// If the pixel hasn't been labelled yet and is in the foreground
							// we need to create a new blob to put it into.
							blobnumber[a] = counter;
							bloblist.add(new ArrayList<Integer>());
							// The blob stack contains all connected pixels
							// which we still need to investigate
							LinkedList<Integer> BlobStack = new LinkedList<Integer>();
							BlobStack.add(a);
							// we need to adjust the pixel values 
							//when we store them into the blob list (because of padding)
							bloblist.get(counter).add((int)(a/(w2*h2)-1)*w*h+(int)((a%(w2*h2))/w2-1)*w+(a%w2)-1);
							// keep evaluating pixels on the stack until it's empty
							while (!BlobStack.isEmpty()){
								a = BlobStack.removeLast();
								// visit the neighbor of each pixel in the stack
								for (int m=0; m<l; m++){
									c = a+d[m];
									// if the pixel neighbor is in the foreground and unlabelled:
									if(im[c]==im[a]&&blobnumber[c]==-1){
										// add it to the stack
										BlobStack.add(c);
										// assign it the current label
										blobnumber[c] = counter;
										// add the pixel to the current blob list
										bloblist.get(counter).add((int)(c/(w2*h2)-1)*w*h+(int)((c%(w2*h2))/w2-1)*w+(c%w2)-1);
									}
								}
							}
							// when we've gone through all the connected pixels of the initial point
							// we increase the counter
							counter++;
						}
					}
				}
			}
		}
	}
	
	// returns an array of the pixels contained in each blob
	public ArrayList<ArrayList<Integer>> getBlobList(){
		return bloblist;
	}
	
	// returns an image with each pixel having the label of the blob it belongs to
	public double[] getBlobNumber(){
		double[] bn = new double[w*h];
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				bn[(j-1)*w+i-1] = this.blobnumber[j*w2+i];
			}
		}
		return bn;
	}
	
	// returns a 3d image with each pixel having the label of the blob it belongs to
	public double[] getBlobNumber3(){
		double[] bn = new double[w*h*b];
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				for (int k=1; k<b2-1; k++){
					bn[(k-1)*w*h+(j-1)*w+i-1] = this.blobnumber[k*w2*h2+j*w2+i];
				}
			}
		}
		return bn;
	}
	
	public double[] getBlobNumber3_2(){
		double[] bn = new double[w*h*b];
		for (int i=0; i<this.bloblist.size(); i++){
			for (int j=0; j<this.bloblist.get(i).size(); j++){
				bn[this.bloblist.get(i).get(j)] = 1;
			}
		}
		return bn;
	}
}