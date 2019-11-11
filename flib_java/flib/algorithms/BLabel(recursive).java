package flib.algorithms;

import flib.fftfunctions.FFTWrapper;
import java.util.LinkedList;

public class BLabel {
	private int w, h, w2, h2, l, counter;
	private LinkedList<Integer>[] bloblist;
	private int[] blobnumber;
	private double[] im;
	private int [] d;
	
	public BLabel(int w, int h, final double[] x, int connectivity){
		this.w = w;
		this.h = h;
		this.blabel2(w,h,x,connectivity);
	}
	
	private void blabel2(int w, int h, final double[] x, int connectivity){
		this.w2 = this.w+2;
		this.h2 = this.h+2;
		this.bloblist = (LinkedList<Integer>[])new LinkedList[w*h];
		this.blobnumber = new int[w2*h2];
		for (int i=0; i<w2*h2; i++){
			blobnumber[i] = -1;
		}
		counter = 0;
		this.im = FFTWrapper.pad2(w,h,x,0,1);
		this.im = FFTWrapper.pad2(w2,h,im,1,1);
		int a,b;
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
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				a = j*w2+i;
				if (blobnumber[a]==-1){
					if (im[a]>0){
						blobnumber[a] = counter;
						bloblist[counter] = new LinkedList<Integer>();
						for (int k=0; k<l; k++){
							traverse(a+d[k]);
						}
						counter++;
					}
				}
			}
		}
	}
		
	private void traverse(int a){
		if (im[a]>0){
			if (blobnumber[a]==-1){
				blobnumber[a] = this.counter;
				bloblist[this.counter].add((int)(a/w2)*w+(a%w2)-1);
				for (int i=0; i<l; i++){
					traverse(a+d[i]);
				}
			}
		}
	}
	
	public LinkedList<Integer>[] getBlobList(){
		LinkedList<Integer>[] temp = (LinkedList<Integer>[])new LinkedList[counter-1];
		for (int i=0; i<counter-1; i++){
			temp[i] = bloblist[i];
		}
		return temp;
	}
	
	public double[] getBlobNumber(){
		double[] bn = new double[w*h];
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				bn[(j-1)*w+i-1] = this.blobnumber[j*w2+i];
			}
		}
		return bn;
	}
}