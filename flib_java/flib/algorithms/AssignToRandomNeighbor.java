package flib.algorithms;

import java.util.ArrayList;
import java.util.Random;
import java.util.Iterator;
import flib.fftfunctions.FFTWrapper;

public class AssignToRandomNeighbor {
	private int w,h;
	private double[] im, im2;
	private int [] d;
	
	public AssignToRandomNeighbor(int w, int h, final double[] x, int connectivity){
		this.w = w;
		this.h = h;
		int w2= this.w+2;
		int h2 = this.h+2;
		this.im = FFTWrapper.pad2(new int[]{w,h},x);
		this.im2 = FFTWrapper.pad2(new int[]{w,h},x);
		int len = x.length;
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
		int a,b;
		int l = d.length;
		Random r = new Random();
		// go through all pixels
		for (int i=0; i<len; i++){
			if (x[i]==0){ 
				a = ((int)(i/w)+1)*w2+(i%w)+1;
				ArrayList<Double> s = new ArrayList<Double>();
				for (int k=0; k<l; k++){
					b = a+d[k];
					if (im[b]!=0){
						s.add(im[b]);
					}
				}
				if (s.size()>0){
					int t = r.nextInt(s.size());
					this.im2[a] = s.get(t);
				}
			}
		}
	}
	
	public double[] getImage(){
		double[] bn = new double[w*h];
		int w2 = w+2;
		int h2 = h+2;
		for (int i=1; i<w2-1; i++){
			for (int j=1; j<h2-1; j++){
				bn[(j-1)*w+i-1] = this.im2[j*w2+i];
			}
		}
		return bn;
	}
}