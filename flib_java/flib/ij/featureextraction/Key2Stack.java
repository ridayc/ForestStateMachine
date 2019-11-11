package flib.ij.featureextraction;

import java.io.File;
import flib.io.ReadWrite;
import flib.math.VectorFun;
import flib.ij.io.ImageReader;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class Key2Stack {
	
	public static ImagePlus convert(String filename, String keydir){
		// find the input file's base name
		int n = filename.length();
		String base = filename.substring(0,n-".key".length());
		base = (new File(base)).getName();
		// find and open the key
		String keyname = keydir+File.separator+base+".key";
		String regname = keydir+File.separator+base+".tif";
		int[][] keys = (int[][])ReadWrite.readObject(keyname);
		double[][] val = (double[][])ReadWrite.readObject(filename);
		// find the number of layers
		int layers = val.length/keys.length;
		int[] size = ImageReader.getTiffSize(regname);
		// target stack
		ImageStack stack = new ImageStack(size[1],size[2]);
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		// go through the collection of keys
		for (int i=0; i<keys.length; i++){
			float[] reg = ImageReader.tiffLayerArray(regname,i);
			for (int j=0; j<layers; j++){
				double[] im = new double[reg.length];
				for (int k=0; k<reg.length; k++){
					im[k] = val[i*layers+j][(int)reg[k]];
				}
				double t = VectorFun.min(im)[0];
				if (t<min){
					min = t;
				}
				t = VectorFun.max(im)[0];
				if (t>max){
					max = t;
				}
				stack.addSlice("None", new FloatProcessor(size[1],size[2],im));
			}
		}
		ImagePlus imp = new ImagePlus("Key Regions",stack);
		imp.getProcessor().setMinAndMax(min,max);
		return imp;
	}
}