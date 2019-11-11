package flib.ij.io;

import java.io.File;
import java.io.RandomAccessFile;
import java.io.IOException;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.Opener;
import ij.io.FileInfo;
import ij.io.TiffDecoder;

public class ImageReader {
	public static int[] getTiffSize(final String filename){
		TiffDecoder td = new TiffDecoder((new File(filename)).getParent(),(new File(filename)).getName());
		FileInfo[] info = null;
		try {
			info = td.getTiffInfo();
		} catch (IOException e) {
			String msg = e.getMessage();
			if (msg==null||msg.equals("")) msg = ""+e;
			IJ.error("Open TIFF", msg);
		}
		FileInfo fi = info[0];
		int n = 0;
		if (info.length==1 && fi.nImages>1) {
			n = fi.nImages;
		}
		else {
			n = info.length;
		}
		int[] size = new int[3];
		// number of images
		size[0] = n;
		// width and height of the image
		size[1] = fi.width;
		size[2] = fi.height;
		return size;
	}
	
	public static int[] getImageSize(final String filename){
		int[] size = new int[3];
		if ((new Opener()).getFileType(filename)==Opener.TIFF){
			size = getTiffSize(filename);
		}
		else {
			ImagePlus imp = IJ.openImage(filename);
			size[0] = 1;
			size[1] = imp.getWidth();
			size[2] = imp.getHeight();
		}
		return size;
	}
	
	public static float[] tiffLayerArray(final String filename, int layer){
		return (float[])(new Opener()).openTiff(filename,layer+1).getImageStack().getProcessor(1).convertToFloatProcessor().getPixels();
	}
	
	// this version is specifically for reading of gray 32bit float tiffs which were prepared within this framework
	// exception handling... is essentially not given
	// layer offset is different from the above code
	public static double[] tiffLayerArray(String filename, int layer, int start, int stop) throws Exception{
		int len = stop-start;
		TiffDecoder td = new TiffDecoder((new File(filename)).getParent(),(new File(filename)).getName());
		FileInfo[] info = null;
		try {
			info = td.getTiffInfo();
		} catch (IOException e) {
			String msg = e.getMessage();
			if (msg==null||msg.equals("")) msg = ""+e;
			IJ.error("Open TIFF", msg);
		}
		FileInfo fi = info[0];
		if (info.length==1 && fi.nImages>1) {
			if (layer<1 || layer>fi.nImages)
				throw new IllegalArgumentException("N out of 1-"+fi.nImages+" range");
			long size = fi.width*fi.height*fi.getBytesPerPixel();
			fi.longOffset = fi.getOffset() + (layer-1)*(size+fi.gapBetweenImages);
			fi.offset = 0;
			fi.nImages = 1;
		} else {
			if (layer<1 || layer>info.length)
				throw new IllegalArgumentException("N out of 1-"+info.length+" range");
			fi.longOffset = info[layer-1].getOffset();
			fi.offset = 0;
			fi.stripOffsets = info[layer-1].stripOffsets; 
			fi.stripLengths = info[layer-1].stripLengths; 
		}
		RandomAccessFile raf = new RandomAccessFile(filename,"r");
		// set the correct offset in the image
		raf.seek(fi.longOffset+start*4);
		byte[] b = new byte[len*4];
		double[] pixels = new double[len];
		// read the pixel values from the file
		raf.read(b);
		int tmp;
		int j = 0;
		if (fi.intelByteOrder){
			for (int i=0; i<len; i++) {
				tmp = (int)(((b[j+3]&0xff)<<24) | ((b[j+2]&0xff)<<16) | ((b[j+1]&0xff)<<8) | (b[j]&0xff));
				if (fi.fileType==FileInfo.GRAY32_FLOAT)
					pixels[i] = Float.intBitsToFloat(tmp);
				else if (fi.fileType==FileInfo.GRAY32_UNSIGNED)
					pixels[i] = (float)(tmp&0xffffffffL);
				else
					pixels[i] = tmp;
				j += 4;
			}
		}
		else {
			for (int i=0; i<len; i++) {
				tmp = (int)(((b[j]&0xff)<<24) | ((b[j+1]&0xff)<<16) | ((b[j+2]&0xff)<<8) | (b[j+3]&0xff));
				if (fi.fileType==FileInfo.GRAY32_FLOAT)
					pixels[i] = Float.intBitsToFloat(tmp);
				else if (fi.fileType==FileInfo.GRAY32_UNSIGNED)
					pixels[i] = (float)(tmp&0xffffffffL);
				else
					pixels[i] = tmp;
				j += 4;
			}
		}
		raf.close();
		return pixels;
	}
	
	public static float[] imageArray(final String filename){
		return (float[])IJ.openImage(filename).getImageStack().getProcessor(1).convertToFloat().getPixels();
	}
}