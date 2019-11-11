package flib.io;

import java.io.File;

public class TypeReader {
	public static String[] TIFF = {".tif"};
	public static String[] IMAGES = {".tif", ".tiff", ".jpg", ".bmp", ".png"};
	
	public static boolean isImage(final String filename, final String[] ext){
		boolean check = false;
		for (int i=0; i<ext.length; i++){
			if (filename.toLowerCase().endsWith(ext[i])){
				check = true;
			}
		}
		return check;
	}
	
	public static boolean isImage(final String filename){
		return isImage(filename,IMAGES);
	}
	
	public static boolean isTiff(final String filename){
		return isImage(filename,TIFF);
	}
	
	public static int[] fileNumber(final File[] directoryListing, final String[] ext){
		int imag_count = 0;
		for (int i=0; i<directoryListing.length; i++){
			String name = directoryListing[i].getAbsolutePath();
			if (isImage(name,ext)){
				imag_count++;
			}
		}
		int[] file_number = new int[imag_count];
		imag_count = 0;
		// list which file which image corresponds to
		for (int i=0; i<directoryListing.length; i++){
			String name = directoryListing[i].getAbsolutePath();
			if (isImage(name,ext)){
				file_number[imag_count] = i;
				imag_count++;
			}
		}
		return file_number;
	}
	
	public static int[] fileNumber(final File[] directoryListing){
		return fileNumber(directoryListing,IMAGES);
	}
	
	public static int[] tiffNumber(final File[] directoryListing){
		return fileNumber(directoryListing,TIFF);
	}
	
	public static String imageBase(final String filename){
		int n = filename.length();
		for (int i=0; i<TypeReader.IMAGES.length; i++){
			if (filename.toLowerCase().endsWith(IMAGES[i])){
				return filename.substring(0,n-IMAGES[i].length());
			}
		}
		return new String();
	}
}