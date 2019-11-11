package flib.ij.featureextraction;

import java.io.File;
import flib.io.TypeReader;
import ij.io.DirectoryChooser;
import ij.IJ;
import ij.ImagePlus;

public class FileInterpreter implements
java.io.Serializable {
	// image file names in the origin folder
	// origin folder in the parent directory
	private String[] file_names, base_names;
	// image pixel information per image
	// first field: number of layers
	// second field: number of pixels per layer
	// third field: cumulative sum of all pixels over all images
	private int[][] pixel_inf;
	
	public FileInterpreter(final String location){
		String targetdir = "";
		// check if we need to let the user choose a directory
		// containing the images to be referenced
		if (location.equals("")){
			targetdir = (new DirectoryChooser("Choose a folder")).getDirectory();
			if (targetdir.isEmpty()){
				IJ.log("User canceled the FileInterpreter dialog! Bye!");
				return;
			}
		}
		else {
			targetdir = location;
		}
		File[] directoryListing = (new File(targetdir)).listFiles();
		int counter = 0;
		for (int i=0; i<directoryListing.length; i++){
			String name = directoryListing[i].getAbsolutePath();
			// find all images in this directory
			if (TypeReader.isImage(name)){
				counter++;
			}
		}
		file_names = new String[counter];
		base_names = new String[counter];
		pixel_inf = new int[3][counter];
		counter = 0;
		int sum = 0;
		for (int i=0; i<directoryListing.length; i++){
			String name = directoryListing[i].getAbsolutePath();
			if (TypeReader.isImage(name)){
				file_names[counter] = new String(name);
				base_names[counter] = TypeReader.imageBase(directoryListing[i].getName());
				// read the image and get the number of layers and pixels
				ImagePlus imp = IJ.openImage(name);
				int w = imp.getWidth();
				int h = imp.getHeight();
				int n = imp.getNSlices();
				pixel_inf[0][counter] = n;
				pixel_inf[1][counter] = w*h;
				sum+=w*h*n;
				pixel_inf[2][counter] = sum;
				counter++;
			}
		}
	}
	
	public String[] getNames(){
		return this.file_names;
	}
	
	public String[] getBases(){
		return this.base_names;
	}
	
	public int[][] getInfo(){
		return this.pixel_inf;
	}
}