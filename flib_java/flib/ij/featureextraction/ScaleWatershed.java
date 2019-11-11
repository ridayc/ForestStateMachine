package flib.ij.featureextraction;

import java.io.File;
import java.util.Random;
import flib.io.ReadWrite;
import flib.io.TypeReader;
import flib.ij.io.ImageReader;
import flib.ij.featureextraction.FolderNames;
import flib.ij.featureextraction.FileInterpreter;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.VectorAccess;
import flib.algorithms.regions.ScaleRegions;
import flib.algorithms.regions.RegionFunctions;

public class ScaleWatershed {

	public static void run(String orig, String keydir, String targetdir, int type, int layers){
		// find all fiji readable files inside the normalized scales directory
		FileInterpreter FI = new FileInterpreter(orig);
		new File(targetdir).mkdirs();
		// make to store the key location file in the folder
		ReadWrite.writeObject(targetdir+File.separator+FolderNames.KEYPARAM,keydir);
		for (int i=0; i<FI.getBases().length; i++){
			String name = targetdir+File.separator+FI.getBases()[i]+".key";
			if (!(new File(name)).exists()){
				// watershed key locations
				int[][] keys = (int[][])ReadWrite.readObject(keydir+File.separator+FI.getBases()[i]+".key");
				// content storage for the regions over all scales
				double[][] val;
				int n = FI.getInfo()[0][i];
				if (type==0||type==1||type==2||type==3){
					val = new double[keys.length*n][];
				}
				else {
					val = new double[keys.length*layers*n][];
				}
				// go through all subimages
				for (int j=0; j<n; j++){
					// here things will start to depend on the type of 
					// investigation at hand
					// we assume there is only one image to be investigated at this point
					double[] im;
					if (TypeReader.isTiff(FI.getNames()[i])){
						im = VectorConv.float2double(ImageReader.tiffLayerArray(FI.getNames()[i],j));
					}
					else {
						im = VectorConv.float2double(ImageReader.imageArray(FI.getNames()[i]));
					}
					// go through all key sets
					for (int k=0; k<keys.length; k++){
						double[] im2 = VectorConv.float2double(ImageReader.tiffLayerArray(keydir+File.separator+FI.getBases()[i]+TypeReader.TIFF[0],k));
						// find all regions for the current scale
						int[][] reg = RegionFunctions.getRegions(im2);
						// simple case: fill in the key point value
						// layers better be one...
						if (type==0){
							val[k*n+j] = new double[keys[k].length];
							for (int l=0; l<keys[k].length; l++){
								val[k*n+j][l] = im[keys[k][l]];
							}
						}
						// majority vote case
						else if (type==1){
							val[k*n+j] = new double[keys[k].length];
							int[] mfi = RegionFunctions.maxFreqInd(reg,VectorAccess.flip(RegionFunctions.valueFrequencies(reg,layers,im)));
							for (int l=0; l<keys[k].length; l++){
								val[k*n+j][l] = mfi[l];
							}
						}
						// mean value case
						else if (type==2){
							val[k*n+j] = new double[keys[k].length];
							double[] mv = RegionFunctions.regionMean(reg,im);
							for (int l=0; l<keys[k].length; l++){
								val[k*n+j][l] = mv[l];
							}
						}
						// variance value case
						else if (type==3){
							val[k*n+j] = new double[keys[k].length];
							double[] mv = RegionFunctions.regionVar(reg,im);
							for (int l=0; l<keys[k].length; l++){
								val[k*n+j][l] = mv[l];
							}
						}
						// class histogram case
						else {
							double[][] vf = RegionFunctions.valueFrequencies(reg,layers,im);
							for (int m=0; m<layers; m++){
								val[(k*layers+m)*n+j] = new double[keys[k].length];
								for (int l=0; l<keys[k].length; l++){
									val[(k*layers+m)*n+j][l] = vf[m][l];
								}
							}
						}
					}
					// store the keys in a file
					ReadWrite.writeObject(name,val);
				}
			}
		}
	}
}