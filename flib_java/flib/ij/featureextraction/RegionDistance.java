package flib.ij.featureextraction;

import java.io.File;
import flib.io.TypeReader;
import flib.ij.io.ImageReader;
import flib.ij.featureextraction.FileInterpreter;
import flib.algorithms.DistanceTransform;
import flib.math.VectorFun;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;

public class RegionDistance {
	public static void run(String orig, String targetdir){
		// create the target directory if necessary
		new File(targetdir).mkdirs();
		// prepare the file interpreter
		FileInterpreter FI = new FileInterpreter(orig);
		// go through all image files
		for (int i=0; i<FI.getBases().length; i++){
			// get the image dimensions
			String name = targetdir+File.separator+FI.getBases()[i]+TypeReader.TIFF[0];
			if (!(new File(name)).exists()){
				int size[] = ImageReader.getTiffSize(FI.getNames()[i]);
				int[][] d = new int[8][2];
				d[0][0] = -1;
				d[0][1] = -1;
				d[1][1] = -1;
				d[2][0] = 1;
				d[2][1] = -1;
				d[3][0] = -1;
				d[4][0] = 1;
				d[5][0] = -1;
				d[5][1] = 1;
				d[6][1] = 1;
				d[7][0] = 1;
				d[7][1] = 1;
				// prepare an image stack
				ImageStack stack = new ImageStack(size[1],size[2]);
				double min = Double.MAX_VALUE;
				double max = -Double.MAX_VALUE;
				for (int j=0; j<size[0]; j++){
					float[] im = ImageReader.tiffLayerArray(FI.getNames()[i],j);
					double[] binim = new double[im.length];
					for (int k=0; k<im.length; k++){
						double m = 1;
						int x = k%size[1];
						int y = k/size[1];
						// check for all region boundary pixels
						for (int l=0; l<d.length; l++){
							int a = x+d[l][0];
							int b = y+d[l][1];
							if (a>0&&a<size[1]&&b>0&&b<size[2]){
								if (im[k]!=im[a+b*size[1]]){
									m = 0;
								}
							}
						}
						binim[k] = m;
					}
					// calculate the distance transform of the resultant binary image
					binim = DistanceTransform.dt2d(size[1],size[2],binim,0);
					stack.addSlice(new FloatProcessor(size[1],size[2],binim));
					double t = VectorFun.max(binim)[0];
					if (t>max){
						max = t;
					}
					 t = VectorFun.min(binim)[0];
					if (t<min){
						min = t;
					}
				}
				ImagePlus imp = new ImagePlus("DistanceTransform",stack);
				imp.getProcessor().setMinAndMax(min,max);
				IJ.saveAsTiff(imp,name);
			}
		}
	}
}
		