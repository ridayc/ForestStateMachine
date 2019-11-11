package flib.ij.segmentation;

import java.lang.Class;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import ini.trakem2.Project;
import ini.trakem2.display.LayerSet;
import ini.trakem2.display.Displayable;
import ini.trakem2.display.ZDisplayable;
import ini.trakem2.display.AreaList;
import ini.trakem2.display.Patch;
import ini.trakem2.display.Layer;
import ij.gui.Roi;
import ij.ImagePlus;
import ij.IJ;
import flib.math.VectorConv;
import flib.math.VectorFun;
import flib.math.BV;
import flib.math.VectorAccess;
import flib.math.random.Shuffle;
import flib.algorithms.randomforest.RandomForest;

public class ApplyForestToFeatures {
	private double[][] votes;
	
	public ApplyForestToFeatures(Project p, RandomForest RF){
		// get the root layerset object to access the individual layers
		// and displayable objects
		LayerSet layers = p.getRootLayerSet();
		// preparation of the training set based on the current images
		double[] pixels;
		pixels = VectorConv.float2double((float[])(((Patch)(layers.getLayers().get(1).getDisplayables(Patch.class).get(0))).getImageProcessor().convertToFloat().getPixels()));
		int n = pixels.length;
		int dim = layers.getLayers().size()-1;
		double[][] datapoints = new double[n][dim];
		for (int i=0; i<n; i++){
			datapoints[i][0] = pixels[i];
		}
		for (int i=1; i<dim; i++){
			// a long command in which we get the correct layer image
			// and select the subset of pixels at the index locations
			pixels = VectorConv.float2double((float[])(((Patch)(layers.getLayers().get(i+1).getDisplayables(Patch.class).get(0))).getImageProcessor().convertToFloat().getPixels()));
			for (int j=0; j<n; j++){
				datapoints[j][i] = pixels[j];
			}
		}
		votes = RF.applyForest(datapoints);
	}
	
	public double[][] getVotes(){
		return this.votes;
	}
}