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

/* The purpose of this class is to be able to take an open trakem project with 
area lists representing training classes and find training point locations.*/

public class TrainForestOnFeatures {
	private RandomForest RF;
	
	public TrainForestOnFeatures(Project p, boolean[] categorical, int mtry, int maxdepth, int maxleafsize, double splitpurity, int ntree, double weightingpower, int numpoints){
		// get the root layerset object to access the individual layers
		// and displayable objects
		LayerSet layers = p.getRootLayerSet();
		// get the area lists which contain the class labels
		ArrayList<ZDisplayable> alis = layers.getZDisplayables(AreaList.class);
		int nclasses = alis.size();
		ArrayList<Displayable> alis2 = new ArrayList<Displayable>();
		for (int i=0; i<nclasses; i++){
			alis2.add(alis.get(i));
		}
		Patch patches = (Patch)(layers.getLayers().get(0).getDisplayables(Patch.class).get(0));
		ImagePlus imp = patches.getImagePlus();
		int w = imp.getWidth();
		int h = imp.getHeight();
		AreaList.exportAsLabels(alis2,new Roi(0,0,w,h),(float)1.0,0,0,false,false,false);
		ImagePlus labelimage = IJ.getImage();
		int[] labels = VectorConv.float2int((float[])(labelimage.getProcessor().convertToFloat().getPixels()));
		labelimage.close();
		labels = VectorFun.add(labels,-1);
		int[] ind = VectorAccess.subset(BV.gt(VectorConv.int2double(labels),-1));
		int n = ind.length;
		if (numpoints<n){
			int[] r = Shuffle.randPerm(n);
			ind = VectorAccess.access(ind,VectorAccess.access(r,0,numpoints));
			n = ind.length;
		}
		// here we reduce the size of the label vector to only contain labeled pixels
		labels = VectorAccess.access(labels,ind);
		int dim = layers.getLayers().size()-1;
		n = labels.length;
		// preparation of the training set based on the current images
		double[][] trainingset = new double[n][dim];
		double[] pixels;
		for (int i=0; i<dim; i++){
			// a long command in which we get the correct layer image
			// and select the subset of pixels at the index locations
			pixels = VectorAccess.access(VectorConv.float2double((float[])(((Patch)(layers.getLayers().get(i+1).getDisplayables(Patch.class).get(0))).getImageProcessor().convertToFloat().getPixels())),ind);
			for (int j=0; j<n; j++){
				trainingset[j][i] = pixels[j];
			}
		}
		//IJ.log(Arrays.toString(labels));
		// weight all labels so that the number of pixels of class times the pixel
		// weight is constant between classes
		double[] weighting = new double[nclasses];
		for (int i=0; i<n; i++){
			weighting[labels[i]]++;
		}
		for (int i=0; i<nclasses; i++){
			weighting[i] = 1/Math.pow(weighting[i],weightingpower);
		}
		double[] weights = new double[n];
		for (int i=0; i<n; i++){
			weights[i] = weighting[labels[i]];
			//weights[i] = 1;
		}
		// for the moment we'll assume our variables are not categorical
		//boolean[] categorical = new boolean[dim];
		this.RF = new RandomForest(trainingset,nclasses,VectorConv.int2double(labels),weights,categorical,new double[1],mtry,maxdepth,maxleafsize,splitpurity,0,ntree);
	}
	
	public TrainForestOnFeatures(Project p, int mtry, int maxdepth, int maxleafsize, double splitpurity, int ntree, double weightingpower, int numpoints){
		this(p,new boolean[p.getRootLayerSet().getZDisplayables(AreaList.class).size()],mtry,maxdepth,maxleafsize,splitpurity,ntree,weightingpower,numpoints);
	}
	public RandomForest getForest(){
		return this.RF;
	}
}