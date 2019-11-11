package flib.ij.gui;

import java.lang.Math;
import java.io.File;
import java.lang.Class;
import java.util.ArrayList;
import java.util.Arrays;
import java.awt.AWTEvent;
import java.util.Date;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.io.DirectoryChooser;
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ini.trakem2.Project;
import ini.trakem2.display.LayerSet;
import ini.trakem2.display.Displayable;
import ini.trakem2.display.Patch;
import ini.trakem2.display.Layer;
import flib.ij.featureextraction.ScaleFeatureStack;
import flib.ij.featureextraction.ExtractTrainingLabels;
import flib.ij.stack.StackOperations;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.VectorAccess;
import flib.math.RankSort;
import flib.math.random.Shuffle;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.clustering.RFC;
import flib.algorithms.randomforest.ReadForest;
import flib.algorithms.sampling.NeighborhoodSample;

/* The TrainForestGui class intends to provide a fairly simple, yet vertasile interface for users to train a random forest
of a predefined set of multiple labels using a log-Gabor filter bank and maximum angles and phases resulting therefrom.
Additionally a recursive random forest algorithm can be trained which uses local information to propagate labels through the 
image space. */
public class TrainRFCGui
	implements DialogListener{
	// parameters
	// lambda1, lambda2: lowest and highest approximate spatial frequency for the log-Gabor (LG) filter bank
	// sigma: LG bandwidth
	// anglewidth: describes the the angular smoothing width of the LG
	// numstep: number of spatial frequencies to be used. These are spaced exponentially
	// numang: number of (rotated) angles to consider in the LG
	// numphase: number of different phases to consider in the LG
	// forest parameters:
	// ppc: maximum number of training points per class of the labeled data
	// nc: number of RFC clusters
	// maxdepth: maximal depth a decision tree can have
	// splitpurity: fraction of points which need to belong to a class before forming a leaf node
	// maxleafsize: the maximum number of points required in an impure leaf node
	// ntree: number of decision trees in the forest
	// mtry: number of random dimensions to choose at each decision node
	
	// LogGabor+LogGaborBand parameters
	private double[] lambda1 = new double[]{2,2};
	private double[] lambda2 = new double[]{80,120};
	private double[] sigma = new double[]{2,2};
	private double[] ang = new double[]{1,-5};
	private int[] numstep = new int[]{5,5};
	private int[] numang = new int[]{16,1};
	private int[] numphase = new int[]{2,1};
	
	// Forest / RFC parameters
	private int[] ppc = new int[]{200,200,200,200,200};
	private int nc = 10;
	private double balance = 1;
	private int maxit = 100;
	private int[] ntree = new int[]{200,200,200,200};
	private int[] mtry = new int[]{0,0,0,0};
	private int[] maxdepth = new int[]{100,100,100,100};
	private int[] maxleafsize = new int[]{5,3,3,2};
	private double[] splitpurity = new double[]{1,1,1,1};
	private int n[] = new int[3];
	
	// Patch parameters
	private double[] phi = new double[]{1,1};
	private double[] patchradius = new double[]{5,5};
	private double[] br = new double[]{1,1};
	private double[] nphi = new double[]{1,1};
	private double[] nrad = new double[]{1,1};
	private int[] arms = new int[]{4,4};
	private int iterations = 5, iterations2 = 5;
	
	// other
	// checkboxes
	private boolean[] defVal = new boolean[10];
	private double[] factor = new double[4];
	private ExtractTrainingLabels etl;
	private String targetdir, featuredir;
	private String[] forestdir = new String[3];
	private String[] votesdir = new String[3];
	private String[] ident = new String[3];
	private String[] feat = new String[9];
	private int type, numclasses;
	
	
	public TrainRFCGui(){
		this("");
	}
	
	public TrainRFCGui(String projectname){
		// choose a storage folder location
		targetdir = (new DirectoryChooser("Choose a folder")).getDirectory();
		if (targetdir.isEmpty()){
			IJ.log("User canceled the dialog! Prepare for complications...");
			return;
		}
		Project p;
		if (projectname.equals("")){
			p = Project.getProjects().get(0);
		}
		else {
			p = Project.openFSProject(projectname,false);
		}
		LayerSet layers = p.getRootLayerSet();
		etl = new ExtractTrainingLabels(p);
		int l = etl.getL();
		int[][] imdim = etl.getImageDim();
		numclasses = etl.getLabels().length;
		// show the parameters dialogue
		dialogueSetup();
		// prepare the name content of the file
		ident[0] = "lf"+String.format("%.2f",lambda1[0])+"_ll"
		+String.format("%.2f",lambda2[0])+"_sg"+String.format("%.2f",sigma[0])+"_aw"
		+String.format("%.2f",ang[0])+"_ns"+String.format("%d",numstep[0])+"_na"
		+String.format("%d",numang[0])+"_np"+String.format("%d",numphase[0]);
		ident[0] = ident[0].replace(".","d");
		ident[1] = "lf"+String.format("%.2f",lambda1[1])+"_ll"
		+String.format("%.2f",lambda2[1])+"_sg"+String.format("%.2f",sigma[1])+"_aw"
		+String.format("%.2f",ang[1])+"_ns"+String.format("%d",numstep[1])+"_na"
		+String.format("%d",numang[1])+"_np"+String.format("%d",numphase[1]);
		ident[1] = ident[1].replace(".","d");
		featuredir = targetdir+File.separator+"features";
		forestdir[0] = targetdir+File.separator+"forest";
		votesdir[0] = targetdir+File.separator+"votes";
		forestdir[1] = targetdir+File.separator+"iterative_forest";
		votesdir[1] = targetdir+File.separator+"iterative_votes";
		forestdir[2] = targetdir+File.separator+"automaton_forest";
		votesdir[2] = targetdir+File.separator+"automaton_votes";
		feat[0] = "LGfeat_";
		feat[1] = "LGMAfeat_";
		feat[2] = "LGMPfeat_";
		feat[3] = "DTfeat_";
		feat[4] = "LGBfeat_";
		feat[5] = "LGBMAfeat_";
		feat[6] = "LGBMPfeat_";
		feat[7] = "Compfeat_";
		feat[8] = "Clindfeat_";
		
		// prepare the features folder
		new File(featuredir).mkdirs();
		// go through the different check box cases
		if (defVal[0]||defVal[1]||defVal[2]||defVal[3]||defVal[4]||defVal[5]||defVal[6]){
			new File(forestdir[0]).mkdirs();
			new File(votesdir[0]).mkdirs();
		}
		IJ.log("Program started running: "+String.format((new Date()).toString()));
		// prepare whatever feature vectors need preparing
		for (int i=0; i<l; i++){
			ImagePlus imp = ((Patch)(layers.getLayers().get(i).getDisplayables(Patch.class).get(0))).getImagePlus();
			// check if we need to create files for the log Gabor forest
			if (defVal[0]||defVal[1]||defVal[2]){
				checkScales(0,0,1,2,0,i,imp);
			}
			// the same for the Log Gabor Band filters
			if (defVal[4]||defVal[5]||defVal[6]){
				checkScales(4,4,5,6,1,i,imp);
			}
			if (defVal[3]){
				String name = featuredir+File.separator+feat[3]+String.format("%d",i)+"_"+ident[0]+".tif";
				if (!(new File(name)).exists()){
					IJ.saveAsTiff(StackOperations.distanceTransform(StackOperations.watershedBoundaries2Sided(ScaleFeatureStack.scaleFeatures(imp,lambda1[0],factor[0],numstep[0],1,-5,1,sigma[0]),8),0),name);
				}
			}
		}
		IJ.log("All features generated: "+String.format((new Date()).toString()));
		if (defVal[0]||defVal[1]||defVal[2]||defVal[3]||defVal[4]||defVal[5]||defVal[6]){
			// prepare the random forest classifier
			IJ.log("RFC Forest: "+String.format((new Date()).toString()));
			prepareForest();
			IJ.log("RFC training: "+String.format((new Date()).toString()));
			prepareRFC();
			applyRFC();
		}
		if (defVal[7]||defVal[8]){
			IJ.log("Initial forest training: "+String.format((new Date()).toString()));
			prepareForest2();
			IJ.log("Apply the initial forest: "+String.format((new Date()).toString()));
			// Apply the trained forest to all images
			applyForest();
			// Here
		}
		if (defVal[9]){
			// make sure the second forest folder are present
			new File(forestdir[1]).mkdirs();
			new File(votesdir[1]).mkdirs();
			new File(forestdir[2]).mkdirs();
			new File(votesdir[2]).mkdirs();
			// copy forest and votes from the first classifier
			for (int i=0; i<l; i++){
				String votesname = votesdir[0]+File.separator+"votes_"+String.format("%d",i)+"_"+ident[0]+".tif";
				String forestname = forestdir[0]+File.separator+"forest_"+String.format("%d",i)+"_"+ident[0]+".tif";
				String targetvotes = votesdir[1]+File.separator+"votes_"+String.format("%d",i)+"_"+ident[0]+".tif";
				String targetforest = forestdir[1]+File.separator+"forest_"+String.format("%d",i)+"_"+ident[0]+".tif";
				ReadForest.copyFile(votesname,targetvotes);
				ReadForest.copyFile(forestname,targetforest);
				targetvotes = votesdir[2]+File.separator+"votes_"+String.format("%d",i)+"_"+ident[0]+".tif";
				targetforest = forestdir[2]+File.separator+"forest_"+String.format("%d",i)+"_"+ident[0]+".tif";
				ReadForest.copyFile(votesname,targetvotes);
				ReadForest.copyFile(forestname,targetforest);
			}
			
			for (int i=0; i<iterations; i++){
				IJ.log("Training forest iteration "+String.format("%d",i)+": "+String.format((new Date()).toString()));
				// prepare the random forest classifier
				iterativeForest(i);
				IJ.log("Applying forest iteration "+String.format("%d",i)+": "+String.format((new Date()).toString()));
				// Apply the trained forest to all images
				
				applyIterativeForest(i);
			}
			IJ.log("Automaton forest training: "+String.format((new Date()).toString()));
			automatonForest();
			for (int i=0; i<iterations2; i++){
				IJ.log("Applying automaton forest iteration "+String.format("%d",i)+": "+String.format((new Date()).toString()));
				applyAutomatonForest(i);
			}
		}
		IJ.log("Everything finished by: "+String.format((new Date()).toString()));
	}
	
	private void checkScales(int b, int v1, int v2, int v3, int i, int j, ImagePlus imp){
		String[] name = new String[3];
		name[0] = featuredir+File.separator+feat[v1]+String.format("%d",j)+"_"+ident[i]+".tif";
		name[1] = featuredir+File.separator+feat[v2]+String.format("%d",j)+"_"+ident[i]+".tif";
		name[2] = featuredir+File.separator+feat[v3]+String.format("%d",j)+"_"+ident[i]+".tif";
		ImagePlus featimp;
		if (defVal[b]&&!(new File(name[0])).exists()){
			if (b%8<3){
				IJ.saveAsTiff(ScaleFeatureStack.scaleFeatures(imp,lambda1[i],factor[i],numstep[i],numang[i],ang[i],numphase[i],sigma[i]),name[0]);
			}
			else{
				IJ.saveAsTiff(ScaleFeatureStack.bandScaleFeatures(imp,lambda1[i],lambda1[i],factor[i],numstep[i],numang[i],ang[i],numphase[i],sigma[i]),name[0]);
			}
		}
		if (defVal[b+1]&&!(new File(name[1])).exists()){
			if(!(new File(name[0])).exists()){
				if (b%8<3){
					featimp = ScaleFeatureStack.scaleFeatures(imp,lambda1[i],factor[i],numstep[i],numang[i],ang[i],numphase[i],sigma[i]);
				}
				else {
					featimp = ScaleFeatureStack.bandScaleFeatures(imp,lambda1[i],lambda1[i],factor[i],numstep[i],numang[i],ang[i],numphase[i],sigma[i]);
				}
			}
			else {
				featimp = IJ.openImage(name[0]);
			}
			IJ.saveAsTiff(ScaleFeatureStack.maxDim(featimp,numstep[i],numang[i],numphase[i],2),name[1]);
		}
		if (defVal[b+2]&&!(new File(name[2])).exists()){
			if((new File(name[1])).exists()){
				featimp = ScaleFeatureStack.maxDim(IJ.openImage(name[1]),numstep[i],1,numphase[i],1);
			}
			else if ((new File(name[0])).exists()){
				featimp = ScaleFeatureStack.maxDim(IJ.openImage(name[i]),numstep[i],numang[i],numphase[i],3);
			}
			else {
				if (b%8<3){
					featimp = ScaleFeatureStack.maxDim(ScaleFeatureStack.scaleFeatures(imp,lambda1[i],factor[i],numstep[i],numang[i],ang[i],numphase[i],sigma[i]),numstep[i],numang[i],numphase[i],3);
				}
				else {
					featimp = ScaleFeatureStack.maxDim(ScaleFeatureStack.bandScaleFeatures(imp,lambda1[i],lambda1[i],factor[i],numstep[i],numang[i],ang[i],numphase[i],sigma[i]),numstep[i],numang[i],numphase[i],3);
				}
			}
			IJ.saveAsTiff(featimp,name[2]);
		}
	}
	
	private void dialogueSetup(){
		type = 1;
		defVal[8] = true;
		defVal[9] = true;
		GenericDialog gd;
		String[] labels = new String[10];
		labels[0] = "Log-Gabor Filter Bank";
		labels[1] = "Maximum Absolute Angle";
		labels[2] = "Maximum Absolute Angle+Phase";
		labels[3] = "Watershed Distance Transform";
		labels[4] = "Log-Gabor Band Filter Bank";
		labels[5] = "Maximum Absolute Angle";
		labels[6] = "Maximum Absolute Angle+Phase";
		labels[7] = "RFC Cluster Distance";
		labels[8] = "RFC Cluster Index";
		labels[9] = "Smoothing Patches";
		gd = new GenericDialog("Training Parameters");
		gd.addMessage("Feature Types: \n");
		for (int i=0; i<10; i++){
			gd.addCheckbox(labels[i],defVal[i]);
		}		
		gd.addMessage(" \n LogGabor Parameters: \n");
		gd.addNumericField("lambda1: ", lambda1[0], 2);
		gd.addNumericField("lambda2: ", lambda2[0], 2);
		gd.addNumericField("Bandwidth: ", sigma[0], 2);
		gd.addNumericField("Number of steps: ", numstep[0], 0);
		gd.addNumericField("Number of angles: ", numang[0], 0);
		gd.addNumericField("Angular width: ", ang[0], 2);
		gd.addNumericField("Number of phases: ", numphase[0], 0);
		gd.addMessage(" \n LogGabor Band Parameters: \n");
		gd.addNumericField("lambda1: ", lambda1[1], 2);
		gd.addNumericField("lambda2: ", lambda2[1], 2);
		gd.addNumericField("Bandwidth: ", sigma[1], 2);
		gd.addNumericField("Number of steps: ", numstep[1], 0);
		gd.addNumericField("Number of angles: ", numang[1], 0);
		gd.addNumericField("Angular width: ", ang[1], 2);
		gd.addNumericField("Number of phases: ", numphase[1], 0);
		gd.addDialogListener(this);
		gd.showDialog();
		if (gd.wasCanceled()) return;
		lambda1[0] = gd.getNextNumber();
		lambda2[0] = gd.getNextNumber();
		sigma[0] = gd.getNextNumber();
		numstep[0] = (int)gd.getNextNumber();
		numang[0] = (int)gd.getNextNumber();
		ang[0] = gd.getNextNumber();
		numphase[0] = (int)gd.getNextNumber();
		lambda1[1] = gd.getNextNumber();
		lambda2[1] = gd.getNextNumber();
		sigma[1] = gd.getNextNumber();
		numstep[1] = (int)gd.getNextNumber();
		numang[1] = (int)gd.getNextNumber();
		ang[1] = gd.getNextNumber();
		numphase[1] = (int)gd.getNextNumber();
		factor[0] = Math.pow(lambda2[0]/lambda1[0],1.0/(numstep[0]-1));
		factor[1] = Math.pow(lambda2[1]/lambda1[1],1.0/(numstep[1]-1));
		for (int i=0; i<10; i++){
			defVal[i] = gd.getNextBoolean();
		}
		// inactivate the mtry update
		type = 0;
		// second dialogue window for the forest parameters
		gd = new GenericDialog("Information Forest Parameters");
		gd.addNumericField("Number of Training Points per Class: ", ppc[0], 0);
		gd.addNumericField("Number of Trees: ", ntree[0], 0);
		gd.addNumericField("Number of Random Dimensions ("+String.format("%d",n[0])+"): ", mtry[0], 0);
		gd.addNumericField("Maximum Tree Depth: ", maxdepth[0], 0);
		gd.addNumericField("Maximmal Number of Points in a Leaf Node: ", maxleafsize[0], 0);
		gd.addNumericField("Label Purity required for a Leaf Node: ", splitpurity[0], 2);
		gd.addMessage(" \n RFC Parameters: \n");
		gd.addNumericField("Number of RFC clusters: ", nc, 0);
		gd.addNumericField("Number of points from the initial classes: ", ppc[1], 0);
		gd.showDialog();
		if (gd.wasCanceled()) return;
		ppc[0] = (int)gd.getNextNumber();
		ntree[0] = (int)gd.getNextNumber();
		mtry[0] = (int)gd.getNextNumber();
		if (mtry[0]<1){
			mtry[0] = (int)Math.sqrt(n[0]);
		}
		maxdepth[0] = (int)gd.getNextNumber();
		maxleafsize[0] = (int)gd.getNextNumber();
		splitpurity[0] = gd.getNextNumber();
		nc = (int)gd.getNextNumber();
		ppc[1] = (int)gd.getNextNumber();
		type = 2;
		gd = new GenericDialog("Information Forest Parameters");
		gd.addNumericField("Number of Training Points per Class: ", ppc[2], 0);
		gd.addNumericField("Number of Trees: ", ntree[1], 0);
		gd.addNumericField("Number of Random Dimensions: ", mtry[1], 0);
		gd.addNumericField("Maximum Tree Depth: ", maxdepth[1], 0);
		gd.addNumericField("Maximmal Number of Points in a Leaf Node: ", maxleafsize[1], 0);
		gd.addNumericField("Label Purity required for a Leaf Node: ", splitpurity[1], 2);
		gd.addMessage("\n Smoothing Spiral: \n");
		gd.addNumericField("Angular increment: ", phi[0], 2);
		gd.addNumericField("Patch Radius: ", patchradius[0], 0);
		gd.addNumericField("Radial increment: ", br[0], 2);
		gd.addNumericField("Angular exponent: ", nphi[0], 2);
		gd.addNumericField("Radial exponent: ", nrad[0], 2);
		gd.addNumericField("Spiral Arms: ", arms[0], 0);
		gd.addDialogListener(this);
		gd.showDialog();
		if (gd.wasCanceled()) return;
		ppc[2] = (int)gd.getNextNumber();
		ntree[1] = (int)gd.getNextNumber();
		mtry[1] = (int)gd.getNextNumber();
		if (mtry[1]<1){
			mtry[1] = (int)Math.sqrt(n[1]);
		}
		maxdepth[1] = (int)gd.getNextNumber();
		maxleafsize[1] = (int)gd.getNextNumber();
		splitpurity[1] = gd.getNextNumber();
		phi[0] = gd.getNextNumber();
		patchradius[0] = gd.getNextNumber();
		br[0] = gd.getNextNumber();
		nphi[0] = gd.getNextNumber();
		nrad[0] = gd.getNextNumber();
		arms[0] = (int)gd.getNextNumber();
		
		if (defVal[9]){
			// activate the mtry update
			type = 3;
			// dialogue for the automaton parameters
			gd = new GenericDialog("Automaton Forest Parameters");
			gd.addNumericField("Number of Training Iterations: ", iterations, 0);
			gd.addNumericField("Number of Automaton Iterations: ", iterations2, 0);
			gd.addMessage("\n Smoothing Spiral: \n");
			gd.addNumericField("Angular increment: ", phi[1], 2);
			gd.addNumericField("Patch Radius: ", patchradius[1], 0);
			gd.addNumericField("Radial increment: ", br[1], 2);
			gd.addNumericField("Angular exponent: ", nphi[1], 2);
			gd.addNumericField("Radial exponent: ", nrad[1], 2);
			gd.addNumericField("Spiral Arms: ", arms[1], 0);
			gd.addMessage(" \n Random Forest Parameters (Iterations): \n");
			gd.addNumericField("Number of Training Points per Class: ", ppc[3], 0);
			gd.addNumericField("Number of Trees: ", ntree[2], 0);
			gd.addNumericField("Number of Random Dimensions: ", mtry[2], 0);
			gd.addNumericField("Maximum Tree Depth: ", maxdepth[2], 0);
			gd.addNumericField("Maximmal Number of Points in a Leaf Node: ", maxleafsize[2], 0);
			gd.addNumericField("Label Purity required for a Leaf Node: ", splitpurity[2], 2);
			gd.addMessage(" \n Random Forest Parameters (Automaton): \n");
			gd.addNumericField("Number of Training Points per Class: ", ppc[4], 0);
			gd.addNumericField("Number of Trees: ", ntree[3], 0);
			gd.addNumericField("Number of Random Dimensions: ", mtry[3], 0);
			gd.addNumericField("Maximum Tree Depth: ", maxdepth[3], 0);
			gd.addNumericField("Maximmal Number of Points in a Leaf Node: ", maxleafsize[3], 0);
			gd.addNumericField("Label Purity required for a Leaf Node: ", splitpurity[3], 2);
			gd.addDialogListener(this);
			gd.showDialog();
			if (gd.wasCanceled()) return;
			iterations = (int)gd.getNextNumber();
			iterations2 = (int)gd.getNextNumber();
			phi[1] = gd.getNextNumber();
			patchradius[1] = gd.getNextNumber();
			br[1] = gd.getNextNumber();
			nphi[1] = gd.getNextNumber();
			nrad[1] = gd.getNextNumber();
			arms[1] = (int)gd.getNextNumber();
			ppc[3] = (int)gd.getNextNumber();
			ntree[2] = (int)gd.getNextNumber();
			mtry[2] = (int)gd.getNextNumber();
			if (mtry[2]<1){
				mtry[2] = (int)Math.sqrt(n[1]+n[2]);
			}
			maxdepth[2] = (int)gd.getNextNumber();
			maxleafsize[2] = (int)gd.getNextNumber();
			splitpurity[2] = gd.getNextNumber();
			ppc[4] = (int)gd.getNextNumber();
			ntree[3] = (int)gd.getNextNumber();
			mtry[3] = (int)gd.getNextNumber();
			if (mtry[3]<1){
				mtry[3] = (int)Math.sqrt(n[1]+n[2]);
			}
			maxdepth[3] = (int)gd.getNextNumber();
			maxleafsize[3] = (int)gd.getNextNumber();
			splitpurity[3] = gd.getNextNumber();
		}
	}
	
	public boolean dialogItemChanged(GenericDialog gd, AWTEvent e){
		if (type==1){
			lambda1[0] = gd.getNextNumber();
			lambda2[0] = gd.getNextNumber();
			sigma[0] = gd.getNextNumber();
			numstep[0] = (int)gd.getNextNumber();
			numang[0] = (int)gd.getNextNumber();
			ang[0] = gd.getNextNumber();
			numphase[0] = (int)gd.getNextNumber();
			lambda1[1] = gd.getNextNumber();
			lambda2[1] = gd.getNextNumber();
			sigma[1] = gd.getNextNumber();
			numstep[1] = (int)gd.getNextNumber();
			numang[1] = (int)gd.getNextNumber();
			ang[1] = gd.getNextNumber();
			numphase[1] = (int)gd.getNextNumber();
			for (int i=0; i<7; i++){
				defVal[i] = gd.getNextBoolean();
			}
			n[0] = 0;
			if (defVal[0]){
				n[0]+=numstep[0]*numang[0]*numphase[0];
			}
			if (defVal[1]){
				n[0]+=numstep[0]*numphase[0];
			}
			if (defVal[2]){
				n[0]+=numstep[0];
			}
			if (defVal[3]){
				n[0]+=numstep[0]*2;
			}
			if (defVal[4]){
				n[0]+=numstep[1]*numang[1]*numphase[1];
			}
			if (defVal[5]){
				n[0]+=numstep[1]*numphase[1];
			}
			if (defVal[6]){
				n[0]+=numstep[1];
			}
			IJ.log("Maximum number of training dimensions: "+String.format("%d",n[0]));
		}
		else if (type==2){
			ppc[2] = (int)gd.getNextNumber();
			ntree[1] = (int)gd.getNextNumber();
			mtry[1] = (int)gd.getNextNumber();
			if (mtry[1]<1){
				mtry[1] = (int)Math.sqrt(n[1]);
			}
			maxdepth[1] = (int)gd.getNextNumber();
			maxleafsize[1] = (int)gd.getNextNumber();
			splitpurity[1] = gd.getNextNumber();
			phi[0] = gd.getNextNumber();
			patchradius[0] = gd.getNextNumber();
			br[0] = gd.getNextNumber();
			nphi[0] = gd.getNextNumber();
			nrad[0] = gd.getNextNumber();
			arms[0] = (int)gd.getNextNumber();
			n[1] = 0;
			int t = NeighborhoodSample.spiralCoord(phi[0],patchradius[0],br[0],nphi[0],nrad[0],arms[0]).length;
			if (defVal[7]){
				n[1]+=nc*t;
			}
			if (defVal[8]){
				n[1]+=t;
			}
			IJ.log("Maximum number initial training dimensions: "+String.format("%d",n[1]));
		}
		else if (type==3){
			iterations = (int)gd.getNextNumber();
			iterations2 = (int)gd.getNextNumber();
			phi[1] = gd.getNextNumber();
			patchradius[1] = gd.getNextNumber();
			br[1] = gd.getNextNumber();
			nphi[1] = gd.getNextNumber();
			nrad[1] = gd.getNextNumber();
			arms[1] = (int)gd.getNextNumber();
			n[2] = NeighborhoodSample.spiralCoord(phi[1],patchradius[1],br[1],nphi[1],nrad[1],arms[1]).length;
			IJ.log("Maximum number iterative of training dimensions: "+String.format("%d",n[1]+n[2]));
		}
		return true;
	}
	
	private void prepareForest(){
		int[][] ind = etl.getLabels();
		int numclasses = ind.length;
		int[] imlen = etl.getImageLen();
		int[] imlen2 = VectorFun.cumsum(imlen);
		int[] imlen3 = VectorFun.sub(imlen2,imlen);
		int[][] imdim = etl.getImageDim();
		// found out if the minimum number of pixels in a class is less than 
		// the proposed number of pixels per class for training
		ArrayList<Integer> labelvec = new ArrayList<Integer>();
		ArrayList<Integer> trainingindvec = new ArrayList<Integer>();
		for (int i=0; i<numclasses; i++){
			int b;
			if (ppc[0]>ind[i].length){
				b = ind[i].length;
			}
			else {
				b = ppc[0];
			}
			int[] tempind = VectorAccess.access(ind[i],VectorAccess.access(Shuffle.randPerm(ind[i].length),0,b));
			for (int j=0; j<b; j++){
				labelvec.add(i);
				trainingindvec.add(tempind[j]);
			}
		}
		// preparation for the random forest
		int len = labelvec.size();
		double[] labels = new double[len];
		int[] trainingind = new int[len];
		double[] weights = VectorFun.add(labels,1);
		boolean[] categorical = new boolean[n[0]];
		double[][] trainingset = new double[len][n[0]];
		for (int i=0; i<len; i++){
			labels[i] = labelvec.get(i);
			trainingind[i] = trainingindvec.get(i);
		}
		// preparation of the dim weights
		// we want the number of samples drawn per class to be roughly in the ratio as if we
		// would have drawn sqrt samples from the individual classes
		int counter =0;
		double alpha = Math.log(mtry[0])/Math.log(n[0]);
		double[] dimweights = VectorFun.add(new double[n[0]],1);
		if (defVal[0]){
			int t = numstep[0]*numang[0]*numphase[0];
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		if (defVal[1]){
			int t = numstep[0]*numphase[0];
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		if (defVal[2]){
			int t = numstep[0];
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		if (defVal[3]){
			int t = numstep[0];
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		if (defVal[4]){
			int t = numstep[1]*numang[1]*numphase[1];
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		if (defVal[5]){
			int t = numstep[1]*numphase[1];
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		if (defVal[6]){
			int t = numstep[1];
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		// reorder the label indices and the according labels
		// now comes the more challenging part of putting the trainingset
		// together. We need to make sure this is managed relatively quickly.
		RankSort r = new RankSort(VectorConv.int2double(trainingind),labels);
		trainingind = VectorConv.double2int(r.getSorted());
		labels = r.getDRank();
		// prepare a list of indices for each image
		int count = 0;
		int countind = 0;
		int len2 = 0;
		double[][] temp;
		int[] point = new int[1];
		for (int i=0; i<len; i++){
			// if we would have to go to the next image
			if (trainingind[i]>=imlen2[countind]){
				// if there were pixels in the last image
				if (len2>0){
					counter = 0;
					String featurename;
					for (int k=0; k<7; k++){
						if (defVal[k]){
							featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[k/4]+".tif";
							temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(featurename)));
							for (int j=0; j<len2; j++){
								VectorAccess.write(trainingset[i-len2+j],temp[trainingind[i-len2+j]-imlen3[countind]],counter);
							}
							counter+=temp[0].length;
						}
					}
				}
				while (trainingind[i]>=imlen2[countind]){
					countind++;
				}
				len2 = 0;
			}
			len2++;
		}
		counter = 0;
		String featurename;
		for (int k=0; k<7; k++){
			if (defVal[k]){
				featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[k/4]+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(featurename)));
				for (int j=0; j<len2; j++){
					VectorAccess.write(trainingset[len-len2+j],temp[trainingind[len-len2+j]-imlen3[countind]],counter);
				}
				counter+=temp[0].length;
			}
		}

		// training of the actual forest
		RandomForest forest = new RandomForest(trainingset,numclasses,labels,weights,categorical,dimweights,mtry[0],maxdepth[0],maxleafsize[0],splitpurity[0],0,ntree[0]);
		String forestser = forestdir[0]+File.separator+"forest_"+ident[0]+".ser";
		ReadForest.writeForest(forestser,forest);
	}
	
	private void prepareRFC(){
		int[][] ind = etl.getLabels();
		int numclasses = ind.length;
		int[] imlen = etl.getImageLen();
		int[] imlen2 = VectorFun.cumsum(imlen);
		int[] imlen3 = VectorFun.sub(imlen2,imlen);
		int[][] imdim = etl.getImageDim();
		// found out if the minimum number of pixels in a class is less than 
		// the proposed number of pixels per class for training
		ArrayList<Integer> trainingindvec = new ArrayList<Integer>();
		for (int i=0; i<numclasses; i++){
			int b;
			if (ppc[1]>ind[i].length){
				b = ind[i].length;
			}
			else {
				b = ppc[1];
			}
			int[] tempind = VectorAccess.access(ind[i],VectorAccess.access(Shuffle.randPerm(ind[i].length),0,b));
			for (int j=0; j<b; j++){
				trainingindvec.add(tempind[j]);
			}
		}
		// preparation for the random forest
		int len = trainingindvec.size();
		int[] trainingind = new int[len];
		double[][] trainingset = new double[len][n[0]];
		for (int i=0; i<len; i++){
			trainingind[i] = trainingindvec.get(i);
		}
		// reorder the label indices and the according labels
		// now comes the more challenging part of putting the trainingset
		// together. We need to make sure this is managed relatively quickly.
		RankSort r = new RankSort(VectorConv.int2double(trainingind));
		trainingind = VectorConv.double2int(r.getSorted());
		// prepare a list of indices for each image
		int count = 0;
		int counter = 0;
		int countind = 0;
		int len2 = 0;
		double[][] temp;
		int[] point = new int[1];
		for (int i=0; i<len; i++){
			// if we would have to go to the next image
			if (trainingind[i]>=imlen2[countind]){
				// if there were pixels in the last image
				if (len2>0){
					counter = 0;
					String featurename;
					for (int k=0; k<7; k++){
						if (defVal[k]){
							featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[k/4]+".tif";
							temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(featurename)));
							for (int j=0; j<len2; j++){
								VectorAccess.write(trainingset[i-len2+j],temp[trainingind[i-len2+j]-imlen3[countind]],counter);
							}
							counter+=temp[0].length;
						}
					}
				}
				while (trainingind[i]>=imlen2[countind]){
					countind++;
				}
				len2 = 0;
			}
			len2++;
		}
		counter = 0;
		String featurename;
		for (int k=0; k<7; k++){
			if (defVal[k]){
				featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[k/4]+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(featurename)));
				for (int j=0; j<len2; j++){
					VectorAccess.write(trainingset[len-len2+j],temp[trainingind[len-len2+j]-imlen3[countind]],counter);
				}
				counter+=temp[0].length;
			}
		}

		// training of the actual forest
		String forestser = forestdir[0]+File.separator+"forest_"+ident[0]+".ser";
		String rfcser = forestdir[0]+File.separator+"rfc_"+ident[0]+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		RFC rfc = new RFC(forest.getLeafIndices(trainingset),nc,forest.getTreeSizes(),balance,maxit,new int[0]);
		ReadForest.writeRFC(rfcser,rfc);
	}
	
	public void applyRFC(){
		int l = etl.getL();
		int[] len = etl.getImageLen();
		int[][] imdim = etl.getImageDim();
		String rfcser = forestdir[0]+File.separator+"rfc_"+ident[0]+".ser";
		RFC rfc = ReadForest.readRFC(rfcser);
		String forestser = forestdir[0]+File.separator+"forest_"+ident[0]+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		for (int i=0; i<l; i++){
			String rfcname1 = featuredir+File.separator+feat[7]+String.format("%d",i)+"_"+ident[0]+".tif";
			String rfcname2 = featuredir+File.separator+feat[8]+String.format("%d",i)+"_"+ident[0]+".tif";
			double[][] temp2 = new double[len[i]][n[0]];
			double[][] temp;
			int counter = 0;
			for (int k=0; k<7; k++){
				if (defVal[k]){
					String featurename = featuredir+File.separator+feat[k]+String.format("%d",i)+"_"+ident[k/4]+".tif";
					temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(featurename)));
					for (int j=0; j<len[i]; j++){
						VectorAccess.write(temp2[j],temp[j],counter);
					}
					counter+=temp[0].length;
				}
			}
			double[][] votes = VectorAccess.flip(rfc.getDist(forest.getLeafIndices(temp2)));
			if (defVal[7]){
				IJ.saveAsTiff(StackOperations.convert2Stack(votes,imdim[i][0],imdim[i][1]),rfcname1);
			}
			if (defVal[8]){
				IJ.saveAsTiff(StackOperations.convert2Stack(StackOperations.maxIndex(votes),imdim[i][0],imdim[i][1],1),rfcname2);
			}
		}
	}
	
	private void prepareForest2(){
		int[][] ind = etl.getLabels();
		int numclasses = ind.length;
		int[] imlen = etl.getImageLen();
		int[] imlen2 = VectorFun.cumsum(imlen);
		int[] imlen3 = VectorFun.sub(imlen2,imlen);
		int[][] imdim = etl.getImageDim();
		int[][] shape = NeighborhoodSample.spiralCoord(phi[0],patchradius[0],br[0],nphi[0],nrad[0],arms[0]);
		int sl = shape.length;
		// found out if the minimum number of pixels in a class is less than 
		// the proposed number of pixels per class for training
		ArrayList<Integer> labelvec = new ArrayList<Integer>();
		ArrayList<Integer> trainingindvec = new ArrayList<Integer>();
		for (int i=0; i<numclasses; i++){
			int b;
			if (ppc[2]>ind[i].length){
				b = ind[i].length;
			}
			else {
				b = ppc[2];
			}
			int[] tempind = VectorAccess.access(ind[i],VectorAccess.access(Shuffle.randPerm(ind[i].length),0,b));
			for (int j=0; j<b; j++){
				labelvec.add(i);
				trainingindvec.add(tempind[j]);
			}
		}
		// preparation for the random forest
		int len = labelvec.size();
		double[] labels = new double[len];
		int[] trainingind = new int[len];
		double[] weights = VectorFun.add(labels,1);
		boolean[] categorical = new boolean[n[1]];
		double[][] trainingset = new double[len][n[1]];
		for (int i=0; i<len; i++){
			labels[i] = labelvec.get(i);
			trainingind[i] = trainingindvec.get(i);
		}
		// preparation of the dim weights
		// we want the number of samples drawn per class to be roughly in the ratio as if we
		// would have drawn sqrt samples from the individual classes
		int counter =0;
		double alpha = Math.log(mtry[1])/Math.log(n[1]);
		double[] dimweights = VectorFun.add(new double[n[1]],1);
		if (defVal[7]){
			int t = sl*nc;
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		if (defVal[8]){
			int t = sl;
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			for (int i=0; i<sl; i++){
				categorical[counter+i] = true;
			}
			counter+=t;
		}
		// reorder the label indices and the according labels
		// now comes the more challenging part of putting the trainingset
		// together. We need to make sure this is managed relatively quickly.
		RankSort r = new RankSort(VectorConv.int2double(trainingind),labels);
		trainingind = VectorConv.double2int(r.getSorted());
		labels = r.getDRank();
		// prepare a list of indices for each image
		int count = 0;
		int countind = 0;
		int len2 = 0;
		double[][] temp;
		int[] point = new int[1];
		for (int i=0; i<len; i++){
			// if we would have to go to the next image
			if (trainingind[i]>=imlen2[countind]){
				// if there were pixels in the last image
				if (len2>0){
					counter = 0;
					String featurename;
					for (int k=7; k<9; k++){
						if (defVal[k]){
							featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[0]+".tif";
							temp = StackOperations.stack2PixelArrays(IJ.openImage(featurename));
							for (int j=0; j<len2; j++){
								point[0] = trainingind[i-len2+j]-imlen3[countind];
								VectorAccess.write(trainingset[i-len2+j],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape,0,nc+1,temp)[0],counter);
							}
							counter+=temp.length*sl;
						}
					}
				}
				while (trainingind[i]>=imlen2[countind]){
					countind++;
				}
				len2 = 0;
			}
			len2++;
		}
		counter = 0;
		String featurename;
		for (int k=7; k<9; k++){
			if (defVal[k]){
				featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[0]+".tif";
				temp = StackOperations.stack2PixelArrays(IJ.openImage(featurename));
				for (int j=0; j<len2; j++){
					point[0] = trainingind[len-len2+j]-imlen3[countind];
					VectorAccess.write(trainingset[len-len2+j],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape,0,nc+1,temp)[0],counter);
				}
				counter+=temp.length*sl;
			}
		}

		// training of the actual forest
		RandomForest forest = new RandomForest(trainingset,numclasses,labels,weights,categorical,dimweights,mtry[1],maxdepth[1],maxleafsize[1],splitpurity[1],0,ntree[1]);
		String forestser = forestdir[0]+File.separator+"forest2_"+ident[0]+".ser";
		ReadForest.writeForest(forestser,forest);
	}
	
	public void applyForest(){
		int l = etl.getL();
		int[] len = etl.getImageLen();
		int[][] imdim = etl.getImageDim();
		int[][] shape = NeighborhoodSample.spiralCoord(phi[0],patchradius[0],br[0],nphi[0],nrad[0],arms[0]);
		int sl = shape.length;
		String forestser = forestdir[0]+File.separator+"forest2_"+ident[0]+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		int[] point = new int[1];
		for (int i=0; i<l; i++){
			String forestname = forestdir[0]+File.separator+"forest_"+String.format("%d",i)+"_"+ident[0]+".tif";
			String votesname = votesdir[0]+File.separator+"votes_"+String.format("%d",i)+"_"+ident[0]+".tif";
			double[][] temp2 = new double[len[i]][n[1]];
			double[][] temp;
			int counter = 0;
			for (int k=7; k<9; k++){
				if (defVal[k]){
					String featurename = featuredir+File.separator+feat[k]+String.format("%d",i)+"_"+ident[0]+".tif";
					temp = StackOperations.stack2PixelArrays(IJ.openImage(featurename));
					for (int j=0; j<len[i]; j++){
						point[0] = j;
						VectorAccess.write(temp2[j],NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape,0,nc+1,temp)[0],counter);
					}
					counter+=temp.length*sl;
				}
			}
			double[][] votes = VectorAccess.flip(forest.applyForest(temp2));
			IJ.saveAsTiff(StackOperations.convert2Stack(votes,imdim[i][0],imdim[i][1]),votesname);
			IJ.saveAsTiff(StackOperations.convert2Stack(StackOperations.maxIndex(votes),imdim[i][0],imdim[i][1],1),forestname);
		}
	}
	
	private void iterativeForest(int it){
		int[][] ind = etl.getLabels();
		int numclasses = ind.length;
		int[] imlen = etl.getImageLen();
		int[] imlen2 = VectorFun.cumsum(imlen);
		int[] imlen3 = VectorFun.sub(imlen2,imlen);
		int[][] imdim = etl.getImageDim();
		// preparation of the patch shape
		int[][] shape0 = NeighborhoodSample.spiralCoord(phi[0],patchradius[0],br[0],nphi[0],nrad[0],arms[0]);
		int[][] shape1 = NeighborhoodSample.spiralCoord(phi[1],patchradius[1],br[1],nphi[1],nrad[1],arms[1]);
		int sl = shape0.length;
		// prepare the explicit training points
		ArrayList<Integer> labelvec = new ArrayList<Integer>();
		ArrayList<Integer> trainingindvec = new ArrayList<Integer>();
		for (int i=0; i<numclasses; i++){
			int b;
			if (ppc[3]>ind[i].length){
				b = ind[i].length;
			}
			else {
				b = ppc[3];
			}
			int[] tempind = VectorAccess.access(ind[i],VectorAccess.access(Shuffle.randPerm(ind[i].length),0,b));
			for (int j=0; j<b; j++){
				labelvec.add(i);
				trainingindvec.add(tempind[j]);
			}
		}
		// preparation for the random forest
		int len = labelvec.size();
		int[] trainingind = new int[len];
		double[] labels = new double[len];
		double[] weights = VectorFun.add(labels,1);
		boolean[] categorical = new boolean[n[1]+n[2]];
		for (int i=0; i<n[2]; i++){
			categorical[n[1]+i] = true;
		}
		double[][] trainingset = new double[len][n[1]+n[2]];
		for (int i=0; i<len; i++){
			labels[i] = labelvec.get(i);
			trainingind[i] = trainingindvec.get(i);
		}
		// preparation of the dim weights
		// we want the number of samples drawn per class to be roughly in the ratio as if we
		// would have drawn sqrt samples from the individual classes
		int counter =0;
		double alpha = Math.log(mtry[2])/Math.log(n[1]+n[2]);
		double[] dimweights = VectorFun.add(new double[n[1]+n[2]],1);
		if (defVal[7]){
			int t = sl*nc;
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		if (defVal[8]){
			int t = nc;
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		VectorAccess.write(dimweights,VectorFun.add(new double[n[2]],1./Math.pow(n[2],alpha)),n[1]);
		// reorder the label indices and the according labels
		RankSort r = new RankSort(VectorConv.int2double(trainingind),labels);
		trainingind = VectorConv.double2int(r.getSorted());
		labels = r.getDRank();
		// prepare a list of indices for each image
		int count = 0;
		int countind = 0;
		int len2 = 0;
		double[][] temp, temp2;
		double[] temp3;
		int[] point = new int[1];
		for (int i=0; i<len; i++){
			// if we would have to go to the next image
			if (trainingind[i]>=imlen2[countind]){
				// if there were pixels in the last image
				if (len2>0){
					counter = 0;
					String featurename;
					for (int k=7; k<9; k++){
						if (defVal[k]){
							featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[0]+".tif";
							temp = StackOperations.stack2PixelArrays(IJ.openImage(featurename));
							for (int j=0; j<len2; j++){
								point[0] = trainingind[i-len2+j]-imlen3[countind];
								VectorAccess.write(trainingset[i-len2+j],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape0,0,nc+1,temp)[0],counter);
							}
							counter+=temp.length*sl;
						}
					}
					String forestname = forestdir[1]+File.separator+"forest_"+String.format("%d",countind)+"_"+ident[0]+".tif";
					ImagePlus imp = IJ.openImage(forestname);
					temp3 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(it+1).convertToFloat().getPixels()));
					for (int j=0; j<len2; j++){
						point[0] = trainingind[i-len2+j]-imlen3[countind];
						// the fill value for border pixels could be set differently
						VectorAccess.write(trainingset[i-len2+j],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape1,0,numclasses+1,temp3)[0],n[1]);
					}
				}
				while (trainingind[i]>=imlen2[countind]){
					countind++;
				}
				len2 = 0;
			}
			len2++;
		}
		counter = 0;
		String featurename;
		for (int k=7; k<9; k++){
			if (defVal[k]){
				featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[0]+".tif";
				temp = StackOperations.stack2PixelArrays(IJ.openImage(featurename));
				for (int j=0; j<len2; j++){
					point[0] = trainingind[len-len2+j]-imlen3[countind];
					VectorAccess.write(trainingset[len-len2+j],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape0,0,nc+1,temp)[0],counter);
				}
				counter+=temp.length*sl;
			}
		}
		String forestname = forestdir[1]+File.separator+"forest_"+String.format("%d",countind)+"_"+ident[0]+".tif";
		ImagePlus imp = IJ.openImage(forestname);
		temp3 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(it+1).convertToFloat().getPixels()));
		for (int j=0; j<len2; j++){
			point[0] = trainingind[len-len2+j]-imlen3[countind];
			// the fill value for border pixels could be set differently
			VectorAccess.write(trainingset[len-len2+j],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape1,0,numclasses+1,temp3)[0],n[1]);
		}
		// training of the actual forest
		RandomForest forest = new RandomForest(trainingset,numclasses,labels,weights,categorical,dimweights,mtry[1],maxdepth[1],maxleafsize[1],splitpurity[1],0,ntree[1]);
		String forestser = forestdir[1]+File.separator+"forest_"+String.format("%d",it)+"it_"+ident[0]+".ser";
		ReadForest.writeForest(forestser,forest);
	}
	
	public void applyIterativeForest(int it){
		int numclasses = etl.getLabels().length;
		int l = etl.getL();
		int[][] ind = etl.getLabels();
		int[][] shape0 = NeighborhoodSample.spiralCoord(phi[0],patchradius[0],br[0],nphi[0],nrad[0],arms[0]);
		int[][] shape1 = NeighborhoodSample.spiralCoord(phi[1],patchradius[1],br[1],nphi[1],nrad[1],arms[1]);
		int sl = shape0.length;
		int[] len = etl.getImageLen();
		int[][] imdim = etl.getImageDim();
		double[] temp3;
		int[] point = new int[1];
		String forestser = forestdir[1]+File.separator+"forest_"+String.format("%d",it)+"it_"+ident[0]+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		for (int i=0; i<l; i++){
			String forestname = forestdir[1]+File.separator+"forest_"+String.format("%d",i)+"_"+ident[0]+".tif";
			String votesname = votesdir[1]+File.separator+"votes_"+String.format("%d",i)+"_"+ident[0]+".tif";
			double[][] temp2 = new double[len[i]][n[1]];
			double[][] temp;
			int counter = 0;
			for (int k=7; k<9; k++){
				if (defVal[k]){
					String featurename = featuredir+File.separator+feat[k]+String.format("%d",i)+"_"+ident[0]+".tif";
					temp = StackOperations.stack2PixelArrays(IJ.openImage(featurename));
					for (int j=0; j<len[i]; j++){
						point[0] = j;
						VectorAccess.write(temp2[j],NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape0,0,nc+1,temp)[0],counter);
					}
					counter+=temp.length*sl;
				}
			}
			// prepare the stacks for classification
			ImagePlus imp = IJ.openImage(forestname);
			temp3 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(it+1).convertToFloat().getPixels()));
			// we don't want to generate an image stack the size of the patch
			// multiplied with the number of pixels... therefore we work pixel for pixel
			double[][] votes = new double[len[i]][numclasses];
			double[] point2 = new double[n[1]+n[2]];
			for (int j=0; j<len[i]; j++){
				point[0] = j;
				VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape1,0,numclasses+1,temp3)[0],n[1]);
				VectorAccess.write(point2,temp2[j],0);
				votes[j] = forest.applyForest(point2);
			}
			imp = IJ.openImage(votesname);
			// add the new votes to the iteration collection
			votes = VectorAccess.flip(votes);
			for (int j=0; j<numclasses; j++){
				imp.getStack().addSlice("None",new FloatProcessor(imdim[i][0],imdim[i][1],votes[j]));
			}
			IJ.saveAsTiff(imp,votesname);
			// add the forest to the forest iteration list
			imp = IJ.openImage(forestname);
			if (it==0){
				ImageStack stack = new ImageStack(imdim[i][0],imdim[i][1]);
				stack.addSlice(imp.getStack().getProcessor(it+1));
				stack.addSlice("None",new FloatProcessor(imp.getWidth(),imp.getHeight(),StackOperations.maxIndex(votes)),it+1);
				imp = new ImagePlus("Iterative Forest",stack);
			}
			else {
				imp.getStack().addSlice("None",new FloatProcessor(imp.getWidth(),imp.getHeight(),StackOperations.maxIndex(votes)),it+1);
			}
			IJ.saveAsTiff(imp,forestname);
		}
	}
	
	private void automatonForest(){
		int numclasses = etl.getLabels().length;
		int l = etl.getL();
		int[][] ind = etl.getLabels();
		double[][] lab = new double[numclasses][];
		for (int i=0; i<numclasses; i++){
			lab[i] = VectorFun.add(new double[ind[i].length],i);
		}
		// training locations which we can choose from
		int[] ind2 = VectorAccess.vertCat(ind);
		double[] lab2 = VectorAccess.vertCat(lab);
		RankSort rank = new RankSort(VectorConv.int2double(ind2),lab2);
		ind2 = VectorConv.double2int(rank.getSorted());
		lab2 = rank.getDRank();
		int[][] imdim = etl.getImageDim();
		int[] imlen = etl.getImageLen();
		// the total number of pixels in all images up to the i'th image
		int[] imlen2 = VectorFun.cumsum(imlen);
		// the starting pixel index of each image in regard to the total number
		// of pixels of all images
		int[] imlen3 = VectorFun.sub(imlen2,imlen);
		// now comes all the fun associated with building up the transition structures
		// needed for the automaton forest
		ImagePlus imp;
		double[] im1, im2, temp1, temp2;
		int countind = 0;
		int count = 0;
		int len = 0;
		int a;
		ArrayList<ArrayList<Integer>> transind = new ArrayList<ArrayList<Integer>>();
		for (int i=0; i<numclasses*numclasses; i++){
			transind.add(new ArrayList<Integer>());
		}
		// looking at all training transitions between all classes
		// indices are sorted according to the image they're in
		for (int i=0; i<ind2.length; i++){
			if (ind2[i]>=imlen2[countind]){
				if (len>0){
					String forestname = forestdir[1]+File.separator+"forest_"+String.format("%d",countind)+"_"+ident[0]+".tif";
					imp = IJ.openImage(forestname);
					// the tempind vector contains the indice locations of the training pixels
					// in relation to the coordinates of the current image
					int[] tempind = VectorFun.add(VectorAccess.access(ind2,i-len,i),-imlen3[countind]);
					// go through all images resulting from the iterative forest of the current image
					for (int j=1; j<iterations+1; j++){
						im1 =  VectorConv.float2double((float[])(imp.getStack().getProcessor(j).convertToFloat().getPixels()));
						im2 =  VectorConv.float2double((float[])(imp.getStack().getProcessor(j+1).convertToFloat().getPixels()));
						temp1 = VectorAccess.access(im1,tempind);
						temp2 = VectorAccess.access(im2,tempind);
						// go through all training pixel locations of the current image
						for (int k=0; k<temp1.length; k++){
							// only use training locations where the target label is the correct label
							if (lab2[k+i-len]==temp2[k]){
								a = (int)(temp1[k]*numclasses+temp2[k]);
								// store the indices according to their transition between classes
								// ... most transitions should be from one class to that same class
								transind.get(a).add((j-1)*imlen[countind]+tempind[k]+imlen3[countind]*iterations);
							}
						}
					}
				}
				while (ind2[i]>=imlen2[countind]){
					countind++;
				}
				len = 0;
			}
			len++;
		}
		String forestname = forestdir[1]+File.separator+"forest_"+String.format("%d",countind)+"_"+ident[0]+".tif";
		imp = IJ.openImage(forestname);
		int[] tempind = VectorFun.add(VectorAccess.access(ind2,ind2.length-len,ind2.length),-imlen3[countind]);
		for (int j=1; j<iterations+1; j++){
			im1 =  VectorConv.float2double((float[])(imp.getStack().getProcessor(j).convertToFloat().getPixels()));
			im2 =  VectorConv.float2double((float[])(imp.getStack().getProcessor(j+1).convertToFloat().getPixels()));
			temp1 = VectorAccess.access(im1,tempind);
			temp2 = VectorAccess.access(im2,tempind);
			for (int k=0; k<temp1.length; k++){
				if (lab2[k+ind2.length-len]==temp2[k]){
					a = (int)(temp1[k]*numclasses+temp2[k]);
					transind.get(a).add((j-1)*imlen[countind]+tempind[k]+imlen3[countind]*iterations);
				}
			}
		}
		// prepare the explicit training points
		ArrayList<Integer> labelvec = new ArrayList<Integer>();
		ArrayList<Integer> trainingindvec = new ArrayList<Integer>();
		for (int i=0; i<transind.size(); i++){
			int b;
			// check if as many training points exist as would be required for the
			// training
			if (ppc[4]>transind.get(i).size()){
				b = transind.get(i).size();
			}
			else {
				b = ppc[4];
			}
			int[] temp = VectorAccess.access(Shuffle.randPerm(transind.get(i).size()),0,b);
			for (int j=0; j<b; j++){
				labelvec.add(i%numclasses);
				trainingindvec.add(transind.get(i).get(temp[j]));
			}
		}
		double[] labels = new double[labelvec.size()]; 
		int[] trainingind = new int[labels.length];
		for (int i=0; i<labelvec.size(); i++){
			labels[i] = labelvec.get(i);
			trainingind[i] = trainingindvec.get(i);
		}
		rank = new RankSort(VectorConv.int2double(trainingind),labels);
		labels = rank.getDRank();
		trainingind = VectorConv.double2int(rank.getSorted());
		// prepare the forest vectors
		double[] weights = VectorFun.add(new double[trainingind.length],1);
		int[][] shape0 = NeighborhoodSample.spiralCoord(phi[0],patchradius[0],br[0],nphi[0],nrad[0],arms[0]);
		int[][] shape1 = NeighborhoodSample.spiralCoord(phi[1],patchradius[1],br[1],nphi[1],nrad[1],arms[1]);
		int sl = shape0.length;
		boolean[] categorical = new boolean[n[1]+n[2]];
		for (int i=0; i<n[2]; i++){
			categorical[n[1]+i] = true;
		}
		// we actually are weighting our input dimensions for a change...
		int counter =0;
		double alpha = Math.log(mtry[3])/Math.log(n[1]+n[2]);
		double[] dimweights = VectorFun.add(new double[n[1]+n[2]],1);
		if (defVal[7]){
			int t = sl*nc;
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		if (defVal[8]){
			int t = nc;
			VectorAccess.write(dimweights,VectorFun.add(new double[t],1/Math.pow(t,alpha)),counter);
			counter+=t;
		}
		VectorAccess.write(dimweights,VectorFun.add(new double[n[2]],1./Math.pow(n[2],alpha)),n[1]);
		
		double[][] trainingset = new double[trainingind.length][n[1]+n[2]];
		countind = 0;
		int countind2 = 0;
		len = 0;
		int len2;
		double[][] temp, temp3;
		double[] tempim;
		// now comes the most interweaved part of the whole automaton forest algorithm... to keep memory calls
		// limited and fairly efficient
		int[] point = new int[1];
		// go through all training pixels
		for (int i=0; i<trainingind.length; i++){
			// go through all intial images
			if (trainingind[i]>=imlen2[countind]*iterations){
				if (len>0){
					// extract the feature vector image for the current image
					temp = new double[n[1]/sl][];
					counter = 0;
					for (int k=7; k<9; k++){
						if (defVal[k]){
							String featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[0]+".tif";
							temp3 = StackOperations.stack2PixelArrays(IJ.openImage(featurename));
							for (int j=0; j<temp3.length; j++){
								temp[counter+j] = temp3[j].clone();
							}
							counter+=temp3.length;
						}
					}
					forestname = forestdir[1]+File.separator+"forest_"+String.format("%d",countind)+"_"+ident[0]+".tif";
					imp = IJ.openImage(forestname);
					a = i-len;
					// the iteration number within the current image
					countind2 = (trainingind[a]-imlen3[countind]*iterations)/(imlen[countind]);
					len2 = 0;
					for (int j=0; j<len; j++){
						if (trainingind[a+j]-imlen3[countind]*iterations>=countind2*imlen[countind]){
							if (len2>0){
								// prepare the vote image of the current iteration in the training set data
								tempim = VectorConv.float2double((float[])(imp.getStack().getProcessor(countind2+1).convertToFloat().getPixels()));
								// go through each point at the current iteration and sample the neighborhood
								for (int k=0; k<len2; k++){
									int b = a+j-len2+k;
									point[0] = (trainingind[b]-imlen3[countind]*iterations)%imlen[countind];
									VectorAccess.write(trainingset[b],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape0,0,numclasses+1,temp)[0],0);
									VectorAccess.write(trainingset[b],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape1,0,numclasses+1,tempim)[0],n[1]);
								}
								
							}
							while (trainingind[a+j]-imlen3[countind]*iterations>=countind2*imlen[countind]){
								countind2++;
							}
							len2 = 0;
						}
						len2++;
					}
					// finish the last iteration of the last image
					tempim = VectorConv.float2double((float[])(imp.getStack().getProcessor(countind2+1).convertToFloat().getPixels()));
					// go through each point at the current iteration and sample the neighborhood
					for (int k=0; k<len2; k++){
						int b = a+len-len2+k;
						point[0] = (trainingind[b]-imlen3[countind]*iterations)%imlen[countind];
						VectorAccess.write(trainingset[b],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape0,0,numclasses+1,temp)[0],0);
						VectorAccess.write(trainingset[b],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape1,0,numclasses+1,tempim)[0],n[1]);
					}
					while (trainingind[i]>=imlen2[countind]*iterations){
						countind++;
					}
					len = 0;
					
				}
			}
			len++;
		}
		// extract the feature vector image for the current image
		temp = new double[n[1]/sl][];
		counter = 0;
		for (int k=7; k<9; k++){
			if (defVal[k]){
				String featurename = featuredir+File.separator+feat[k]+String.format("%d",countind)+"_"+ident[0]+".tif";
				temp3 = StackOperations.stack2PixelArrays(IJ.openImage(featurename));
				for (int j=0; j<temp3.length; j++){
					temp[counter+j] = temp3[j].clone();
				}
				counter+=temp3.length;
			}
		}
		forestname = forestdir[1]+File.separator+"forest_"+String.format("%d",countind)+"_"+ident[0]+".tif";
		imp = IJ.openImage(forestname);
		a = trainingind.length-len;
		// the iteration number within the current image
		countind2 = (trainingind[a]-imlen3[countind]*iterations)/(imlen[countind]);
		len2 = 0;
		for (int j=0; j<len; j++){
			if (trainingind[a+j]-imlen3[countind]*iterations>=countind2*imlen[countind]){
				if (len2>0){
					// prepare the vote image of the current iteration in the training set data
					tempim = VectorConv.float2double((float[])(imp.getStack().getProcessor(countind2+1).convertToFloat().getPixels()));
					// go through each point at the current iteration and sample the neighborhood
					for (int k=0; k<len2; k++){
						int b = a+j-len2+k;
						point[0] = (trainingind[b]-imlen3[countind]*iterations)%imlen[countind];
						VectorAccess.write(trainingset[b],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape0,0,numclasses+1,temp)[0],0);
						VectorAccess.write(trainingset[b],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape1,0,numclasses+1,tempim)[0],n[1]);
					}
					
				}
				while (trainingind[a+j]-imlen3[countind]*iterations>=countind2*imlen[countind]){
					countind2++;
				}
				len2 = 0;
			}
			len2++;
		}
		// finish the last iteration of the last image
		tempim = VectorConv.float2double((float[])(imp.getStack().getProcessor(countind2+1).convertToFloat().getPixels()));
		// go through each point at the current iteration and sample the neighborhood
		for (int k=0; k<len2; k++){
			int b = a+len-len2+k;
			point[0] = (trainingind[b]-imlen3[countind]*iterations)%imlen[countind];
			VectorAccess.write(trainingset[b],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape0,0,numclasses+1,temp)[0],0);
			VectorAccess.write(trainingset[b],NeighborhoodSample.sample2d(point,imdim[countind][0],imdim[countind][1],shape1,0,numclasses+1,tempim)[0],n[1]);
		}
		// training of the actual forest
		RandomForest forest = new RandomForest(trainingset,numclasses,labels,weights,categorical,dimweights,mtry[3],maxdepth[3],maxleafsize[3],splitpurity[3],0,ntree[3]);
		String forestser = forestdir[2]+File.separator+"forest_"+ident[0]+".ser";
		ReadForest.writeForest(forestser,forest);
	}
	
	public void applyAutomatonForest(int it){
		int numclasses = etl.getLabels().length;
		int l = etl.getL();
		int[][] ind = etl.getLabels();
		int[][] shape0 = NeighborhoodSample.spiralCoord(phi[0],patchradius[0],br[0],nphi[0],nrad[0],arms[0]);
		int[][] shape1 = NeighborhoodSample.spiralCoord(phi[1],patchradius[1],br[1],nphi[1],nrad[1],arms[1]);
		int sl = shape0.length;
		int[] len = etl.getImageLen();
		int[][] imdim = etl.getImageDim();
		String forestser = forestdir[2]+File.separator+"forest_"+ident[0]+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		for (int i=0; i<l; i++){
			String votesname = votesdir[2]+File.separator+"votes_"+String.format("%d",i)+"_"+ident[0]+".tif";
			String forestname = forestdir[2]+File.separator+"forest_"+String.format("%d",i)+"_"+ident[0]+".tif";
			double[][] temp2 = new double[n[1]/sl][];
			double[][] temp;
			double[] temp3;
			int counter = 0;
			for (int k=7; k<9; k++){
				if (defVal[k]){
					String featurename = featuredir+File.separator+feat[k]+String.format("%d",i)+"_"+ident[0]+".tif";
					temp = StackOperations.stack2PixelArrays(IJ.openImage(featurename));
					for (int j=0; j<temp.length; j++){
						temp2[j+counter] = temp[j].clone();
					}
					counter+=temp.length;
				}
			}
			// prepare the stacks for classification
			int[] point = new int[1];
			ImagePlus imp = IJ.openImage(forestname);
			temp3 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(it+1).convertToFloat().getPixels()));
			// we don't want to generate an image stack the size of the patch
			// multiplied with the number of pixels... therefore we work pixel for pixel
			double[][] votes = new double[len[i]][numclasses];
			double[] point2 = new double[n[1]+n[2]];
			for (int j=0; j<len[i]; j++){
				point[0] = j;
				VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape0,0,numclasses+1,temp2)[0],0);
				VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape1,0,numclasses+1,temp3)[0],n[1]);
				votes[j] = forest.applyForest(point2);
			}
			imp = IJ.openImage(votesname);
			// add the new votes to the iteration collection
			votes = VectorAccess.flip(votes);
			for (int j=0; j<numclasses; j++){
				imp.getStack().addSlice("None",new FloatProcessor(imdim[i][0],imdim[i][1],votes[j]));
			}
			IJ.saveAsTiff(imp,votesname);
			// add the forest to the forest iteration list
			imp = IJ.openImage(forestname);
			if (it==0){
				ImageStack stack = new ImageStack(imdim[i][0],imdim[i][1]);
				stack.addSlice(imp.getStack().getProcessor(it+1));
				stack.addSlice("None",new FloatProcessor(imp.getWidth(),imp.getHeight(),StackOperations.maxIndex(votes)),it+1);
				imp = new ImagePlus("Automaton Forest",stack);
			}
			else {
				imp.getStack().addSlice("None",new FloatProcessor(imp.getWidth(),imp.getHeight(),StackOperations.maxIndex(votes)),it+1);
			}
			IJ.saveAsTiff(imp,forestname);
		}
	}
	
	public ExtractTrainingLabels getETL(){
		return etl;
	}
}