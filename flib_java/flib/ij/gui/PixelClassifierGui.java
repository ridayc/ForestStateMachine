package flib.ij.gui;

import java.lang.System;
import java.lang.Math;
import java.lang.Class;
import java.lang.Exception;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
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
import flib.math.random.Sample;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.clustering.RFC;
import flib.algorithms.randomforest.ReadForest;
import flib.algorithms.sampling.NeighborhoodSample;

/* The PixelClassifierGui class combines many previously developed random forest related
classification tools into a large framework with the purpose of multifaceted classification.
*/

public class PixelClassifierGui
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
	// maxdepth: maximal depth a decision tree can have
	// splitpurity: fraction of points which need to belong to a class before forming a leaf node
	// maxleafsize: the maximum number of points required in an impure leaf node
	// ntree: number of decision trees in the forest
	// mtry: number of random dimensions to choose at each decision node
	
	// LogGabor+LogGaborBand parameters
	private double[] lambda1 = new double[]{2,2,2};
	private double[] lambda2 = new double[]{80,3,80};
	private double lambda3 = 120; 
	private double[] sigma = new double[]{2,2,2};
	private double[] ang = new double[]{1,-5,1};
	private int[] numstep = new int[]{5,5,5};
	private int[] numang = new int[]{16,1,16};
	private int[] numphase = new int[]{2,1,2};
	
	// Forest parameters
	private int[] ppc = new int[]{2000,200,200,200,200};
	private int[] ntree = new int[]{200,200,200,200,200};
	private int[] mtry = new int[]{0,0,0,0,0};
	private int[] maxdepth = new int[]{100,100,100,100};
	private int[] maxleafsize = new int[]{5,3,2,3,2};
	private double[] splitpurity = new double[]{0.9,0.9,0.6,1,1};
	double[] splitpurity2 = new double[]{1,1,1};
	private int[] iterations = new int[]{10,20};
	
	// RFC parameters
	private int[] nc = new int[]{4,2,4};
	private double[] balance = new double[]{2,2,2};
	
	// Patch parameters
	private double phi = 1, patchradius = 5, br = 1, nphi = 1, nrad = 1;
	private int arms = 4, nps;
	// Automaton parameters
	private double[] trainingbalance = new double[]{1,1};
	
	// other
	// checkboxes
	private boolean[] overwrite = new boolean[]{false,false,false,false,false,false,false};
	private double[] factor = new double[3];
	private ExtractTrainingLabels etl;
	private LayerSet layers;
	private int[] imlen,imlen2,imlen3;
	private int[][] imdim;
	private String rfcforestdir, lrfcforestdir, subforestdir, simpleforestdir, iterativeforestdir, automatonforestdir;
	private String[] votesdir = new String[3];
	private String[] ident = new String[5];
	private String[] feat = new String[7];
	private int type, numclasses, numim, numdim = numstep[0]*numang[0]*numphase[0]+numstep[1]*numang[1]*numphase[1]+numstep[2], numdim2;
	// Strings
	private String targetdir, featuredir, infodir,featname, basename, RFCname,rfcforest, LRFCname,lrfcforest, subname, subforest, simplename, simpleforest, automatonname, automatonforest;
	
	public PixelClassifierGui(){
		this("");
	}
	
	public PixelClassifierGui(String projectname){
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
		layers = p.getRootLayerSet();
		etl = new ExtractTrainingLabels(p);
		numim = etl.getL();
		imlen = etl.getImageLen();
		imlen2 = VectorFun.cumsum(imlen);
		imlen3 = VectorFun.sub(imlen2,imlen);
		imdim = etl.getImageDim();
		numclasses = etl.getLabels().length;
		// name for the feature directory
		featuredir = targetdir+File.separator+"features";
		infodir = targetdir+File.separator+"info";
		rfcforestdir = targetdir+File.separator+"rfcforest";
		subforestdir = targetdir+File.separator+"subforest";
		lrfcforestdir = targetdir+File.separator+"lrfcforest";
		simpleforestdir = targetdir+File.separator+"simpleforest";
		iterativeforestdir = targetdir+File.separator+"iterativeforest";
		automatonforestdir = targetdir+File.separator+"automatonforest";
		featname = "feat";
		// prepare the features folder if it doesn't exist
		new File(featuredir).mkdirs();
		new File(infodir).mkdirs();
		new File(rfcforestdir).mkdirs();
		new File(subforestdir).mkdirs();
		new File(lrfcforestdir).mkdirs();
		new File(simpleforestdir).mkdirs();
		new File(iterativeforestdir).mkdirs();
		new File(automatonforestdir).mkdirs();
		// show the parameters dialogue
		transformationDialogue();
		RFCDialogue();
		subclassDialogue();
		automatonDialogue();
		//LRFCDialogue();
		IJ.log("Program started running: "+String.format((new Date()).toString()));
		transformationSetup();
		RFCSetup();
		applyRFC();
		subclassSetup();
		//LRFCSetup();
		applySubclass();
		applyLRFC();
		simpleForest();
		applySimpleForest();
		String name = iterativeforestdir+File.separator+automatonname+"_"+String.format("%d",0)+".ser";
		if(!(new File(name)).exists()||overwrite[5]){
			for (int i=0; i<numim; i++){
				ReadForest.copyFile(simpleforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".tif",
					iterativeforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".tif");
				ReadForest.copyFile(simpleforestdir+File.separator+automatonname+"_votes_"+String.format("%d",i)+".tif",
					iterativeforestdir+File.separator+automatonname+"_votes_"+String.format("%d",i)+".tif");
			}
			for (int i=0; i<iterations[0]; i++){
				iterativeForest(i);
				applyIterativeForest(i);
			}
		}
		name = automatonforestdir+File.separator+automatonname+"_"+String.format("%d",0)+".ser";
		if(!(new File(name)).exists()||overwrite[6]){
			for (int i=0; i<numim; i++){
				ReadForest.copyFile(simpleforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".tif",
					automatonforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".tif");
				ReadForest.copyFile(simpleforestdir+File.separator+automatonname+"_votes_"+String.format("%d",i)+".tif",
					automatonforestdir+File.separator+automatonname+"_votes_"+String.format("%d",i)+".tif");
			}
			automatonForest();
		}
		for (int i=0; i<iterations[1]; i++){
			applyAutomatonForest();
		}
	}
	
	private void transformationDialogue(){
		type = 1;
		GenericDialog gd = new GenericDialog("Transformation Parameters");
		gd.addStringField("Base File Name","untitled",20);
		gd.addCheckbox("overwrite",overwrite[0]);
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
		gd.addNumericField("lambda3: ", lambda3, 2);
		gd.addNumericField("Bandwidth: ", sigma[1], 2);
		gd.addNumericField("Number of steps: ", numstep[1], 0);
		gd.addNumericField("Number of angles: ", numang[1], 0);
		gd.addNumericField("Angular width: ", ang[1], 2);
		gd.addNumericField("Number of phases: ", numphase[1], 0);
		gd.addMessage(" \n LogGabor Maxed Angle Parameters: \n");
		gd.addNumericField("lambda1: ", lambda1[2], 2);
		gd.addNumericField("lambda2: ", lambda2[2], 2);
		gd.addNumericField("Bandwidth: ", sigma[2], 2);
		gd.addNumericField("Number of steps: ", numstep[2], 0);
		gd.addNumericField("Number of angles: ", numang[2], 0);
		gd.addNumericField("Angular width: ", ang[2], 2);
		gd.addNumericField("Number of phases: ", numphase[2], 0);
		gd.addDialogListener(this);
		gd.showDialog();
		if (gd.wasCanceled()) return;
		basename = gd.getNextString();
		basename = basename.replaceAll("//s+","");
		overwrite[0] = gd.getNextBoolean();
		lambda1[0] = gd.getNextNumber();
		lambda2[0] = gd.getNextNumber();
		sigma[0] = gd.getNextNumber();
		numstep[0] = (int)gd.getNextNumber();
		numang[0] = (int)gd.getNextNumber();
		ang[0] = gd.getNextNumber();
		numphase[0] = (int)gd.getNextNumber();
		lambda1[1] = gd.getNextNumber();
		lambda2[1] = gd.getNextNumber();
		lambda3 = gd.getNextNumber();
		sigma[1] = gd.getNextNumber();
		numstep[1] = (int)gd.getNextNumber();
		numang[1] = (int)gd.getNextNumber();
		ang[1] = gd.getNextNumber();
		numphase[1] = (int)gd.getNextNumber();
		lambda1[2] = gd.getNextNumber();
		lambda2[2] = gd.getNextNumber();
		sigma[2] = gd.getNextNumber();
		numstep[2] = (int)gd.getNextNumber();
		numang[2] = (int)gd.getNextNumber();
		ang[2] = gd.getNextNumber();
		numphase[2] = (int)gd.getNextNumber();
		factor[0] = Math.pow(lambda2[0]/lambda1[0],1.0/(numstep[0]-1));
		factor[1] = Math.pow(lambda3/lambda2[1],1.0/(numstep[1]-1));
		factor[2] = Math.pow(lambda2[2]/lambda1[2],1.0/(numstep[2]-1));
		numdim = numstep[0]*numang[0]*numphase[0]+numstep[1]*numang[1]*numphase[1]+numstep[2];
	}
	
	private void RFCDialogue(){
		type = 0;
		// check if there already exists a feature file which will be used to get the number of feature dimensions
		String name = featuredir+File.separator+basename+"_"+String.format("%d",0)+".tif";
		if((new File(name)).exists()&&!overwrite[0]){
			numdim = (StackOperations.stack2PixelArrays(IJ.openImage(name))).length;
		}
		GenericDialog gd = new GenericDialog("Random Forest Clustering Parameters");
		gd.addStringField("RFC File Name","untitled",20);
		gd.addCheckbox("overwrite",overwrite[1]);
		gd.addMessage(" \n RFC parameters: \n");
		gd.addNumericField("Number of clusters: ", nc[0], 0);
		gd.addNumericField("Cluster balance: ", balance[0], 0);
		gd.addNumericField("Number of sample points: ", ppc[0], 0);
		gd.addNumericField("Number of trees: ", ntree[0], 0);
		gd.addNumericField("Number of random dimensions ("+String.format("%d",numdim)+"): ", mtry[0], 0);
		gd.addNumericField("Maximum tree depth: ", maxdepth[0], 0);
		gd.addNumericField("Maximmal number of points in a leaf node: ", maxleafsize[0], 0);
		gd.addNumericField("Label purity required for a leaf node in the RFC: ", splitpurity[0], 2);
		gd.addNumericField("Label purity required for a leaf node after the RFC: ", splitpurity2[0], 2);
		gd.showDialog();
		if (gd.wasCanceled()) return;
		RFCname = gd.getNextString();
		RFCname = RFCname.replaceAll("//s+","");
		overwrite[1] = gd.getNextBoolean();
		nc[0] = (int)gd.getNextNumber();
		balance[0] = gd.getNextNumber();
		ppc[0] = (int)gd.getNextNumber();
		ntree[0] = (int)gd.getNextNumber();
		mtry[0] = (int)gd.getNextNumber();
		if (mtry[0]<1){
			mtry[0] = (int)Math.sqrt(numdim);
		}
		maxdepth[0] = (int)gd.getNextNumber();
		maxleafsize[0] = (int)gd.getNextNumber();
		splitpurity[0] = gd.getNextNumber();
		splitpurity2[0] = gd.getNextNumber();
	}
	
	private void subclassDialogue(){
		type = 0;
		// check if there already exists a feature file which will be used to get the number of feature dimensions
		String name = featuredir+File.separator+basename+"_"+String.format("%d",0)+".tif";
		if((new File(name)).exists()&&!overwrite[0]){
			numdim = (StackOperations.stack2PixelArrays(IJ.openImage(name))).length;
		}
		GenericDialog gd = new GenericDialog("Subclassing Parameters");
		gd.addStringField("Subclassing File Name","untitled",20);
		gd.addCheckbox("overwrite",overwrite[2]);
		gd.addMessage(" \n Subclassing parameters: \n");
		gd.addNumericField("Number of sub clusters: ", nc[1], 0);
		gd.addNumericField("Cluster balance: ", balance[1], 0);
		gd.addNumericField("Number of sample points: ", ppc[1], 0);
		gd.addNumericField("Number of trees: ", ntree[1], 0);
		gd.addNumericField("Number of random dimensions ("+String.format("%d",numdim)+"): ", mtry[1], 0);
		gd.addNumericField("Maximum tree depth: ", maxdepth[1], 0);
		gd.addNumericField("Maximmal number of points in a leaf node: ", maxleafsize[1], 0);
		gd.addNumericField("Label purity required for a leaf node in the RFC: ", splitpurity[1], 2);
		gd.addNumericField("Label purity required for a leaf node after the RFC: ", splitpurity2[1], 2);
		gd.addStringField("LRFC File Name","untitled",20);
		gd.addCheckbox("overwrite",overwrite[3]);
		gd.addMessage(" \n LRFC parameters: \n");
		gd.addNumericField("Number of clusters: ", nc[2], 0);
		gd.addNumericField("Cluster balance: ", balance[2], 0);
		gd.addNumericField("Number of sample points: ", ppc[2], 0);
		gd.addNumericField("Number of trees: ", ntree[2], 0);
		gd.addNumericField("Number of random dimensions ("+String.format("%d",numdim)+"): ", mtry[2], 0);
		gd.addNumericField("Maximum tree depth: ", maxdepth[2], 0);
		gd.addNumericField("Maximmal number of points in a leaf node: ", maxleafsize[2], 0);
		gd.addNumericField("Label purity required for a leaf node in the RFC: ", splitpurity[2], 2);
		gd.addNumericField("Label purity required for a leaf node after the RFC: ", splitpurity2[2], 2);
		gd.showDialog();
		if (gd.wasCanceled()) return;
		subname = gd.getNextString();
		subname = subname.replaceAll("//s+","");
		overwrite[2] = gd.getNextBoolean();
		nc[1] = (int)gd.getNextNumber();
		balance[1] = gd.getNextNumber();
		ppc[1] = (int)gd.getNextNumber();
		ntree[1] = (int)gd.getNextNumber();
		mtry[1] = (int)gd.getNextNumber();
		if (mtry[1]<1){
			mtry[1] = (int)Math.sqrt(numdim);
		}
		maxdepth[1] = (int)gd.getNextNumber();
		maxleafsize[1] = (int)gd.getNextNumber();
		splitpurity[1] = gd.getNextNumber();
		splitpurity2[1] = gd.getNextNumber();
		LRFCname = gd.getNextString();
		LRFCname = LRFCname.replaceAll("//s+","");
		overwrite[3] = gd.getNextBoolean();
		nc[2] = (int)gd.getNextNumber();
		balance[2] = gd.getNextNumber();
		ppc[2] = (int)gd.getNextNumber();
		ntree[2] = (int)gd.getNextNumber();
		mtry[2] = (int)gd.getNextNumber();
		if (mtry[2]<1){
			mtry[2] = (int)Math.sqrt(numdim);
		}
		maxdepth[2] = (int)gd.getNextNumber();
		maxleafsize[2] = (int)gd.getNextNumber();
		splitpurity[2] = gd.getNextNumber();
		splitpurity2[2] = gd.getNextNumber();
	}
	
	private void automatonDialogue(){
		type = 2;
		// check if there already exists a feature file which will be used to get the number of feature dimensions
		String name = featuredir+File.separator+basename+"_"+String.format("%d",0)+".tif";
		GenericDialog gd = new GenericDialog("Automaton Parameters");
		gd.addStringField("Automaton File Name","untitled",20);
		gd.addCheckbox("overwrite initial",overwrite[4]);
		gd.addCheckbox("overwrite iterative",overwrite[5]);
		gd.addCheckbox("overwrite automaton",overwrite[6]);
		gd.addMessage(" \n Iterative forest parameters: \n");
		gd.addNumericField("Number of training iterations: ", iterations[0], 0);
		gd.addNumericField("Number of automaton iterations: ", iterations[1], 0);
		gd.addMessage("\n Smoothing Spiral: \n");
		gd.addNumericField("Angular increment: ", phi, 2);
		gd.addNumericField("Patch Radius: ", patchradius, 0);
		gd.addNumericField("Radial increment: ", br, 2);
		gd.addNumericField("Angular exponent: ", nphi, 2);
		gd.addNumericField("Radial exponent: ", nrad, 2);
		gd.addNumericField("Spiral Arms: ", arms, 0);
		gd.addMessage(" \n Random Forest Parameters (Iterations): \n");
		gd.addNumericField("Number of Training Points per Class: ", ppc[3], 0);
		gd.addNumericField("Training Weight Rebalancing: ", trainingbalance[0], 2);
		gd.addNumericField("Number of Trees: ", ntree[3], 0);
		gd.addNumericField("Number of Random Dimensions: ", mtry[3], 0);
		gd.addNumericField("Maximum Tree Depth: ", maxdepth[3], 0);
		gd.addNumericField("Maximmal Number of Points in a Leaf Node: ", maxleafsize[3], 0);
		gd.addNumericField("Label Purity required for a Leaf Node: ", splitpurity[3], 2);
		gd.addMessage(" \n Random Forest Parameters (Automaton): \n");
		gd.addNumericField("Number of Training Points per Class: ", ppc[4], 0);
		gd.addNumericField("Training Weight Rebalancing: ", trainingbalance[1], 2);
		gd.addNumericField("Number of Trees: ", ntree[4], 0);
		gd.addNumericField("Number of Random Dimensions: ", mtry[4], 0);
		gd.addNumericField("Label Purity required for a Leaf Node: ", splitpurity[4], 2);
		gd.addDialogListener(this);
		gd.showDialog();
		if (gd.wasCanceled()) return;
		automatonname = gd.getNextString();
		automatonname = automatonname.replaceAll("//s+","");
		overwrite[4] = gd.getNextBoolean();
		overwrite[5] = gd.getNextBoolean();
		overwrite[6] = gd.getNextBoolean();
		iterations[0] = (int)gd.getNextNumber();
		iterations[1] = (int)gd.getNextNumber();
		phi = gd.getNextNumber();
		patchradius = gd.getNextNumber();
		br = gd.getNextNumber();
		nphi = gd.getNextNumber();
		nrad = gd.getNextNumber();
		arms = (int)gd.getNextNumber();
		nps = NeighborhoodSample.spiralCoord(phi,patchradius,br,nphi,nrad,arms).length;
		numdim2 = numdim+nps*3;
		ppc[3] = (int)gd.getNextNumber();
		trainingbalance[0] = gd.getNextNumber();
		ntree[3] = (int)gd.getNextNumber();
		mtry[3] = (int)gd.getNextNumber();
		if (mtry[3]<1){
			mtry[3] = (int)Math.sqrt(numdim2);
		}
		maxdepth[3] = (int)gd.getNextNumber();
		maxleafsize[3] = (int)gd.getNextNumber();
		splitpurity[3] = gd.getNextNumber();
		ppc[4] = (int)gd.getNextNumber();
		trainingbalance[1] = gd.getNextNumber();
		ntree[4] = (int)gd.getNextNumber();
		mtry[4] = (int)gd.getNextNumber();
		if (mtry[4]<1){
			mtry[4] = (int)Math.sqrt(numdim2);
		}
		splitpurity[4] = gd.getNextNumber();
	}
	
	public boolean dialogItemChanged(GenericDialog gd, AWTEvent e){
		if (type==1){
			gd.getNextNumber();
			gd.getNextNumber();
			gd.getNextNumber();
			numstep[0] = (int)gd.getNextNumber();
			numang[0] = (int)gd.getNextNumber();
			gd.getNextNumber();
			numphase[0] = (int)gd.getNextNumber();
			gd.getNextNumber();
			gd.getNextNumber();
			gd.getNextNumber();
			gd.getNextNumber();
			numstep[1] = (int)gd.getNextNumber();
			numang[1] = (int)gd.getNextNumber();
			gd.getNextNumber();
			numphase[1] = (int)gd.getNextNumber();
			lambda1[2] = gd.getNextNumber();
			lambda2[2] = gd.getNextNumber();
			sigma[2] = gd.getNextNumber();
			numstep[2] = (int)gd.getNextNumber();
			numang[2] = (int)gd.getNextNumber();
			numdim = numstep[0]*numang[0]*numphase[0]+numstep[1]*numang[1]*numphase[1]+numstep[2];
			IJ.log("Number of feature dimensions: "+String.format("%d",numdim));
		}
		else if (type==2){
			gd.getNextNumber();
			gd.getNextNumber();
			phi = gd.getNextNumber();
			patchradius = gd.getNextNumber();
			br = gd.getNextNumber();
			nphi = gd.getNextNumber();
			nrad = gd.getNextNumber();
			arms = (int)gd.getNextNumber();
			nps = NeighborhoodSample.spiralCoord(phi,patchradius,br,nphi,nrad,arms).length;
			numdim2 = numdim+nps*3;
			IJ.log("Maximum number of iterative of training dimensions: "+String.format("%d",numdim2));
		}
		return true;
	}
	
	private void transformationSetup(){
		// test if the first image of the patch images has the corresponding transformation set
		String name = featuredir+File.separator+basename+"_"+String.format("%d",0)+".tif";
		// prepare an info txt file which contains the parameters used 
		if(!(new File(name)).exists()||overwrite[0]){
			name = infodir+File.separator+basename+"_"+featname+".txt";
			try {
				File infotxt = new File(name);
				BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
				String newline = System.getProperty("line.separator");
				// now comes the ugly storage of all the parameters...
				writer.write("Log Gabor parameters:"+newline+"lambda1: "+String.format("%.2f",lambda1[0])+
					newline+"lambda2: "+String.format("%.2f",lambda2[0])+newline+"Bandwidth: "+String.format("%.2f",sigma[0])+
					newline+"Number of steps: "+String.format("%d",numstep[0])+newline+"Number of angles: "+String.format("%d",numang[0])+
					newline+"Angular width: "+String.format("%.2f",ang[0])+newline+"Number of phases: "+String.format("%d",numphase[0])+
					newline+newline+
					"Log Gabor Band parameters:"+newline+"lambda1: "+String.format("%.2f",lambda1[1])+
					newline+"lambda2: "+String.format("%.2f",lambda2[1])+newline+"lambda3: "+String.format("%.2f",lambda3)+newline+"Bandwidth: "+String.format("%.2f",sigma[1])+
					newline+"Number of steps: "+String.format("%d",numstep[1])+newline+"Number of angles: "+String.format("%d",numang[1])+
					newline+"Angular width: "+String.format("%.2f",ang[1])+newline+"Number of phases: "+String.format("%d",numphase[1])+
					newline+newline+
					"Log Gabor maxed angle parameters:"+newline+"lambda1: "+String.format("%.2f",lambda1[2])+
					newline+"lambda2: "+String.format("%.2f",lambda2[2])+newline+"Bandwidth: "+String.format("%.2f",sigma[2])+
					newline+"Number of steps: "+String.format("%d",numstep[2])+newline+"Number of angles: "+String.format("%d",numang[2])+
					newline+"Angular width: "+String.format("%.2f",ang[2])+newline+"Number of phases: "+String.format("%d",numphase[2]));
				writer.close();
			}
			catch (Exception e){}
		}
			
		// prepare whatever feature vectors need preparing
		for (int i=0; i<numim; i++){
			// first we check if a set of transformation images already exists
			name = featuredir+File.separator+basename+"_"+String.format("%d",i)+".tif";
			if(!(new File(name)).exists()||overwrite[0]){
				// prepare an info txt file which contains the parameters used 
				ImagePlus imp = ((Patch)(layers.getLayers().get(i).getDisplayables(Patch.class).get(0))).getImagePlus();
				ImagePlus featimp1 = ScaleFeatureStack.scaleFeatures(imp,lambda1[0],factor[0],numstep[0],numang[0],ang[0],numphase[0],sigma[0]);
				ImagePlus featimp2 = ScaleFeatureStack.bandScaleFeatures(imp,lambda1[1],lambda2[1],factor[1],numstep[1],numang[1],ang[1],numphase[1],sigma[1]);
				ImagePlus featimp3 = ScaleFeatureStack.maxDim(ScaleFeatureStack.scaleFeatures(imp,lambda1[2],factor[2],numstep[2],numang[2],ang[2],numphase[2],sigma[2]),numstep[2],numang[2],numphase[2],3);
				IJ.saveAsTiff(StackOperations.merge(featimp1,featimp2,featimp3),name);
			}
		}
	}
	
	private void RFCSetup(){
		String name;
		String forestser = rfcforestdir+File.separator+RFCname+".ser";
		if(!(new File(forestser)).exists()||overwrite[1]){
			// prepare an info txt file which contains the parameters used
			name = infodir+File.separator+RFCname+"_RFC.txt";
			try {
				File infotxt = new File(name);
				BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
				String newline = System.getProperty("line.separator");
				// now comes the ugly storage of all the parameters...
				writer.write("RFC parameters:"+newline+"Feature file: " +basename+"_"+featname+".txt"+newline+"Number of clusters: "+String.format("%d",nc[0])+
					newline+"Cluster balance: "+String.format("%.2f",balance[0])+newline+"Number of sample points: "+String.format("%d",ppc[0])+
					newline+"Number of trees: "+String.format("%d",ntree[0])+newline+"Number of random dimensions ("+String.format("%d",numdim)+"): "+String.format("%d",mtry[0])+
					newline+"Maximum tree depth: "+String.format("%d",maxdepth[0])+newline+"Maximum leaf size: "+String.format("%d",maxleafsize[0])+
					newline+"Label purity required before the RFC: "+String.format("%.2f",splitpurity[0])+
					newline+"Label purity required after the RFC: "+String.format("%.2f",splitpurity2[0]));
				writer.close();
			}
			catch (Exception e){}
			// choose a set of random points for clustering
			int l = ppc[0];
			int[] trainingloc = Sample.sample(imlen2[numim-1],l);
			RankSort r = new RankSort(VectorConv.int2double(trainingloc));
			trainingloc = VectorConv.double2int(r.getSorted());
			double[][] trainingset = new double[2*l][];
			double[] labelset = new double[2*l];
			double[] weights = VectorFun.add(new double[2*l],1);
			VectorAccess.write(labelset,VectorFun.add(new double[l],1),l);
			int count = 0;
			int len = 0;
			double[][] temp;
			for (int i=0; i<l; i++){
				if(trainingloc[i]>=imlen2[count]){
					name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
					temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
					if (len>0){
						for (int j=0; j<len; j++){
							trainingset[i-len+j] = temp[trainingloc[i-len+j]-imlen3[count]].clone();
						}
					}
					while (trainingloc[i]>=imlen2[count]){
						count++;
					}
					len = 0;
				}
				len++;
			}
			name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
			temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
			if (len>0){
				for (int j=0; j<len; j++){
					trainingset[l-len+j] = temp[trainingloc[l-len+j]-imlen3[count]].clone();
				}
			}
			//IJ.log("Number of training dimensions: "+String.format("%d",dim));
			boolean[] categorical = new boolean[numdim];
			double[] dimweights = new double[0];
			// generation of the artificial random training data
			Random rand = new Random();
			for (int i=0; i<l; i++){
				trainingset[i+l] = new double[numdim];
				for (int j=0; j<numdim; j++){
					trainingset[i+l][j] = trainingset[rand.nextInt(l)][j];
				}
			}
			RandomForest forest = new RandomForest(trainingset,2,labelset,weights,categorical,dimweights,mtry[0],maxdepth[0],maxleafsize[0],splitpurity[0],1,ntree[0]);
			double[][] trainingset2 = new double[ppc[0]][numdim];
			for (int i=0; i<l; i++){
				trainingset2[i] = trainingset[i].clone();
			}
			// Here follows the random forest clustering
			int[][] leafindices = forest.getLeafIndices(trainingset2);
			RFC rfc = new RFC(leafindices,nc[0],forest.getTreeSizes(),balance[0]);
			int[] clusters = rfc.assignCluster(leafindices);
			// base a random forest off of these values
			forest = new RandomForest(trainingset2,nc[0],VectorConv.int2double(clusters),VectorFun.add(new double[l],1),categorical,dimweights,mtry[0],maxdepth[0],maxleafsize[0],splitpurity2[0],0,ntree[0]);
			ReadForest.writeForest(forestser,forest);
		}
	}
	
	private void applyRFC(){
		String forestser = rfcforestdir+File.separator+RFCname+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		String name = rfcforestdir+File.separator+RFCname+"_"+String.format("%d",0)+".tif";
		if(!(new File(name)).exists()||overwrite[1]){
			for (int i=0; i<numim; i++){
				String forestname = rfcforestdir+File.separator+RFCname+"_"+String.format("%d",i)+".tif";
				String votesname = rfcforestdir+File.separator+RFCname+"_votes_"+String.format("%d",i)+".tif";
				double[][] temp;
				name = featuredir+File.separator+basename+"_"+String.format("%d",i)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
				double[][] votes = VectorAccess.flip(forest.applyForest(temp));
				IJ.saveAsTiff(StackOperations.convert2Stack(votes,imdim[i][0],imdim[i][1]),votesname);
				IJ.saveAsTiff(StackOperations.convert2Stack(StackOperations.maxIndex(votes),imdim[i][0],imdim[i][1],1),forestname);
			}
		}
	}
	
	private void subclassSetup(){
		String name;
		String forestser = subforestdir+File.separator+subname+".ser";
		if(!(new File(forestser)).exists()||overwrite[2]){
			// prepare an info txt file which contains the parameters used
			name = infodir+File.separator+subname+"_sub.txt";
			try {
				File infotxt = new File(name);
				BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
				String newline = System.getProperty("line.separator");
				// now comes the ugly storage of all the parameters...
				writer.write("Subclass parameters:"+newline+"Feature file: " +basename+newline+"Number of sub classes: "+String.format("%d",nc[1])+
					newline+"Cluster balance: "+String.format("%.2f",balance[1])+newline+"Number of sample points: "+String.format("%d",ppc[1])+
					newline+"Number of trees: "+String.format("%d",ntree[1])+newline+"Number of random dimensions ("+String.format("%d",numdim)+"): "+String.format("%d",mtry[1])+
					newline+"Maximum tree depth: "+String.format("%d",maxdepth[1])+newline+"Maximum leaf size: "+String.format("%d",maxleafsize[1])+
					newline+"Label purity required before the RFC: "+String.format("%.2f",splitpurity[1])+
					newline+"Label purity required after the RFC: "+String.format("%.2f",splitpurity2[1]));
				writer.close();
			}
			catch (Exception e){}
			// choose a set of random points for clustering
			int l = ppc[1];
			int[] trainingloc = new int[l];
			int[][] labels = etl.getLabels();
			double[][] trainingset = new double[2*l][numdim];
			double[] weights = VectorFun.add(new double[2*l],1);
			double[] labelset = new double[2*l];
			VectorAccess.write(labelset,VectorFun.add(new double[l],1),l);
			boolean[] categorical = new boolean[numdim];
			double[] dimweights = new double[0];
			double[][] trainingset2 = new double[l][numdim];
			double[][] trainingset3;
			ArrayList<ArrayList<Integer>> sublabellist = new ArrayList<ArrayList<Integer>>();
			Random rand = new Random();
			// for all classes
			for (int i=0; i<numclasses; i++){
				trainingloc = Sample.sample(labels[i].length,l);
				RankSort r = new RankSort(VectorConv.int2double(trainingloc));
				trainingloc = VectorConv.double2int(r.getSorted());
				int count = 0;
				int len = 0;
				double[][] temp;
				for (int j=0; j<l; j++){
					if(trainingloc[j]>=imlen2[count]){
						name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
						temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
						if (len>0){
							for (int k=0; k<len; k++){
								trainingset[j-len+k] = temp[trainingloc[j-len+k]-imlen3[count]].clone();
							}
						}
						while (trainingloc[j]>=imlen2[count]){
							count++;
						}
						len = 0;
					}
					len++;
				}
				name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
				if (len>0){
					for (int k=0; k<len; k++){
						trainingset[l-len+k] = temp[trainingloc[l-len+k]-imlen3[count]].clone();
					}
				}
				for (int j=0; j<l; j++){
					trainingset[j+l] = new double[numdim];
					for (int k=0; k<numdim; k++){
						trainingset[j+l][k] = trainingset[rand.nextInt(l)][k];
					}
				}
				RandomForest forest = new RandomForest(trainingset,2,labelset,weights,categorical,dimweights,mtry[1],maxdepth[1],maxleafsize[1],splitpurity[1],1,ntree[1]);
				for (int j=0; j<l; j++){
					trainingset2[j] = trainingset[j].clone();
				}
					// Here follows the random forest clustering
				int[][] leafindices = forest.getLeafIndices(trainingset2);
				RFC rfc = new RFC(leafindices,nc[1],forest.getTreeSizes(),balance[1]);
				// assign all label points to a cluster
				trainingset3 = new double[labels[i].length][numdim];
				count = 0;
				len = 0;
				for (int j=0; j<labels[i].length; j++){
					if(labels[i][j]>=imlen2[count]){
						name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
						temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
						if (len>0){
							for (int k=0; k<len; k++){
								trainingset3[j-len+k] = temp[labels[i][j-len+k]-imlen3[count]].clone();
							}
						}
						while (labels[i][j]>=imlen2[count]){
							count++;
						}
						len = 0;
					}
					len++;
				}
				name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
				if (len>0){
					for (int k=0; k<len; k++){
						trainingset3[labels[i].length-len+k] = temp[labels[i][labels[i].length-len+k]-imlen3[count]].clone();
					}
				}
				leafindices = forest.getLeafIndices(trainingset3);
				int[] clusters = rfc.assignCluster(leafindices);
				for (int j=0; j<nc[1]; j++){
					sublabellist.add(new ArrayList<Integer>());
				}
				// Assign all the label points to one of the new clusters
				for (int j=0; j<clusters.length; j++){
					sublabellist.get(nc[1]*i+clusters[j]).add(labels[i][j]);
				}
			}
			int[][] sublabels = new int[sublabellist.size()][];
			for (int i=0; i<sublabellist.size(); i++){
				sublabels[i] = new int[sublabellist.get(i).size()];
				for (int j=0; j<sublabellist.get(i).size(); j++){
					sublabels[i][j] = sublabellist.get(i).get(j);
				}
			}
			trainingloc = new int[sublabels.length*l];
			trainingset = new double[sublabels.length*l][numdim];
			weights = VectorFun.add(new double[sublabels.length*l],1);
			labelset = new double[sublabels.length*l];
			for (int i=0; i<sublabels.length; i++){
				VectorAccess.write(labelset,VectorFun.add(new double[l],i),l*i);
				VectorAccess.write(trainingloc,VectorAccess.access(sublabels[i],Sample.sample(sublabels[i].length,l)),l*i);
			}
			RankSort r = new RankSort(VectorConv.int2double(trainingloc),labelset);
			trainingloc = VectorConv.double2int(r.getSorted());
			labelset = r.getDRank();
			int count = 0;
			int len = 0;
			double[][] temp;
			for (int i=0; i<l*sublabels.length; i++){
				if(trainingloc[i]>=imlen2[count]){
					name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
					temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
					if (len>0){
						for (int j=0; j<len; j++){
							trainingset[i-len+j] = temp[trainingloc[i-len+j]-imlen3[count]].clone();
						}
					}
					while (trainingloc[i]>=imlen2[count]){
						count++;
					}
					len = 0;
				}
				len++;
			}
			name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
			temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
			if (len>0){
				for (int j=0; j<len; j++){
					trainingset[l*sublabels.length-len+j] = temp[trainingloc[l*sublabels.length-len+j]-imlen3[count]].clone();
				}
			}
			RandomForest forest = new RandomForest(trainingset,sublabels.length,labelset,weights,categorical,dimweights,mtry[1],maxdepth[1],maxleafsize[1],splitpurity[1],0,ntree[1]);
			ReadForest.writeForest(forestser,forest);
			name = subforestdir+File.separator+subname+"_subclass.ser";
			ReadForest.writeIntAA(name,sublabels);
			forestser = lrfcforestdir+File.separator+LRFCname+".ser";
			if(!(new File(forestser)).exists()||overwrite[3]){
				// prepare an info txt file which contains the parameters used
				name = infodir+File.separator+LRFCname+"_LRFC.txt";
				try {
					File infotxt = new File(name);
					BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
					String newline = System.getProperty("line.separator");
					// now comes the ugly storage of all the parameters...
					writer.write("LRFC parameters:"+newline+"Feature file: " +basename+
						newline+"Subclass forest file: "+subname+".ser"+newline+"Number of clusters: "+String.format("%d",nc[2])+
						newline+"Cluster balance: "+String.format("%.2f",balance[1])+newline+"Number of sample points: "+String.format("%d",ppc[2])+
						newline+"Number of trees: "+String.format("%d",ntree[2])+newline+"Number of random dimensions ("+String.format("%d",numdim)+"): "+String.format("%d",mtry[2])+
						newline+"Maximum tree depth: "+String.format("%d",maxdepth[2])+newline+"Maximum leaf size: "+String.format("%d",maxleafsize[2])+
						newline+"Label purity required before the RFC: "+String.format("%.2f",splitpurity[2])+
						newline+"Label purity required after the RFC: "+String.format("%.2f",splitpurity2[2]));
					writer.close();
				}
				catch (Exception e){}
				// choose a set of random points for clustering
				l = ppc[2];
				trainingloc = new int[l*sublabels.length];
				trainingset = new double[sublabels.length*l][numdim];
				labelset = new double[sublabels.length*l];
				weights = VectorFun.add(new double[sublabels.length*l],1);
				for (int i=0; i<sublabels.length; i++){
					VectorAccess.write(labelset,VectorFun.add(new double[l],i),l*i);
					VectorAccess.write(trainingloc,VectorAccess.access(sublabels[i],Sample.sample(sublabels[i].length,l)),l*i);
				}
				r = new RankSort(VectorConv.int2double(trainingloc),labelset);
				trainingloc = VectorConv.double2int(r.getSorted());
				labelset = r.getDRank();
				count = 0;
				len = 0;
				for (int i=0; i<l*sublabels.length; i++){
					if(trainingloc[i]>=imlen2[count]){
						name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
						temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
						if (len>0){
							for (int j=0; j<len; j++){
								trainingset[i-len+j] = temp[trainingloc[i-len+j]-imlen3[count]].clone();
							}
						}
						while (trainingloc[i]>=imlen2[count]){
							count++;
						}
						len = 0;
					}
					len++;
				}
				name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
				if (len>0){
					for (int j=0; j<len; j++){
						trainingset[l*sublabels.length-len+j] = temp[trainingloc[l*sublabels.length-len+j]-imlen3[count]].clone();
					}
				}
				forest = new RandomForest(trainingset,sublabels.length,labelset,weights,categorical,dimweights,mtry[2],maxdepth[2],maxleafsize[2],splitpurity[2],0,ntree[2]);
				// Here follows the random forest clustering
				int[][] leafindices = forest.getLeafIndices(trainingset);
				RFC rfc = new RFC(leafindices,nc[2],forest.getTreeSizes(),balance[2]);
				int[] clusters = rfc.assignCluster(leafindices);
				// base a random forest off of these values
				forest = new RandomForest(trainingset,nc[2],VectorConv.int2double(clusters),weights,categorical,dimweights,mtry[2],maxdepth[2],maxleafsize[2],splitpurity2[2],0,ntree[2]);
				ReadForest.writeForest(forestser,forest);
			}
		}
	}
	
	private void applySubclass(){
		String forestser = subforestdir+File.separator+subname+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		String name = subforestdir+File.separator+subname+"_"+String.format("%d",0)+".tif";
		if(!(new File(name)).exists()||overwrite[2]){
			for (int i=0; i<numim; i++){
				String forestname = subforestdir+File.separator+subname+"_"+String.format("%d",i)+".tif";
				String votesname = subforestdir+File.separator+subname+"_votes_"+String.format("%d",i)+".tif";
				double[][] temp;
				name = featuredir+File.separator+basename+"_"+String.format("%d",i)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
				double[][] votes = VectorAccess.flip(forest.applyForest(temp));
				IJ.saveAsTiff(StackOperations.convert2Stack(votes,imdim[i][0],imdim[i][1]),votesname);
				IJ.saveAsTiff(StackOperations.convert2Stack(StackOperations.maxIndex(votes),imdim[i][0],imdim[i][1],1),forestname);
			}
		}
	}
	
	private void LRFCSetup(){
		String name;
		String forestser = lrfcforestdir+File.separator+LRFCname+".ser";
		if(!(new File(forestser)).exists()||overwrite[3]){
			// prepare an info txt file which contains the parameters used
			name = infodir+File.separator+LRFCname+"_LRFC.txt";
			try {
				File infotxt = new File(name);
				BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
				String newline = System.getProperty("line.separator");
				// now comes the ugly storage of all the parameters...
				writer.write("LRFC parameters:"+newline+"Feature file: " +basename+newline+"Number of clusters: "+String.format("%d",nc[1])+
					newline+"Cluster balance: "+String.format("%.2f",balance[1])+newline+"Number of sample points: "+String.format("%d",ppc[1])+
					newline+"Number of trees: "+String.format("%d",ntree[1])+newline+"Number of random dimensions ("+String.format("%d",numdim)+"): "+String.format("%d",mtry[1])+
					newline+"Maximum tree depth: "+String.format("%d",maxdepth[1])+newline+"Maximum leaf size: "+String.format("%d",maxleafsize[1])+
					newline+"Label purity required before the RFC: "+String.format("%.2f",splitpurity[1])+
					newline+"Label purity required after the RFC: "+String.format("%.2f",splitpurity2[1]));
				writer.close();
			}
			catch (Exception e){}
			// choose a set of random points for clustering
			int l = ppc[2];
			int[] trainingloc = new int[l*numclasses];
			int[][] labels = etl.getLabels();
			double[][] trainingset = new double[numclasses*l][];
			double[] labelset = new double[numclasses*l];
			double[] weights = VectorFun.add(new double[numclasses*l],1);
			for (int i=0; i<numclasses; i++){
				VectorAccess.write(labelset,VectorFun.add(new double[l],i),l*i);
				VectorAccess.write(trainingloc,VectorAccess.access(labels[i],Sample.sample(labels[i].length,l)),l*i);
			}
			RankSort r = new RankSort(VectorConv.int2double(trainingloc),labelset);
			trainingloc = VectorConv.double2int(r.getSorted());
			labelset = r.getDRank();
			int count = 0;
			int len = 0;
			double[][] temp;
			for (int i=0; i<l*numclasses; i++){
				if(trainingloc[i]>=imlen2[count]){
					name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
					temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
					if (len>0){
						for (int j=0; j<len; j++){
							trainingset[i-len+j] = temp[trainingloc[i-len+j]-imlen3[count]].clone();
						}
					}
					while (trainingloc[i]>=imlen2[count]){
						count++;
					}
					len = 0;
				}
				len++;
			}
			name = featuredir+File.separator+basename+"_"+String.format("%d",count)+".tif";
			temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
			if (len>0){
				for (int j=0; j<len; j++){
					trainingset[l*numclasses-len+j] = temp[trainingloc[l*numclasses-len+j]-imlen3[count]].clone();
				}
			}
			//IJ.log("Number of training dimensions: "+String.format("%d",dim));
			boolean[] categorical = new boolean[numdim];
			double[] dimweights = new double[0];
			RandomForest forest = new RandomForest(trainingset,numclasses,labelset,weights,categorical,dimweights,mtry[2],maxdepth[2],maxleafsize[2],splitpurity[2],0,ntree[2]);
			// Here follows the random forest clustering
			int[][] leafindices = forest.getLeafIndices(trainingset);
			RFC rfc = new RFC(leafindices,nc[1],forest.getTreeSizes(),balance[1]);
			int[] clusters = rfc.assignCluster(leafindices);
			// base a random forest off of these values
			forest = new RandomForest(trainingset,nc[1],VectorConv.int2double(clusters),weights,categorical,dimweights,mtry[1],maxdepth[1],maxleafsize[1],splitpurity2[1],0,ntree[1]);
			ReadForest.writeForest(forestser,forest);
		}
	}
	
	private void applyLRFC(){
		String forestser = lrfcforestdir+File.separator+LRFCname+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		String name = lrfcforestdir+File.separator+LRFCname+"_"+String.format("%d",0)+".tif";
		if(!(new File(name)).exists()||overwrite[3]){
			for (int i=0; i<numim; i++){
				String forestname = lrfcforestdir+File.separator+LRFCname+"_"+String.format("%d",i)+".tif";
				String votesname = lrfcforestdir+File.separator+LRFCname+"_votes_"+String.format("%d",i)+".tif";
				double[][] temp;
				name = featuredir+File.separator+basename+"_"+String.format("%d",i)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
				double[][] votes = VectorAccess.flip(forest.applyForest(temp));
				IJ.saveAsTiff(StackOperations.convert2Stack(votes,imdim[i][0],imdim[i][1]),votesname);
				IJ.saveAsTiff(StackOperations.convert2Stack(StackOperations.maxIndex(votes),imdim[i][0],imdim[i][1],1),forestname);
			}
		}
	}
	
	private void simpleForest(){
		String name;
		String forestser = simpleforestdir+File.separator+automatonname+".ser";
		if(!(new File(forestser)).exists()||overwrite[4]){
			// prepare an info txt file which contains the parameters used
			name = infodir+File.separator+automatonname+"_simpleforest.txt";
			try {
				File infotxt = new File(name);
				BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
				String newline = System.getProperty("line.separator");
				// now comes the ugly storage of all the parameters...
				writer.write("Simple forest parameters:"+newline+"Feature file: "+basename+newline+"RFC file: "+RFCname+newline+"LRFC file: "+LRFCname+
					newline+"Subclass file: "+subname+newline+"Spiral parameters"+
					newline+"Angular increment: "+String.format("%.2f",phi)+newline+"Patch radius: "+String.format("%.2f",patchradius)+
					newline+"Radial increment: "+String.format("%.2f",br)+newline+"Angular Exponent: "+String.format("%.2f",nphi)+newline+"Radial Exponent: "+String.format("%.2f",nrad)+
					newline+"Number of spiral arms: "+String.format("%d",arms)+newline+"Forest parameters"+
					newline+"Number of sample points: "+String.format("%d",ppc[3])+newline+"Number of trees: "+String.format("%d",ntree[3])+
					newline+"Number of random dimensions ("+String.format("%d",numdim2-nps)+"): "+String.format("%d",(mtry[3]*2)/3)+
					newline+"Maximum tree depth: "+String.format("%d",maxdepth[3])+newline+"Maximum leaf size: "+String.format("%d",maxleafsize[3])+
					newline+"Label purity: "+String.format("%.2f",splitpurity[3]));
				writer.close();			
			}
			catch (Exception e){}
			// first we'll need to read out some sublabels
			name = subforestdir+File.separator+subname+"_subclass.ser";
			int[][] sublabels = ReadForest.readIntAA(name);
			int nsc = sublabels.length;
			int l = ppc[3];
			int nd = numdim2-nps;
			int[] trainingloc = new int[l*nsc];
			double[][] trainingset = new double[nsc*l][nd];
			double[] labelset = new double[nsc*l];
			double[] weights = VectorFun.add(new double[nsc*l],1);
			for (int i=0; i<nsc; i++){
				VectorAccess.write(labelset,VectorFun.add(new double[l],i),l*i);
				VectorAccess.write(trainingloc,VectorAccess.access(sublabels[i],Sample.sample(sublabels[i].length,l)),l*i);
			}
			RankSort r = new RankSort(VectorConv.int2double(trainingloc),labelset);
			trainingloc = VectorConv.double2int(r.getSorted());
			labelset = r.getDRank();
			// preparation of the dim weights
			// we want the number of samples drawn per class to be roughly in the ratio as if we
			// would have drawn sqrt samples from the individual classes
			double[] dimweights = VectorFun.add(new double[nd],1);
			boolean[] categorical = new boolean[nd];
			for (int i=numdim; i<nd; i++){
				categorical[i] = true;
			}
			// preparation of the patch shape
			int[][] shape = NeighborhoodSample.spiralCoord(phi,patchradius,br,nphi,nrad,arms);
			int count = 0;
			int len = 0;
			int counter = 0;
			VectorAccess.write(dimweights,VectorFun.add(new double[numdim],1/Math.sqrt(numdim)),counter);
			counter+=numdim;
			for (int i=0; i<2; i++){
				VectorAccess.write(dimweights,VectorFun.add(new double[nps],1/Math.sqrt(nps)),counter);
				counter+=nps;
			}
			double[][] temp;
			int[] point = new int[1];
			String[] imagename = new String[]{featuredir+File.separator+basename+"_",rfcforestdir+File.separator+RFCname+"_",lrfcforestdir+File.separator+LRFCname+"_"};
			for (int i=0; i<l*nsc; i++){
				// if we would have to go to the next image
				if (trainingloc[i]>=imlen2[count]){
					// if there were pixels in the last image
					if (len>0){
						name = imagename[0]+String.format("%d",count)+".tif";
						temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
						for (int j=0; j<len; j++){
							VectorAccess.write(trainingset[i-len+j],temp[trainingloc[i-len+j]-imlen3[count]],0);
						}
						counter = numdim;
						for (int k=0; k<2; k++){
							name = imagename[k+1]+String.format("%d",count)+".tif";
							temp = StackOperations.stack2PixelArrays(IJ.openImage(name));
							for (int j=0; j<len; j++){
								point[0] = trainingloc[i-len+j]-imlen3[count];
								VectorAccess.write(trainingset[i-len+j],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,temp)[0],counter);
							}
							counter+=nps;
						}
					}
					while (trainingloc[i]>=imlen2[count]){
						count++;
					}
					len = 0;
				}
				len++;
			}
			if (len>0){
				name = imagename[0]+String.format("%d",count)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
				for (int j=0; j<len; j++){
					VectorAccess.write(trainingset[l*nsc-len+j],temp[trainingloc[l*nsc-len+j]-imlen3[count]],0);
				}
				counter = numdim;
				for (int k=0; k<2; k++){
					name = imagename[k+1]+String.format("%d",count)+".tif";
					temp = StackOperations.stack2PixelArrays(IJ.openImage(name));
					for (int j=0; j<len; j++){
						point[0] = trainingloc[l*nsc-len+j]-imlen3[count];
						VectorAccess.write(trainingset[l*nsc-len+j],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,temp)[0],counter);
					}
				}
			}
			// training of the actual forest
			RandomForest forest = new RandomForest(trainingset,nsc,labelset,weights,categorical,dimweights,(mtry[3]*2)/3,maxdepth[3],maxleafsize[3],splitpurity[3],0,ntree[3]);
			forestser = simpleforestdir+File.separator+automatonname+".ser";
			ReadForest.writeForest(forestser,forest);
		}
	}
	
	private void applySimpleForest(){
		String forestser = simpleforestdir+File.separator+automatonname+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		String name = simpleforestdir+File.separator+automatonname+"_"+String.format("%d",0)+".tif";
		String[] imagename = new String[]{featuredir+File.separator+basename+"_",rfcforestdir+File.separator+RFCname+"_",lrfcforestdir+File.separator+LRFCname+"_"};
		if(!(new File(name)).exists()||overwrite[4]){
			double[][] temp, temp2, temp3;
			int[] point = new int[1];
			name = subforestdir+File.separator+subname+"_subclass.ser";
			int[][] sublabels = ReadForest.readIntAA(name);
			int nsc = sublabels.length;
			int nd = numdim2-nps;
			int[][] shape = NeighborhoodSample.spiralCoord(phi,patchradius,br,nphi,nrad,arms);
			double[] point2 = new double[nd];
			for (int i=0; i<numim; i++){
				String forestname = simpleforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".tif";
				String votesname = simpleforestdir+File.separator+automatonname+"_votes_"+String.format("%d",i)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(imagename[0]+String.format("%d",i)+".tif")));
				temp2 = StackOperations.stack2PixelArrays(IJ.openImage(imagename[1]+String.format("%d",i)+".tif"));
				temp3 = StackOperations.stack2PixelArrays(IJ.openImage(imagename[2]+String.format("%d",i)+".tif"));
				double[][] votes = new double[temp.length][nsc];
				for (int j=0; j<temp.length; j++){
					point[0] = j;
					VectorAccess.write(point2,temp[j],0);
					VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape,0,nsc+1,temp2)[0],numdim);
					VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape,0,nsc+1,temp3)[0],numdim+nps);
					VectorAccess.write(votes[j],forest.applyForest(point2),0);
				}
				IJ.saveAsTiff(StackOperations.convert2Stack(VectorAccess.flip(votes),imdim[i][0],imdim[i][1]),votesname);
				IJ.saveAsTiff(StackOperations.convert2Stack(StackOperations.maxIndex(VectorAccess.flip(votes)),imdim[i][0],imdim[i][1],1),forestname);
				forestname = iterativeforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".tif";
				votesname = iterativeforestdir+File.separator+automatonname+"_votes_"+String.format("%d",i)+".tif";
				IJ.saveAsTiff(StackOperations.convert2Stack(VectorAccess.flip(votes),imdim[i][0],imdim[i][1]),votesname);
				IJ.saveAsTiff(StackOperations.convert2Stack(StackOperations.maxIndex(VectorAccess.flip(votes)),imdim[i][0],imdim[i][1],1),forestname);
			}
		}
	}
	
	private void iterativeForest(int it){
		String name;
		String forestser = iterativeforestdir+File.separator+automatonname+"_"+String.format("%d",it)+".ser";
		if(!(new File(forestser)).exists()||overwrite[5]){
			if (it==0){
				// prepare an info txt file which contains the parameters used
				name = infodir+File.separator+automatonname+"_iterativeforest.txt";
				try {
					File infotxt = new File(name);
					BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
					String newline = System.getProperty("line.separator");
					// now comes the ugly storage of all the parameters...
					writer.write("Iterative forest parameters:"+newline+"Feature file: "+basename+newline+"RFC file: "+RFCname+newline+"LRFC file: "+LRFCname+
						newline+"Subclass file: "+subname+newline+"Simple forest file: "+automatonname+newline+"Spiral parameters"+
						newline+"Angular increment: "+String.format("%.2f",phi)+newline+"Patch radius: "+String.format("%.2f",patchradius)+
						newline+"Radial increment: "+String.format("%.2f",br)+newline+"Angular Exponent: "+String.format("%.2f",nphi)+newline+"Radial Exponent: "+String.format("%.2f",nrad)+
						newline+"Number of spiral arms: "+String.format("%d",arms)+newline+"Forest parameters"+
						newline+"Number of sample points: "+String.format("%d",ppc[3])+newline+"Training Weight Rebalance: "+String.format("%.2f",trainingbalance[0])+
						newline+"Number of trees: "+String.format("%d",ntree[3])+
						newline+"Number of random dimensions ("+String.format("%d",numdim2)+"): "+String.format("%d",mtry[3])+
						newline+"Maximum tree depth: "+String.format("%d",maxdepth[3])+newline+"Maximum leaf size: "+String.format("%d",maxleafsize[3])+
						newline+"Label purity: "+String.format("%.2f",splitpurity[3]));
					writer.close();			
				}
				catch (Exception e){}
			}
			// first we'll need to read out some sublabels
			name = subforestdir+File.separator+subname+"_subclass.ser";
			int[][] sublabels = ReadForest.readIntAA(name);
			int nsc = sublabels.length;
			int l = ppc[3];
			int nd = numdim2;
			int[] trainingloc = new int[l*nsc];
			int sc = nsc/numclasses;
			int[] fractions = new int[nsc];
			double[] tempim;
			ImagePlus imp;
			String[] imagename = new String[]{featuredir+File.separator+basename+"_",rfcforestdir+File.separator+RFCname+"_",
				lrfcforestdir+File.separator+LRFCname+"_",iterativeforestdir+File.separator+automatonname+"_"};
			int count = 0;
			int len = 0;
			ArrayList<ArrayList<Integer>> correctlist = new ArrayList<ArrayList<Integer>>();
			ArrayList<ArrayList<Integer>> incorrectlist = new ArrayList<ArrayList<Integer>>();
			for (int i=0; i<nsc; i++){
				correctlist.add(new ArrayList<Integer>());
				incorrectlist.add(new ArrayList<Integer>());
			}
			int[] labels = VectorAccess.vertCat(sublabels);
			int l2 = labels.length;
			int[] labelsind = new int[l2];
			int counter = 0;
			for (int i=0; i<nsc; i++){
				VectorAccess.write(labelsind,VectorFun.add(new int[sublabels[i].length],i),counter);
				counter+=sublabels[i].length;
			}
			RankSort r = new RankSort(VectorConv.int2double(labels),labelsind);
			labels = VectorConv.double2int(r.getSorted());
			labelsind = r.getRank();
			for (int i=0; i<l2; i++){
				// if we would have to go to the next image
				if (labels[i]>=imlen2[count]){
					// if there were pixels in the last image
					if (len>0){
						name = imagename[3]+String.format("%d",count)+".tif";
						imp = IJ.openImage(name);
						tempim = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
						for (int j=0; j<len; j++){
							int a = (int)(tempim[labels[i-len+j]-imlen3[count]]/sc);
							int b = labelsind[i-len+j];
							if (a==(b/sc)){
								correctlist.get(b).add(labels[i-len+j]);
							}
							else {
								incorrectlist.get(b).add(labels[i-len+j]);
							}
						}
					}
					while (labels[i]>=imlen2[count]){
						count++;
					}
					len = 0;
				}
				len++;
			}
			if (len>0){
				name = imagename[3]+String.format("%d",count)+".tif";
				imp = IJ.openImage(name);
				tempim = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
				for (int j=0; j<len; j++){
					int a = (int)(tempim[labels[l2-len+j]-imlen3[count]]/sc);
					int b = labelsind[l2-len+j];
					if (a==(b/sc)){
						correctlist.get(b).add(labels[l2-len+j]);
					}
					else {
						incorrectlist.get(b).add(labels[l2-len+j]);
					}
				}
			}
			
			int[][] correct = new int[nsc][];
			int[][] incorrect = new int[nsc][];
			for (int i=0; i<nsc; i++){
				correct[i] = new int[correctlist.get(i).size()];
				incorrect[i] = new int[incorrectlist.get(i).size()];
				for (int j=0; j<correctlist.get(i).size(); j++){
					correct[i][j] = correctlist.get(i).get(j);
				}
				for (int j=0; j<incorrectlist.get(i).size(); j++){
					incorrect[i][j] = incorrectlist.get(i).get(j);
				}
				fractions[i] = (int)((Math.pow(correct[i].length,trainingbalance[0]))/((Math.pow(correct[i].length,trainingbalance[0]))+Math.pow(incorrect[i].length,trainingbalance[0]))*l);
			}
			
			double[][] trainingset = new double[nsc*l][nd];
			double[] labelset = new double[nsc*l];
			double[] weights = VectorFun.add(new double[nsc*l],1);
			for (int i=0; i<nsc; i++){
				VectorAccess.write(labelset,VectorFun.add(new double[l],i),l*i);
				VectorAccess.write(trainingloc,VectorAccess.access(correct[i],Sample.sample(correct[i].length,fractions[i])),l*i);
				VectorAccess.write(trainingloc,VectorAccess.access(incorrect[i],Sample.sample(incorrect[i].length,l-fractions[i])),l*i+fractions[i]);
			}
			r = new RankSort(VectorConv.int2double(trainingloc),labelset);
			trainingloc = VectorConv.double2int(r.getSorted());
			labelset = r.getDRank();
			// preparation of the dim weights
			// we want the number of samples drawn per class to be roughly in the ratio as if we
			// would have drawn sqrt samples from the individual classes
			double[] dimweights = VectorFun.add(new double[nd],1);
			boolean[] categorical = new boolean[nd];
			for (int i=numdim; i<nd; i++){
				categorical[i] = true;
			}
			// preparation of the patch shape
			int[][] shape = NeighborhoodSample.spiralCoord(phi,patchradius,br,nphi,nrad,arms);
			counter = 0;
			VectorAccess.write(dimweights,VectorFun.add(new double[numdim],1/Math.sqrt(numdim)),counter);
			counter+=numdim;
			for (int i=0; i<3; i++){
				VectorAccess.write(dimweights,VectorFun.add(new double[nps],1/Math.sqrt(nps)),counter);
				counter+=nps;
			}
			double[][] temp;
			len = 0;
			count = 0;
			int[] point = new int[1];
			for (int i=0; i<l*nsc; i++){
				// if we would have to go to the next image
				if (trainingloc[i]>=imlen2[count]){
					// if there were pixels in the last image
					if (len>0){
						name = imagename[0]+String.format("%d",count)+".tif";
						temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
						for (int j=0; j<len; j++){
							VectorAccess.write(trainingset[i-len+j],temp[trainingloc[i-len+j]-imlen3[count]],0);
						}
						counter = numdim;
						for (int k=0; k<3; k++){
							name = imagename[k+1]+String.format("%d",count)+".tif";
							imp = IJ.openImage(name);
							tempim = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
							for (int j=0; j<len; j++){
								point[0] = trainingloc[i-len+j]-imlen3[count];
								VectorAccess.write(trainingset[i-len+j],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,tempim)[0],counter);
							}
							counter+=nps;
						}
					}
					while (trainingloc[i]>=imlen2[count]){
						count++;
					}
					len = 0;
				}
				len++;
			}
			if (len>0){
				name = imagename[0]+String.format("%d",count)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
				for (int j=0; j<len; j++){
					VectorAccess.write(trainingset[l*nsc-len+j],temp[trainingloc[l*nsc-len+j]-imlen3[count]],0);
				}
				counter = numdim;
				for (int k=0; k<3; k++){
					name = imagename[k+1]+String.format("%d",count)+".tif";
					imp = IJ.openImage(name);
					tempim = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
					for (int j=0; j<len; j++){
						point[0] = trainingloc[l*nsc-len+j]-imlen3[count];
						VectorAccess.write(trainingset[l*nsc-len+j],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,tempim)[0],counter);
					}
					counter+=nps;
				}
			}
			// training of the actual forest
			RandomForest forest = new RandomForest(trainingset,nsc,labelset,weights,categorical,dimweights,mtry[3],maxdepth[3],maxleafsize[3],splitpurity[3],0,ntree[3]);
			forestser = iterativeforestdir+File.separator+automatonname+"_"+String.format("%d",it)+".ser";
			ReadForest.writeForest(forestser,forest);
		}
	}
	
	private void applyIterativeForest(int it){
		String forestser = iterativeforestdir+File.separator+automatonname+"_"+String.format("%d",it)+".ser";
		RandomForest forest = ReadForest.readForest(forestser);
		String name = iterativeforestdir+File.separator+automatonname+"_"+String.format("%d",0)+".tif";
		String[] imagename = new String[]{featuredir+File.separator+basename+"_",rfcforestdir+File.separator+RFCname+"_",lrfcforestdir+File.separator+LRFCname+"_",
		iterativeforestdir+File.separator+automatonname+"_"};
		double[][] temp;
		double[] temp2, temp3, temp4;
		int[] point = new int[1];
		name = subforestdir+File.separator+subname+"_subclass.ser";
		int[][] sublabels = ReadForest.readIntAA(name);
		int nsc = sublabels.length;
		int nd = numdim2;
		int[][] shape = NeighborhoodSample.spiralCoord(phi,patchradius,br,nphi,nrad,arms);
		double[] point2 = new double[nd];
		ImagePlus imp;
		for (int i=0; i<numim; i++){
			String forestname = iterativeforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".tif";
			String votesname = iterativeforestdir+File.separator+automatonname+"_votes_"+String.format("%d",i)+".tif";
			temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(imagename[0]+String.format("%d",i)+".tif")));
			name = imagename[1]+String.format("%d",i)+".tif";
			imp = IJ.openImage(name);
			temp2 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
			name = imagename[2]+String.format("%d",i)+".tif";
			imp = IJ.openImage(name);
			temp3 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
			name = imagename[3]+String.format("%d",i)+".tif";
			imp = IJ.openImage(name);
			temp4 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
			double[][] votes = new double[temp.length][nsc];
			for (int j=0; j<temp.length; j++){
				point[0] = j;
				VectorAccess.write(point2,temp[j],0);
				VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape,0,nsc+1,temp2)[0],numdim);
				VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape,0,nsc+1,temp3)[0],numdim+nps);
				VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape,0,nsc+1,temp4)[0],numdim+2*nps);
				VectorAccess.write(votes[j],forest.applyForest(point2),0);
			}
			imp = IJ.openImage(votesname);
			votes = VectorAccess.flip(votes);
			for (int j=0; j<nsc; j++){
				imp.getStack().addSlice("None",new FloatProcessor(imdim[i][0],imdim[i][1],votes[j]));
			}
			IJ.saveAsTiff(imp,votesname);
			imp = IJ.openImage(forestname);
			if (it==0){
				ImageStack stack = new ImageStack(imdim[i][0],imdim[i][1]);
				stack.addSlice(imp.getStack().getProcessor(1));
				stack.addSlice("None",new FloatProcessor(imp.getWidth(),imp.getHeight(),StackOperations.maxIndex(votes)));
				imp = new ImagePlus("Iterative Forest",stack);
			}
			else {
				imp.getStack().addSlice("None",new FloatProcessor(imp.getWidth(),imp.getHeight(),StackOperations.maxIndex(votes)));
			}
			IJ.saveAsTiff(imp,forestname);
		}
	}
	
	private void automatonForest(){
		String name;
		String forestser = automatonforestdir+File.separator+automatonname+"_"+String.format("%d",0)+".ser";
		if(!(new File(forestser)).exists()||overwrite[6]){
			// prepare an info txt file which contains the parameters used
			name = infodir+File.separator+automatonname+"_automatonforest.txt";
			try {
				File infotxt = new File(name);
				BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
				String newline = System.getProperty("line.separator");
				// now comes the ugly storage of all the parameters...
				writer.write("Automaton forest parameters:"+newline+"Feature file: "+basename+newline+"RFC file: "+RFCname+newline+"LRFC file: "+LRFCname+
					newline+"Subclass file: "+subname+newline+"Simple forest file: "+automatonname+newline+"Spiral parameters"+
					newline+"Angular increment: "+String.format("%.2f",phi)+newline+"Patch radius: "+String.format("%.2f",patchradius)+
					newline+"Radial increment: "+String.format("%.2f",br)+newline+"Angular Exponent: "+String.format("%.2f",nphi)+newline+"Radial Exponent: "+String.format("%.2f",nrad)+
					newline+"Number of spiral arms: "+String.format("%d",arms)+newline+"Forest parameters"+
					newline+"Number of sample points: "+String.format("%d",ppc[4])+newline+"Training Weight Rebalance: "+String.format("%.2f",trainingbalance[1])+
					newline+"Number of trees: "+String.format("%d",ntree[4])+newline+"Number of random dimensions ("+String.format("%d",numdim2)+"): "+String.format("%d",mtry[4])+
					newline+"Maximum tree depth: "+String.format("%d",maxdepth[3])+newline+"Maximum leaf size: "+String.format("%d",maxleafsize[3])+
					newline+"Label purity: "+String.format("%.2f",splitpurity[4]));
				writer.close();			
			}
			catch (Exception e){}
			// first we'll need to read out some sublabels
			name = subforestdir+File.separator+subname+"_subclass.ser";
			int[][] sublabels = ReadForest.readIntAA(name);
			// the number of classes we have after subclassing
			int nsc = sublabels.length;
			String[] imagename = new String[]{featuredir+File.separator+basename+"_",rfcforestdir+File.separator+RFCname+"_",
				lrfcforestdir+File.separator+LRFCname+"_",iterativeforestdir+File.separator+automatonname+"_"};
			ImagePlus imp = IJ.openImage(imagename[3]+String.format("%d",0)+".tif");
			// number of iterations for the iterative forest
			int nsl = imp.getNSlices()-1;
			// number of labels which transition in the iterative images from any class to a correct super class
			int[][] numlabels = new int[nsc][nsc];
			// collection of all the labelled pixels with their subclass value. Will be sorted according to pixel location
			int[] labels = VectorAccess.vertCat(sublabels);
			int l2 = labels.length;
			// value of the subclass. We keep this for when the pixel locations are reordered.
			int[] labelsind = new int[l2];
			// number sub classes in a single class
			int sc = nsc/numclasses;
			int nd = numdim2;
			int[][] shape = NeighborhoodSample.spiralCoord(phi,patchradius,br,nphi,nrad,arms);
			int counter = 0;
			for (int i=0; i<nsc; i++){
				VectorAccess.write(labelsind,VectorFun.add(new int[sublabels[i].length],i),counter);
				counter+=sublabels[i].length;
			}
			RankSort r = new RankSort(VectorConv.int2double(labels),labelsind);
			labels = VectorConv.double2int(r.getSorted());
			labelsind = r.getRank();
			int count = 0;
			int len =0;
			double[][] tempim;
			// count the number of all possible training pixels
			for (int i=0; i<l2; i++){
				// if we would have to go to the next image
				if (labels[i]>=imlen2[count]){
					// if there were pixels in the last image
					if (len>0){
						name = imagename[3]+String.format("%d",count)+".tif";
						imp = IJ.openImage(name);
						tempim = StackOperations.stack2PixelArrays(imp);
						for (int j=0; j<nsl; j++){
							for (int k=0; k<len; k++){
								// the current iteration pixel value in this image
								int a = (int)(tempim[j][labels[i-len+k]-imlen3[count]]);
								// the target iteration pixel value in this image
								int b = (int)(tempim[j+1][labels[i-len+k]-imlen3[count]]);
								// the correct subclass pixel value
								int c = labelsind[i-len+k];
								// we remember all cases where the target pixel and subclass pixel have the same super class
								if ((b/sc)==(c/sc)){
									numlabels[a][b]++;
								}
							}
						}
					}
					while (labels[i]>=imlen2[count]){
						count++;
					}
					len = 0;
				}
				len++;
			}
			if (len>0){
				name = imagename[3]+String.format("%d",count)+".tif";
				imp = IJ.openImage(name);
				tempim = StackOperations.stack2PixelArrays(imp);
				for (int j=0; j<nsl; j++){
					for (int k=0; k<len; k++){
						// the current iteration pixel value in this image
						int a = (int)(tempim[j][labels[l2-len+k]-imlen3[count]]);
						// the target iteration pixel value in this image
						int b = (int)(tempim[j+1][labels[l2-len+k]-imlen3[count]]);
						// the correct subclass pixel value
						int c = labelsind[l2-len+k];
						// we remember all cases where the target pixel and subclass pixel have the same super class
						if ((b/sc)==(c/sc)){
							numlabels[a][b]++;
						}
					}
				}
			}
			int l = ppc[4];
			// training locations according to intial and to target value
			int[][][] trainingloc = new int[nsc][nsc][];
			// collection of labels for each initial value
			double[][] labelset = new double[nsc][];
			double[][][] trainingset = new double[nsc][][];
			int[] point = new int[1];
			double[] temp2, temp3;
			double[][] temp;
			// get the location for each training location according to the above counting
			int d;
			for (int i=0; i<nsc; i++){
				d = 0;
				// traingsample lengths
				double[] tsl = new double[nsc];
				double sum = 0;
				for (int j=0; j<nsc; j++){
					sum+=numlabels[i][j];
					tsl[j] = numlabels[i][j];
				}
				VectorFun.multi(tsl,1/sum);
				VectorFun.powi(tsl,trainingbalance[1]);
				VectorFun.multi(tsl,1/VectorFun.sum(tsl));
				for (int j=0; j<nsc; j++){
					int t = (int)(l*tsl[j]);
					if (numlabels[i][j]>0&&t==0){
						t = 1;
					}
					trainingloc[i][j] = Sample.sample(numlabels[i][j],t);
					r = new RankSort(VectorConv.int2double(trainingloc[i][j]));
					trainingloc[i][j] = VectorConv.double2int(r.getSorted());
					d+=t;
				}
				labelset[i] = new double[d];
				trainingset[i] = new double[d][nd];
			}
			// current value locations according to the above location extraction
			int[][] numlab2 = new int[nsc][nsc];
			count = 0;
			len = 0;
			// current value locations according to the above location extraction
			// number of explicity observed training locations
			int[][] labelcount = new int[nsc][nsc];
			int[] labelcounter = new int[nsc];
			// go through all labels a second time to get the right training pixels
			for (int i=0; i<l2; i++){
				// if we would have to go to the next image
				if (labels[i]>=imlen2[count]){
					// if there were pixels in the last image
					if (len>0){
						name = imagename[3]+String.format("%d",count)+".tif";
						imp = IJ.openImage(name);
						tempim = StackOperations.stack2PixelArrays(imp);
						name = imagename[0]+String.format("%d",count)+".tif";
						temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
						name = imagename[1]+String.format("%d",count)+".tif";
						imp = IJ.openImage(name);
						temp2 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
						name = imagename[2]+String.format("%d",count)+".tif";
						imp = IJ.openImage(name);
						temp3 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
						// go through all iterative slices
						for (int j=0; j<nsl; j++){
							for (int k=0; k<len; k++){
								// the current iteration pixel value in this image
								int a = (int)(tempim[j][labels[i-len+k]-imlen3[count]]);
								// the target iteration pixel value in this image
								int b = (int)(tempim[j+1][labels[i-len+k]-imlen3[count]]);
								// the correct subclass pixel value
								int c = labelsind[i-len+k];
								// we remember all cases where the target pixel and subclass pixel have the same super class
								if ((b/sc)==(c/sc)){
									if (numlab2[a][b]==trainingloc[a][b][labelcount[a][b]]){
										while (numlab2[a][b]==trainingloc[a][b][labelcount[a][b]]){
											point[0] = labels[i-len+k]-imlen3[count];
											VectorAccess.write(trainingset[a][labelcounter[a]],temp[point[0]],0);
											VectorAccess.write(trainingset[a][labelcounter[a]],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,temp2)[0],numdim);
											VectorAccess.write(trainingset[a][labelcounter[a]],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,temp3)[0],numdim+nps);
											VectorAccess.write(trainingset[a][labelcounter[a]],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,tempim[j])[0],numdim+2*nps);
											labelset[a][labelcounter[a]] = tempim[j+1][point[0]];
											labelcounter[a]++;
											if(labelcount[a][b]<trainingloc[a][b].length-1){
												labelcount[a][b]++;
											}
											else {
												break;
											}
										}
									}
									numlab2[a][b]++;
								}
							}
						}
					}
					while (labels[i]>=imlen2[count]){
						count++;
					}
					len = 0;
				}
				len++;
			}
			if (len>0){
				name = imagename[3]+String.format("%d",count)+".tif";
				imp = IJ.openImage(name);
				tempim = StackOperations.stack2PixelArrays(imp);
				name = imagename[0]+String.format("%d",count)+".tif";
				temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(name)));
				name = imagename[1]+String.format("%d",count)+".tif";
				imp = IJ.openImage(name);
				temp2 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
				name = imagename[2]+String.format("%d",count)+".tif";
				imp = IJ.openImage(name);
				temp3 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
				for (int j=0; j<nsl; j++){
					for (int k=0; k<len; k++){
						int a = (int)(tempim[j][labels[l2-len+k]-imlen3[count]]);
						int b = (int)(tempim[j+1][labels[l2-len+k]-imlen3[count]]);
						int c = labelsind[l2-len+k];
						if ((b/sc)==(c/sc)){
							if (numlab2[a][b]==trainingloc[a][b][labelcount[a][b]]){
								while (numlab2[a][b]==trainingloc[a][b][labelcount[a][b]]){
									point[0] = labels[l2-len+k]-imlen3[count];
									VectorAccess.write(trainingset[a][labelcounter[a]],temp[point[0]],0);
									VectorAccess.write(trainingset[a][labelcounter[a]],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,temp2)[0],numdim);
									VectorAccess.write(trainingset[a][labelcounter[a]],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,temp3)[0],numdim+nps);
									VectorAccess.write(trainingset[a][labelcounter[a]],NeighborhoodSample.sample2d(point,imdim[count][0],imdim[count][1],shape,0,nsc+1,tempim[j])[0],numdim+2*nps);
									labelset[a][labelcounter[a]] = tempim[j+1][point[0]];
									labelcounter[a]++;
									if(labelcount[a][b]<trainingloc[a][b].length-1){
										labelcount[a][b]++;
									}
									else {
										break;
									}
								}
							}				
							numlab2[a][b]++;
						}
					}
				}
			}
			
			double[] weights = VectorFun.add(new double[nsc*l],1);
			double[] dimweights = VectorFun.add(new double[nd],1);
			boolean[] categorical = new boolean[nd];
			for (int i=numdim; i<nd; i++){
				categorical[i] = true;
			}
			// preparation of the patch shape
			counter = 0;
			VectorAccess.write(dimweights,VectorFun.add(new double[numdim],1/Math.sqrt(numdim)),counter);
			counter+=numdim;
			for (int i=0; i<3; i++){
				VectorAccess.write(dimweights,VectorFun.add(new double[nps],1/Math.sqrt(nps)),counter);
				counter+=nps;
			}
			// now we actually train all the individual forests
			for (int i=0; i<nsc; i++){
				RandomForest forest = new RandomForest(trainingset[i],nsc,labelset[i],weights,categorical,dimweights,mtry[4],maxdepth[3],maxleafsize[4],splitpurity[4],0,ntree[4]);
				forestser = automatonforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".ser";
				ReadForest.writeForest(forestser,forest);
			}				
		}
	}
	
	private void applyAutomatonForest(){
		String name = automatonforestdir+File.separator+automatonname+"_"+String.format("%d",0)+".tif";
		String[] imagename = new String[]{featuredir+File.separator+basename+"_",rfcforestdir+File.separator+RFCname+"_",lrfcforestdir+File.separator+LRFCname+"_",
			automatonforestdir+File.separator+automatonname+"_"};
		double[][] temp;
		double[] temp2, temp3, temp4;
		int[] point = new int[1];
		int a;
		name = subforestdir+File.separator+subname+"_subclass.ser";
		int[][] sublabels = ReadForest.readIntAA(name);
		int nsc = sublabels.length;
		int nd = numdim2;
		int[][] shape = NeighborhoodSample.spiralCoord(phi,patchradius,br,nphi,nrad,arms);
		double[] point2 = new double[nd];
		ImagePlus imp;
		ArrayList<RandomForest> forests = new ArrayList<RandomForest>();
		for (int i=0; i<nsc; i++){
			String forestser = automatonforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".ser";
			forests.add(ReadForest.readForest(forestser));
		}
		for (int i=0; i<numim; i++){
			String forestname = automatonforestdir+File.separator+automatonname+"_"+String.format("%d",i)+".tif";
			String votesname = automatonforestdir+File.separator+automatonname+"_votes_"+String.format("%d",i)+".tif";
			temp = VectorAccess.flip(StackOperations.stack2PixelArrays(IJ.openImage(imagename[0]+String.format("%d",i)+".tif")));
			name = imagename[1]+String.format("%d",i)+".tif";
			imp = IJ.openImage(name);
			temp2 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
			name = imagename[2]+String.format("%d",i)+".tif";
			imp = IJ.openImage(name);
			temp3 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
			name = imagename[3]+String.format("%d",i)+".tif";
			imp = IJ.openImage(name);
			temp4 = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(imp.getNSlices()).convertToFloat().getPixels()));
			double[][] votes = new double[temp.length][nsc];
			for (int j=0; j<temp.length; j++){
				point[0] = j;
				a = (int)temp4[point[0]];
				VectorAccess.write(point2,temp[j],0);
				VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape,0,nsc+1,temp2)[0],numdim);
				VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape,0,nsc+1,temp3)[0],numdim+nps);
				VectorAccess.write(point2,NeighborhoodSample.sample2d(point,imdim[i][0],imdim[i][1],shape,0,nsc+1,temp4)[0],numdim+2*nps);
				VectorAccess.write(votes[j],forests.get(a).applyForest(point2),0);
			}
			imp = IJ.openImage(votesname);
			votes = VectorAccess.flip(votes);
			for (int j=0; j<nsc; j++){
				imp.getStack().addSlice("None",new FloatProcessor(imdim[i][0],imdim[i][1],votes[j]));
			}
			IJ.saveAsTiff(imp,votesname);
			imp = IJ.openImage(forestname);
			if (imp.getNSlices()==1){
				ImageStack stack = new ImageStack(imdim[i][0],imdim[i][1]);
				stack.addSlice(imp.getStack().getProcessor(1));
				stack.addSlice("None",new FloatProcessor(imp.getWidth(),imp.getHeight(),StackOperations.maxIndex(votes)));
				imp = new ImagePlus("Iterative Forest",stack);
			}
			else {
				imp.getStack().addSlice("None",new FloatProcessor(imp.getWidth(),imp.getHeight(),StackOperations.maxIndex(votes)));
			}
			IJ.saveAsTiff(imp,forestname);
		}
	}
}			
			
			
			
			
			
			