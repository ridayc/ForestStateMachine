package flib.algorithms.automata;

import java.util.ArrayList;
//import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.splitfunctions.GiniClusterSplit;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.VectorAccess;
import flib.math.BV;
import flib.math.RankSort;
import flib.math.random.Shuffle;

public class SimpleLearner {
	private RandomForest current;
	private RandomForest automat;
	private int numclasses, w, h, npl, npa, ntp;
	private int ntreel,ntreer;
	private double[] splitpurityl,splitpurityr;
	private double[] labelim, labels, weights, trainlab, parametersl, parametersr, dimweights;
	private double[][] trainingset;
	private boolean[] categorical;
	private double rad;
	private ImageStack levels, iterations;
	private int[] count, count2, allind, transmat, transcount, trainingloc;
	private int[][] shape, labelind, transind;
	
	public SimpleLearner(final double[] im, final double[] labelim, final double[] traininglocim, int w, int h, double rad, int numit, int npl, int npa, 
	final double[] rf_paraml, double splitpurityl, int ntreel, final double[] rf_paramr, double splitpurityr, int ntreer, int it){
		// variables:
		// im: initial starting labels
		// labelim: final target labels
		// trainingloc: locations in the image which we can choose training samples from
		// we suppose the image is binary
		// w: width of the image
		// h: height of the image
		// r: neighborhood patch radius
		// numit: number of training levels to go through
		// npl: number of training points at some given level
		// npa: number of training points for the automaton
		
		// copy all variables that need copying
		this.labelim = labelim.clone();
		this.w = w;
		this.h = h;
		this.npl = npl;
		this.npa = npa;
		this.rad = rad;
		// find the number of classes is the number of classes in the label image
		// the +1 is for because we start counting at zero
		this.numclasses = (int)VectorFun.max(this.labelim)[0]+1;
		this.parametersl = rf_paraml.clone();
		this.splitpurityl = VectorFun.add(new double[numclasses],splitpurityl);
		this.ntreel = ntreel;
		this.parametersr = rf_paramr.clone();
		this.splitpurityr = VectorFun.add(new double[numclasses],splitpurityr);
		this.ntreer = ntreer;
		this.trainingloc = VectorAccess.subset(BV.gt(traininglocim,0));
		this.trainlab = VectorAccess.access(this.labelim,this.trainingloc);
		this.count = new int[this.numclasses];
		this.transmat = new int[this.numclasses*this.numclasses];
		// prepare the image stack for the different levels
		this.levels = new ImageStack(this.w,this.h);
		this.levels.addSlice("None", new FloatProcessor(this.w,this.h,im));
		this.iterations = new ImageStack(this.w,this.h);
		this.iterations.addSlice("None", new FloatProcessor(this.w,this.h,im));
		this.shape = NeighborhoodSample.circleCoord(this.rad);
		this.labelind = VectorAccess.labels2Indices(VectorConv.double2int(this.trainlab),this.numclasses);
		// total number of training pixels
		this.ntp = 0;
		for (int i=0; i<this.numclasses; i++){
			if (this.npl<this.labelind[i].length){
				this.ntp+=this.npl;
				this.count[i] = this.npl;
				
			}
			else {
				this.ntp+=this.labelind[i].length;
				this.count[i] = this.labelind[i].length;
			}
		}
		this.count2 = new int[this.numclasses];
		VectorAccess.write(count2,VectorAccess.access(VectorFun.cumsum(this.count),0,this.numclasses-1),1);
		this.labels = new double[this.ntp];
		for (int i=0; i<this.numclasses; i++){
			VectorAccess.write(this.labels,VectorFun.add(new double[this.count[i]],i),this.count2[i]);
		}
		this.weights = VectorFun.add(new double[this.ntp],1);
		this.categorical = new boolean[shape.length];
		this.dimweights = VectorFun.add(new double[shape.length],1);
		for (int i=0; i<this.categorical.length; i++){
			this.categorical[i] = true;
		}
		this.allind = new int[im.length];
		for (int i=0; i<this.allind.length; i++){
			this.allind[i] = i;
		}
		for (int i=0; i<numit; i++){
			this.iterate();
		}
		this.generateAutomaton();
		for (int i=0; i<it; i++){
			this.applyAutomaton();
		}
	}
	
	public SimpleLearner(final double[] im, final double[] labelim, int w, int h, double rad, int numit, int npl, int npa, 
	final double[] parametersl, double splitpurityl, int ntreel, final double[] parametersr, double splitpurityr, int ntreer, int it){
		this(im,labelim, VectorFun.add(new double[im.length],1),w,h,rad,numit,npl,npa,parametersl,splitpurityl,ntreel,parametersr,splitpurityr,ntreer,it);
	}
	
	public void iterate(){
		int n = levels.getSize();
		int[] ind = new int[this.ntp];
		int[] temp;
		for (int i=0; i<this.numclasses; i++){
			if (this.count[i]>0){
				temp = Shuffle.randPerm(this.labelind[i].length);
				// changed to adapt for the preset training locations
				VectorAccess.write(ind,VectorAccess.access(this.trainingloc,VectorAccess.access(this.labelind[i],VectorAccess.access(temp,0,this.count[i]))),count2[i]);
			}
		}
		double[] im = VectorConv.float2double((float[])(this.levels.getProcessor(n).convertToFloat().getPixels()));
		this.trainingset = NeighborhoodSample.sample2d(ind,this.w,this.h,this.shape,0,this.numclasses+1,im);
		// train the forest at this level
		this.current = new RandomForest(this.trainingset,this.labels,this.weights,this.categorical,this.dimweights,this.parametersl,this.splitpurityl,new GiniClusterSplit(),this.ntreel);
		// get the resultant classification image
		this.levels.addSlice("None", new FloatProcessor(w,h,VectorFun.maxind(this.current.applyForest(NeighborhoodSample.sample2d(this.allind,this.w,this.h,this.shape,0,this.numclasses+1,im)))));
	}
	
	public void generateAutomaton(){
		int n = this.levels.getSize();
		double[] im1;
		double[] im2;
		double[] temp1;
		double[] temp2;
		int a;
		for (int i=1; i<n; i++){
			im1 =  VectorConv.float2double((float[])(this.levels.getProcessor(i).convertToFloat().getPixels()));
			im2 =  VectorConv.float2double((float[])(this.levels.getProcessor(i+1).convertToFloat().getPixels()));
			temp1 = VectorAccess.access(im1,this.trainingloc);
			temp2 = VectorAccess.access(im2,this.trainingloc);
			for (int j=0; j<temp1.length; j++){
				this.transmat[(int)(temp1[j]*this.numclasses+temp2[j])]++;
			}
		}
		this.transind = new int[this.numclasses*this.numclasses][];
		for (int i=0; i<this.transind.length; i++){
			this.transind[i] = new int[this.transmat[i]];
		}
		this.transcount = new int[this.numclasses*this.numclasses];
		for (int i=1; i<n; i++){
			im1 =  VectorConv.float2double((float[])(this.levels.getProcessor(i).convertToFloat().getPixels()));
			im2 =  VectorConv.float2double((float[])(this.levels.getProcessor(i+1).convertToFloat().getPixels()));
			temp1 = VectorAccess.access(im1,this.trainingloc);
			temp2 = VectorAccess.access(im2,this.trainingloc);
			for (int j=0; j<temp1.length; j++){
				a = (int)(temp1[j]*this.numclasses+temp2[j]);
				this.transind[a][this.transcount[a]] = j+(i-1)*temp1.length;
				this.transcount[a]++;
			}
		}
		int[] tc1 = new int[this.transmat.length];
		this.ntp = 0;
		for (int i=0; i<this.transmat.length; i++){
			if (this.npa<this.transmat[i]){
				this.ntp+=this.npa;
				tc1[i] = this.npa;
				
			}
			else {
				this.ntp+=this.transmat[i];
				tc1[i] = this.transmat[i];
			}
		}
		int[] tc2 = new int[this.transmat.length];
		VectorAccess.write(tc2,VectorAccess.access(VectorFun.cumsum(tc1),0,this.transmat.length-1),1);
		int[] ind = new int[this.ntp];
		int[] temp;
		for (int i=0; i<this.transmat.length; i++){
			if (tc1[i]>0){
				temp = Shuffle.randPerm(this.transmat[i]);
				VectorAccess.write(ind,VectorAccess.access(this.transind[i],VectorAccess.access(temp,0,tc1[i])),tc2[i]);
			}
		}
		this.labels = new double[this.ntp];
		for (int i=0; i<this.transmat.length; i++){
			VectorAccess.write(this.labels,VectorFun.add(new double[tc1[i]],i%this.numclasses),tc2[i]);
		}
		this.weights = VectorFun.add(new double[this.ntp],1);
		this.categorical = new boolean[shape.length];
		for (int i=0; i<this.categorical.length; i++){
			this.categorical[i] = true;
		}
		RankSort rank = new RankSort(VectorConv.int2double(ind),this.labels);
		ind = VectorConv.double2int(rank.getSorted());
		this.labels = rank.getDRank();
		this.trainingset = new double[this.ntp][this.shape.length];
		//this.trainingset = new double[this.ntp][];
		int len = 0;
		int[] point = new int[1];
		int counter = this.trainingloc.length;
		int counter2 = 0;
		for (int i=0; i<this.ntp; i++){
			if (ind[i]>=counter){
				im1 =  VectorConv.float2double((float[])(this.levels.getProcessor((int)(counter2/(this.trainingloc.length)+1)).convertToFloat().getPixels()));
				for (int j=0; j<len; j++){
					point[0] = this.trainingloc[ind[i-len+j]-counter2];
					trainingset[i-len+j] = NeighborhoodSample.sample2d(point,this.w,this.h,shape,0,this.numclasses+1,im1)[0];
				}
				counter = (int)(ind[i]/(this.trainingloc.length)+1)*this.trainingloc.length;
				counter2 = counter-this.trainingloc.length;
				len = 0;
			}
			len++;
		}
		im1 =  VectorConv.float2double((float[])(this.levels.getProcessor((int)(counter2/(this.trainingloc.length)+1)).convertToFloat().getPixels()));
		for (int j=0; j<len; j++){
			point[0] = this.trainingloc[ind[ind.length-len+j]-counter2];
			trainingset[ind.length-len+j] = NeighborhoodSample.sample2d(point,this.w,this.h,shape,0,this.numclasses+1,im1)[0];
		}
		this.automat = this.current = new RandomForest(this.trainingset,this.labels,this.weights,this.categorical,this.dimweights,this.parametersr,this.splitpurityr,new GiniClusterSplit(),this.ntreer);
	}
	
	public void applyAutomaton(){
		int n = this.iterations.getSize();
		double[] im = VectorConv.float2double((float[])(this.iterations.getProcessor(n).convertToFloat().getPixels()));
		this.iterations.addSlice("None", new FloatProcessor(w,h,VectorFun.maxind(this.automat.applyForest(NeighborhoodSample.sample2d(this.allind,this.w,this.h,this.shape,0,this.numclasses+1,im)))));
	}
	
	public ImagePlus getLevelStack(){
		return (new ImagePlus("Levels",this.levels));
	}
	
	public ImagePlus getIterationStack(){
		return (new ImagePlus("Iterations",this.iterations));
	}	
}