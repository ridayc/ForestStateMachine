package flib.algorithms.randomforest;

import java.util.ArrayList;
import java.util.Random;
import java.util.Arrays;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.lang.Runtime;
import java.lang.Math;
import java.lang.reflect.Constructor;
import flib.math.SortPair2;
import flib.math.VectorFun;
import flib.math.RankSort;
import flib.math.random.Shuffle;
import flib.math.random.Shuffler;
import flib.math.random.Sample;
import flib.algorithms.randomforest.DecisionTree;
import flib.algorithms.randomforest.splitfunctions.SplitFunction;

/* An implemenation of the Random Forest algorithm based on Breimann's random forest algorithm (and exception is the handling of categorical dimensions)
Meaning of the class variables:
trainingset: the training points used to generate the random forest. First dimension are the point indices, second dimension the dimension values of each point
labels: class which the training points belong to
weights: weight of each individual training point
forest: an ArrayList contain each decision tree for the classification
mtry: number of dimensions to look at for a splitting dimension in each node
ntree: number of decision trees in the forest
maxdepth: maximal depth to which a tree can be grown
maxleafsize: minmal number of points in a leaf when training a decision tree
splitpurity: if the maximum percentage of any label in the total weight of the trainingset in a node exceed this value, it will be regarded as a leaf
importance: a matrix to store the classification error of each tree based on dimension. Used to calculate the variable importance for each dimension
oob: the first column vector contains the total weight of the out of bag samples in the corresponding tree. The second column vector contains the total weight of correctly classified oob samples
oobe: weighted sum of votes for each class over all trees for each individual point
categorical: indicates if a dimension is categorical or not
dim: number of dimension each point has
n: number of points used to train the tree
e_1: a constant to approximate the number of samples to be out of bag when training a decision tree
*/
public class RandomForest implements
java.io.Serializable {
	// storage of the variables needed to generate the original forest
	// the original trainingset
	private double[][] trainingset;
	// the labels of the training data
	private double[] labels;
	// the weight assigned to each training point
	// not for selection, but when weighting splits and leaf probabilities
	private double[] weights;
	// the shuffler allows a quicker repeated drawing of a small number of dimensions
	// from a set of weighted dimensions
	private Shuffler dimweights;
	// the explicit dimweight values
	private double[] dw;
	// the splitting function used for this forest
	private SplitFunction G;
	// the parameters used to created the forest
	private double[] parameters;
	// indicates whether the input dimensions are categorical or not
	private boolean[] categorical;
	// the split purity necessary for any leaf node for classification forests
	// or the minimal squared error for regression forests
	private double[] splitpurity;
	// number of trees within this forest
	private int ntree;
	// input dimensionality of the data, number of training points, number of classes (for
	// classification)
	private int dim,n,num_class;
	// collection of decision trees
	private ArrayList<DecisionTree> forest = new ArrayList<DecisionTree>();
	// ordering of the samples for each decision tree. Lets us determine which samples
	// we in bag, and which were out of bag
	private int[][] samples;
	// leafindices of the training data in all trees
	private int[][] leafind;
	
	// generally unused constructor
	public RandomForest(){
		
	}
	
	// Initialization of the random forest
	public RandomForest(final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final double[] dimweights, final double[] parameters, final double[] splitpurity, final SplitFunction G, int ntree){
		this.dim = trainingset[0].length;
		this.n = trainingset.length;
		this.trainingset = new double[n][this.dim];
		for (int i=0; i<n; i++){
			this.trainingset[i] = trainingset[i].clone();
		}
		this.categorical = categorical.clone();
		this.dimweights = new Shuffler(dimweights);
		this.dw = dimweights.clone();
		// the ugly reference copy
		// splitfunctions should preferably not contain any private variables
		// but only function definitions
		this.G = G;
		this.labels = labels.clone();
		this.weights = weights.clone();
		this.parameters = parameters.clone();
		this.ntree = ntree;
		this.num_class = (int)parameters[0];
		// what percentage of the points in a node must belong to a single class to be counted as a terminal node
		// 1 is the same as the classical random forest
		this.splitpurity = splitpurity.clone();
		this.generateForest();
	}
	
	// This forest generates a collection of decision trees based on random samples of the training set
	public void generateForest(){
		this.samples = new int[this.ntree][this.n];
		// initiate the variable used for proximity calculations
		this.leafind = new int[this.n][];
		// for each tree
		for (int i=0; i<this.ntree; i++){
			// create a new tree
			forest.add(new DecisionTree());
			// choose a random sample of points to train on
			int[] randind = Shuffle.randPerm(this.n);
			// this is the typical fraction of examples that would be drawn at least once
			// from all examples if we would draw with resampling
			int nt = (int)(this.n*(1.-1./Math.E));
			if (parameters[10]>0&&parameters[10]<1){
				(int)(this.n*parameters[10]);
			}
			int[] points = new int[nt];
			for (int j=0; j<nt; j++){
				points[j] = randind[j];
			}
			this.samples[i] = randind.clone();
			// generate the decision tree
			forest.get(i).generateTree(points, this.trainingset, this.labels, this.weights, this.categorical, this.dimweights, this.parameters, this.splitpurity, this.G);
		}
	}
	
	// The out of bag error is the average classification error of all samples in all trees
	// in which the samples were not in the tree training set
	// Out of bag error for the regression tree is the average squared sum difference of the regression value
	// of all out of bag cases versus the training label
	public double[] outOfBagError(){
		// general error rate&individual error rates
		return G.oobe(this.samples,(int)this.parameters[0],this.trainingset,this.labels,this.weights,this.forest,this.parameters);
	}
	
	// variable importance is calculated by taking over all trees the average of the correct number
	// of classifications of a point minus the correct number of classifications when 
	// a variable is replaced by a randomly drawn variable
	public double[][] variableImportance(){
		return G.variableImportance(this.samples,(int)this.parameters[0],this.trainingset,this.labels,this.weights,this.forest,this.parameters);
	}
	
	// the splitting function decides what to return when the forest is applied
	// will typically return something related to the distribution of leaf outputs
	public double[] applyForest(final double[] point){
		return G.applyForest(point,this.forest);
	}
	
	public double[][] applyForest(final double[][] point){
		return G.applyForest(point,this.forest);
	}
	
	public double[] applyForest(final double[] point, int[] missing){
		return G.applyForest(point,missing,this.forest,this.num_class);
	}
	
	// same as for the forest application
	// prediction which be a single return value though
	public double predict(final double[] point){
		return G.predict(point,this.forest);
	}
	
	public double[] predict(final double[][] points){
		double[] probabilities = new double[points.length];
		for (int i=0; i<points.length; i++){
			probabilities[i] = this.predict(points[i]);
		}
		return probabilities;
	}
	
	// this function generates the proximity matrix between all points given
	// in the training set variable
	// proximity between two points is the number of trees in which their evaluation
	// ends within the same leaf node
	public int[][] generateProximities(final double[][] trainingset){
		int n = trainingset.length;
		int dim = trainingset[0].length;
		// theoretically we could also store a sparse representation of this matrix
		// this is what is done in the spectral matrices codes. Therefore it's preferable
		// to use that code if spectral clustering based on proximities is desired.
		int[][] proximities = new int[n][n];
		int[][] leafind = new int[n][this.ntree];
		for (int i=0; i<n; i++){
			for (int j=0; j<this.ntree; j++){
				leafind[i][j] = this.forest.get(j).getLeafIndex(trainingset[i]);
			}
		}
		for (int i=0; i<n; i++){
			for (int j=0; j<n; j++){
				for (int k=0; k<this.ntree; k++){
					if (leafind[i][k]==leafind[j][k]){
						proximities[i][j]++;
					}
				}
			}
		}
		return proximities;
	}
	
	// find the proximities of a point to a set of points of the initial training set
	public int[] pointToSetProximities(final double[] point, final int[] indices){
		int n = indices.length;
		int dim = this.trainingset[0].length;
		int[] proximities = new int[n];
		int[] pointind = new int[this.ntree];
		for (int i=0; i<n; i++){
			if (this.leafind[indices[i]]==null){
				this.leafind[indices[i]] = new int[this.ntree];
				for (int j=0; j<this.ntree; j++){
					this.leafind[indices[i]][j] = this.forest.get(j).getLeafIndex(trainingset[indices[i]]);
				}
			}
		}
		for (int i=0; i<this.ntree; i++){
			pointind[i] = this.forest.get(i).getLeafIndex(point);
		}
		for (int i=0; i<n; i++){
			for (int k=0; k<this.ntree; k++){
				if (leafind[i][k]==pointind[k]){
					proximities[i]++;
				}
			}
		}
		return proximities;
	}
	
	// get the number of leaves of each tree in the forest
	public int[] getTreeSizes(){
		int[] treeSizes = new int[this.ntree];
		int a;
		for (int i=0; i<this.ntree; i++){
			treeSizes[i] = this.forest.get(i).getLeafIndex();
		}
		return treeSizes;
	}
	
	// get all leaf indices of the forest for a single point
	public int[] getLeafIndices(final double[] point){
		int[] indices = new int[this.ntree];
		for (int i=0; i<this.ntree; i++){
			indices[i] = this.forest.get(i).getLeafIndex(point);
		}
		return indices;
	}
	
	public int[][] getLeafIndices(final double[][] point){
		return G.getLeafIndices(point,this.forest);
	}
	
	// generate a classification specialized sample based on the forest structure
	// this doesn't uphold correlations in the input data... therefore the usefulness
	// of this sampling is questionable
	public double[] sample(final double[] label){
		double[] point = new double[dim];
		int[] missing = new int[dim];
		for (int i=0; i<dim; i++){
			missing[i] = i;
		}
		return G.sample(point,missing,label,forest,trainingset,labels,weights);
	}
	
	public double[] sample(final double[] point, final int[] missing,final double[] label){
		return G.sample(point,missing,label,forest,trainingset,labels,weights);
	}
	
	public double[] sample(){
		double[] point = new double[dim];
		int[] missing = new int[dim];
		for (int i=0; i<dim; i++){
			missing[i] = i;
		}
		return sample(point,missing);
	}
	
	public double[] sample(final double[] point, final int[] missing){
		int l = missing.length;
		Random rng = new Random();
		double[] samp = point.clone();
		int[] randind = Shuffle.randPerm(l);
		for (int i=0; i<l; i++){
			int[] miss = new int[l-i];
			for (int j=0; j<l-i;j++){
				miss[j] = missing[randind[i+j]];
			}
			Arrays.sort(miss);
			TreeSet<SortPair2> list = new TreeSet<SortPair2>();
			for (int j=0; j<ntree; j++){
				forest.get(j).neighbors(point,miss,list,weights);
			}
			int[] neighbors = new int[list.size()];
			double[] weights = new double[list.size()];
			Iterator<SortPair2> itr = list.iterator();
			SortPair2 sp = itr.next();
			neighbors[0] = (int)sp.getValue();
			weights[0] = sp.getOriginalIndex();
			int counter = 1;
			while (itr.hasNext()){
				sp = itr.next();
				neighbors[counter] = (int)sp.getValue();
				weights[counter] = weights[counter-1]+sp.getOriginalIndex();
				counter++;
			}
			int b = VectorFun.binarySearch(weights,rng.nextDouble()*(weights[counter-1]));
			samp[missing[randind[i]]] = trainingset[neighbors[b]][missing[randind[i]]];
		}
		return samp;
	}
	
	public double[] nearestNeighbor(final double[] point, final int[] missing, final double[] label){
		return G.nearestNeighbor(point,missing,label,forest,trainingset,labels);
	}
	
	public double[] nearestNeighbor(final double[] point, final int[] missing){
		TreeSet<SortPair2> list = new TreeSet<SortPair2>();
		for (int i=0; i<ntree; i++){
			forest.get(i).neighbors(point,missing,list);
		}
		double[] m = new double[2];
		Iterator<SortPair2> itr = list.iterator();
		while (itr.hasNext()){
			SortPair2 sp = itr.next();
			if (sp.getOriginalIndex()>m[1]){
				m[1] = sp.getOriginalIndex();
				m[0] = sp.getValue();
			}
		}
		return trainingset[(int)m[0]].clone();
	}
	
	public double[] sampledNeighbor(final double[] point, final int[] missing, int type, double p){
		TreeSet<SortPair2> list = new TreeSet<SortPair2>();
		for (int i=0; i<ntree; i++){
			forest.get(i).neighbors(point,missing,list);
		}
		double[] cs = new double[list.size()];
		int[] o = new int[list.size()];
		Iterator<SortPair2> itr = list.iterator();
		SortPair2 sp = itr.next();
		cs[0] = Math.pow(sp.getOriginalIndex(),p);
		o[0] = (int)sp.getValue();
		int counter = 1;
		while (itr.hasNext()){
			sp = itr.next();
			if (type%2==0){
				cs[counter] = cs[counter-1]+Math.pow(sp.getOriginalIndex(),p);
			}
			else {
				cs[counter] = sp.getOriginalIndex();
			}
			o[counter] = (int)sp.getValue();
			counter++;
		}
		int a = 0;
		if (type%2==0){
			a = o[VectorFun.binarySearch(cs,(new Random()).nextDouble()*cs[counter-1])];
		}
		else {
			int[] o2 = (new RankSort(cs)).getRank();
			int b = (int)p;
			if (b>o2.length){
				b = o2.length;
			}
			a = o[o2[o2.length-1-(new Random()).nextInt(b)]];
		}
		if (type/2==0){
			return trainingset[a].clone();
		}
		else {
			double[] temp = new double[1];
			temp[0] = a;
			return temp;
		}
	}
	
	public double[] sampledNeighbor(final double[] point, final int[] missing, final double[] label, int type, double p){
		return G.sampledNeighbor(point,missing,label,forest,trainingset,labels,type,p);
	}
	
	public double[][] getTrainingset(){
		return this.trainingset;
	}
	
	public double[] getWeights(){
		return this.weights;
	}
	
	public double[] getLabels(){
		return this.labels;
	}
	
	public double[] getParameters(){
		return this.parameters;
	}
	
	public boolean[] getCategorical(){
		return this.categorical;
	}
	
	public Shuffler getDimWeights(){
		return this.dimweights;
	}
	
	public double[] getDW(){
		return this.dw;
	}
	
	public int getNumClasses(){
		return (int)this.parameters[0];
	}
	
	public int getMtry(){
		return (int)this.parameters[1];
	}
	
	public int getNtree(){
		return this.ntree;
	}
	
	public int getMaxDepth(){
		return (int)this.parameters[2];
	}

	public double getMaxLeafSize(){
		return this.parameters[1];
	}
	
	public double[] getSplitPurity(){
		return this.splitpurity;
	}
	
	public SplitFunction getSplitFunction(){
		return this.G;
	}
	
	public int[][] getSamples(){
		return samples;
	}
	
	public ArrayList<DecisionTree> getTrees(){
		return this.forest;
	}
	
	public void copyForestContent(RandomForest rf){
		// shuffler function and decision trees aren't copied...
		trainingset = new double[rf.getTrainingset().length][rf.getTrainingset()[0].length];
		for (int i=0; i<trainingset.length; i++){
			trainingset[i] = rf.getTrainingset()[i];
		}
		samples = new int[rf.getSamples().length][rf.getSamples()[0].length];
		for (int i=0; i<samples.length; i++){
			samples[i] = rf.getSamples()[i];
		}
		labels = rf.getLabels().clone();
		weights = rf.getWeights().clone();
		dw = rf.getDW().clone();
		parameters = rf.getParameters().clone();
		categorical = rf.getCategorical().clone();
		splitpurity = rf.getSplitPurity().clone();
		ntree = rf.getNtree();
		dim = trainingset[0].length;
		n = trainingset.length;
		num_class = (int)parameters[0];
		// let's just assume G our splitting functions will only have a default
		// constructor...
		// soooo ugly....
		try {
			Class<?> clazz = rf.getSplitFunction().getClass();
			Constructor<?> ctor = clazz.getConstructors()[0];
			G = (SplitFunction)ctor.newInstance();
		} catch (Exception e)
		{
			System.out.println("Problem copying the Splitfunction"+
				" in the random forest");
		}
	}
}