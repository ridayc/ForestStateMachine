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
import flib.math.SortPair2;
import flib.math.VectorFun;
import flib.math.RankSort;
import flib.math.random.Shuffle;
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
	// using weights instead of multiple copies of input provides more reliable estimates of the oobe
	private double[] weights;
	// ordering of the samples for each decision tree. Lets us determine which samples
	// were in bag, and which were out of bag
	private int[][] samples;
	// leafindices of the training data in all trees
	private int[][] leafind;
	// the splitting function used for this forest
	// indicates whether the input dimensions are categorical or not
	private boolean[] categorical;
	private SplitFunction G;
	// number of trees within this forest
	private int ntree;
	// input dimensionality of the data, number of training points, number of classes (for
	// classification)
	private int dim,n,num_class;
	// fraction of input points for resampling (standard value is 
	private double resample;
	// seed for the random number generator
	private long seed;
	// if the forest should be generated in parallel
	private boolean parallel;
	// collection of decision trees
	private ArrayList<DecisionTree> forest = new ArrayList<DecisionTree>();
	private double[][] oob;
	private double[][] oobe;
	private double[][] variableImportance;
	
	// generally unused constructor
	public RandomForest(){
		
	}
	
	// Initialization of the random forest
	public RandomForest(final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final SplitFunction G, int ntree, long seed, boolean parallel, double resample){
		this.dim = trainingset[0].length;
		this.n = trainingset.length;
		this.categorical = categorical.clone();
		this.trainingset = new double[n][this.dim];
		for (int i=0; i<n; i++){
			this.trainingset[i] = trainingset[i].clone();
		}
		Random rng = new Random(seed);
		this.ntree = ntree;
		this.resample = resample;
		this.seed = seed;
		this.parallel = parallel;
		this.G = G.clone();
		this.labels = labels.clone();
		this.weights = weights.clone();
		this.generateForest();
	}
	
	// This forest generates a collection of decision trees based on random samples of the training set
	public void generateForest(){
		if (ntree > 0){
			this.samples = new int[this.ntree][this.n];
			// initiate the variable used for proximity calculations
			this.leafind = new int[this.n][];
			// for each tree
			for (int i=0; i<this.ntree; i++){
				forest.add(new DecisionTree());
			}
			if (parallel){
				int NUM_CORES = Runtime.getRuntime().availableProcessors();
				//final int NUM_CORES = 1;
				ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
				try {
					for (int i=0; i<this.ntree; i++){
						exec.submit(new DecisionTreeThread(this,samples,i));
					}
				}
				finally {
					exec.shutdown();
				}
				while(!exec.isTerminated()){
					// wait
				}
			}
			else {
				for (int i=0; i<this.ntree; i++){
					this.samples[i] = generateDecisionTree(this.getForest(), this.getForest.get(i))
				}
			}
		}
		else {
			ntree = -ntree;
			// generate tree one by one
		}
	}
	
	private int[] generateDecisionTree(RandomForest RF, DecisionTree DT){
		int n = RF.getTrainingset().length;
		int nt = (int)(n*(1.-resample));
		int[] randind = Shuffle.randPerm(n,rng);
		int[] points = new int[nt];
		for (int j=0; j<nt; j++){
			points[j] = randind[j];
		}
		int[] samples = randind.clone();
		DT.generateTree(points, RF.getTrainingset(), RF.getLabels(), RF.getWeights(), RF.getCategorical(), RF.getSplitFunction());
		return samples;
	}
	
	private class DecisionTreeThread implements Runnable{
		private RandomForest RF;
		private int[][] samples;
		private int i;
		
		public DecisionTreeThread(RandomForest RF, int[][] samples, int i){
			this.RF = RF;
			this.samples = samples;
			this.i = i;
		}
		
		public void run(){
			try {
				this.samples[i] = generateDecisionTree(RF,RF.getTrees().get(i));
				}
			catch (Throwable t){
				System.out.println("Didn't fully construct the decision tree");
				t.printStackTrace();
			}
		}
	}
	
	// The out of bag error is the average classification error of all samples in all trees
	// in which the samples were not in the tree training set
	// Out of bag error for the regression tree is the average squared sum difference of the regression value
	// of all out of bag cases versus the training label
	public double[] outOfBagError(){
		// general error rate&individual error rates
		return G.oobe(this.samples,(int)this.parameters[0],this.trainingset,this.labels,this.weights,this.tree_weights,this.forest,this.parameters);
	}
	
	public double[][] outOfBagErrorConvergence(){
		// general error rate&individual error rates
		return G.oobeConvergence(this.samples,(int)this.parameters[0],this.trainingset,this.labels,this.weights,this.tree_weights,this.forest,this.parameters);
	}
	
	// variable importance is calculated by taking over all trees the average of the correct number
	// of classifications of a point minus the correct number of classifications when 
	// a variable is replaced by a randomly drawn variable
	public double[][] variableImportance(){
		return G.variableImportance(this.samples,(int)this.parameters[0],this.trainingset,this.labels,this.weights,this.tree_weights,this.forest,this.parameters);
	}
	
	// the splitting function decides what to return when the forest is applied
	// will typically return something related to the distribution of leaf outputs
	public double[] applyForest(final double[] point){
		return G.applyForest(point,this.tree_weights,this.forest);
	}
	
	public double[][] applyForest(final double[][] point){
		return G.applyForest(point,this.tree_weights,this.forest);
	}
	
	public double[] applyForest(final double[] point, int[] missing){
		return G.applyForest(point,missing,this.forest,this.num_class);
	}
	
	// same as for the forest application
	// prediction which be a single return value though
	public double predict(final double[] point){
		return G.predict(point,this.tree_weights,this.forest);
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
	
	public void setProximities(final int[] indices){
		int n = indices.length;
		int dim = this.trainingset[0].length;
		for (int i=0; i<n; i++){
			if (this.leafind[indices[i]]==null){
				this.leafind[indices[i]] = new int[this.ntree];
				for (int j=0; j<this.ntree; j++){
					this.leafind[indices[i]][j] = this.forest.get(j).getLeafIndex(trainingset[indices[i]]);
				}
			}
		}
	}
	
	// always make sure the indices have been set at some point
	// find the proximities of a point to a set of points of the initial training set
	public int[] pointToSetProximities(final double[] point, final int[] indices){
		int n = indices.length;
		int dim = this.trainingset[0].length;
		int[] proximities = new int[n];
		int[] pointind = new int[this.ntree];
		for (int i=0; i<this.ntree; i++){
			pointind[i] = this.forest.get(i).getLeafIndex(point);
		}
		for (int i=0; i<n; i++){
			int ind = indices[i];
			for (int k=0; k<this.ntree; k++){
				if (leafind[ind][k]==pointind[k]){
					proximities[i]++;
				}
			}
		}
		return proximities;
	}
	
	// find the proximities of multiple point to a set of points of the initial training set
	public int[][] pointToSetProximities(final double[][] point, final int[] indices){
		final int[][] proximities = new int[point.length][indices.length];
		final int NUM_CORES = Runtime.getRuntime().availableProcessors();
		ExecutorService exec = Executors.newFixedThreadPool(NUM_CORES);
		try {
			for (int i=0; i<NUM_CORES; i++){
				final int i2 = i;
				exec.submit(new Runnable() {
					@Override
					public void run(){
						try {
							for (int j=i2; j<point.length; j+=NUM_CORES){
								int[] temp = pointToSetProximities(point[j],indices);
								for (int k=0; k<temp.length; k++){
									proximities[j][k] = temp[k];
								}
							}
						}
						catch (Throwable t){
							System.out.println("set proximities problem");
							t.printStackTrace();
						}
					}
				});
			}
		}
		finally {
			exec.shutdown();
		}
		while(!exec.isTerminated()){
			// wait
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
	
	public double[] getKNearestNeighbors(final double[] point, int k){
		TreeSet<SortPair2> list = new TreeSet<SortPair2>();
		// find all points which share a leaf with the current point
		for (int i=0; i<this.forest.size(); i++){
			int[] temp = this.forest.get(i).getLeafNeighbors(point);
			double w = tree_weights[i];
			for (int j=0; j<temp.length; j++){
				SortPair2 sp = new SortPair2(temp[j],w);
				if (!list.contains(sp)){
					list.add(sp);
				}
				else {
					list.floor(sp).setOriginalIndex(list.floor(sp).getOriginalIndex()+w);
				}
			}
		}
		// sort the points according to number of leaf overlaps
		double[] val = new double[list.size()];
		double[] orig = new double[list.size()];
		Iterator<SortPair2> itr = list.iterator();
		int count = 0;
		while (itr.hasNext()){
			SortPair2 sp = itr.next();
			val[count] = sp.getValue();
			orig[count] = sp.getOriginalIndex();
			count++;
		}
		// find the k nearest points
		RankSort rs = new RankSort(orig,val);
		int[] o = rs.getRank();
		double[] s = rs.getSorted();
		double[] neighbors = new double[2*k];
		for (int i=0; i<k; i++){
			int l = o.length-1-(i%o.length);
			neighbors[2*i] = o[l];
			neighbors[2*i+1] = s[l];
		}
		return neighbors;
	}
	
	// proximity k nearest neighbors vote
	public double[] getKNNV(final double[] point, int k){
		int nclass = (int)parameters[0];
		double[] votes = new double[nclass];
		double[] loc = getKNearestNeighbors(point,k);
		double c = 0;
		if (nclass>1){
			for (int i=0; i<k; i++){
				votes[(int)labels[(int)loc[2*i]]]+=loc[2*i+1];
				c+=loc[2*i+1];
			}
		}
		else {
			for (int i=0; i<k; i++){
				votes[0]+=labels[(int)loc[2*i]]*loc[2*i+1];
				c+=loc[2*i+1];
			}
		}
		VectorFun.multi(votes,1./c);
		return votes;
	}
	
	public double[][] getKNNV(final double[][] point, int k){
		int nclass = (int)parameters[0];
		double[][] votes = new double[point.length][nclass];
		for (int i=0; i<point.length; i++){
			votes[i] = getKNNV(point[i],k);
		}
		return votes;
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
	
	public double[] getTree_weights(){
		return this.tree_weights;
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
		return this.forest.size();
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
	
	public void setTree_weights(final double[] tree_weights){
		this.tree_weights = tree_weights.clone();
	}
	
	public void setSamples(final int[][] samples){
		this.ntree = samples.length;
		this.samples = new int[ntree][samples[0].length];
		for (int i=0; i<this.ntree; i++){
			this.samples[i] = samples[i].clone();
		}
	}
	
	public void setTrainingset(final double[][] trainingset){
		this.trainingset = new double[trainingset.length][trainingset[0].length];
		for (int i=0; i<trainingset.length; i++){
			this.trainingset[i] = trainingset[i].clone();
		}
	}
	
	public void setTrees(ArrayList<DecisionTree> forest){
		this.forest = forest;
	}
	
	public void setNtree(int ntree){
		this.ntree = ntree;
	}
	
	public void setParameters(final double[] parameters){
		this.parameters = parameters.clone();
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
		tree_weights = rf.getTree_weights().clone();
		dw = rf.getDW().clone();
		parameters = rf.getParameters().clone();
		categorical = rf.getCategorical().clone();
		splitpurity = rf.getSplitPurity().clone();
		ntree = rf.getNtree();
		dim = trainingset[0].length;
		n = trainingset.length;
		num_class = (int)parameters[0];
		leafind = new int[n][];
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