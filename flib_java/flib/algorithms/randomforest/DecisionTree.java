package flib.algorithms.randomforest;

import java.util.LinkedList;
import java.util.Random;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Arrays;
import java.util.ArrayList;
import flib.math.RankSort;
import flib.math.SortPair2;
import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;
import flib.algorithms.randomforest.splitfunctions.SplitFunction;

// The decision tree class contains the prototype for random forest decision trees (though the decision trees could also be used for different purposes
public class DecisionTree implements
java.io.Serializable {
	// The actual decision tree, composed of a root node and its linked child nodes (down to a leaf level)
	private TreeNode<DecisionNode> myTree = new TreeNode<DecisionNode>();
	private int leafindex;
	
	/* This is the main tree generation function (as the name states). Meaning of the input variables:
	training set: a list of all the points used to generate the decision. The second dimension contains the different dimension entries of the points
	labels: the class labels of the individual training points
	weights: the weight that should be given to each individual training point
	categorical: indicates if the corresponding dimension is a categorical dimension or not
	paramters: contains the following variables
		numclasses: the number of classes the tree should split the data into
		mtry: number of dimensions to randomly choose at each splitting node to choose a splitting dimension with
		maxdepth: maxdepth of the tree, in cases we don't want to fully grow the tree for memory reasons
		maxleafsize: similar in effect to maxdepth. This variable gives the minimum number of training points a leaf node should contain
		dimtype: method of choosing the mtry dimensions for splitting;
	splitpurity: a third measure related to maxdepth and maxleafsize. This variable gives the percentage of for alls labels which a leaf node has to contain
	of that label to require no further splitting.
	*/
	public void generateTree(final int[] points, final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final SplitFunction G){
		// initiate the root node
		myTree = new TreeNode<DecisionNode>();
		// prepare some temporary variables
		TreeNode<DecisionNode> tempnode;
		int[] templist;
		int tempdepth;
		// prepare a stack to construct the tree in a recursive manner
		// initiate the stack with the root node and trainingset
		LinkedList<StackElement> myStack = new LinkedList<StackElement>();
		myStack.add(new StackElement());
		myStack.getLast().setNode(myTree);
		myStack.getLast().setDepth(0);
		myStack.getLast().setPoints(points);
		// initiate the trees leafindex
		this.leafindex = 0;
		// number of points on both sides of the current split
		// recursively find the best split at each node and add the children of the split to the stack
		// repeat till the stack is empty
		// the size of the stack at any time will be at most twice the tree depth
		while (!myStack.isEmpty()){
			// variable split already performs the splitting when called
			if (this.variableSplit(myStack.getLast().getNode(),myStack.getLast().getPoints(),trainingset, labels, weights, categorical, dimweights, parameters, splitpurity, myStack.getLast().getDepth(), leafindex,G)){
				tempnode = myStack.getLast().getNode();
				templist = myStack.getLast().getPoints().clone();
				tempdepth = myStack.getLast().getDepth()+1;
				myStack.removeLast();
				// count the elements on each side of the split
				int count_left = 0, count_right = 0;
				// temporary lists for the points on each split side
				int[] pleft = new int[templist.length];
				int[] pright = new int[templist.length];
				// here we split the training points according to the side of the split they belong to
				for (int i=0; i<templist.length; i++){
					if (tempnode.getData().evaluatePoint(trainingset[templist[i]])){
						pleft[count_left] = templist[i];
						count_left++;
					}
					else {
						pright[count_right] = templist[i];
						count_right++;
					}
				}
				// update the stack elements for both branches
				myStack.add(new StackElement());
				myStack.getLast().setNode(tempnode.getLeft());
				myStack.getLast().setDepth(tempdepth);
				myStack.getLast().setPoints(new int[count_left]);
				for (int i=0; i<count_left; i++){
					myStack.getLast().getPoints()[i] = pleft[i];
				}
				myStack.add(new StackElement());
				myStack.getLast().setNode(tempnode.getRight());
				myStack.getLast().setPoints(new int[count_right]);
				for (int i=0; i<count_right; i++){
					myStack.getLast().getPoints()[i] = pright[i];
				}
				myStack.getLast().setDepth(tempdepth);
			}
			else {
				// increase the leaf index whenever we define a leaf node
				myStack.removeLast();
				this.leafindex++;
			}
			// reallocate the tree in memory?
		}
		
	}
	
	// this is the most important function for the tree generation
	// in this function it is determined where and how the splitting axis should be placed or if a node should be a leaf node
	public boolean variableSplit(TreeNode<DecisionNode> current, final int[] points, final double[][] trainingset,  final double[] labels, final double[] weights, final boolean[] categorical, final Shuffler dimweights, double[] parameters, final double[] splitpurity, int depth, int leafindex, SplitFunction G){
		// number of points contained at the current node
		int n = points.length;
		// check if the current node is a leaf node according to the splitting function
		// the splitting function has to update current as well, if current is a leaf node
		if (G.isLeaf(current,points,labels,weights,categorical,parameters,splitpurity,depth,leafindex)){
			return false;
		}
		
		// choose the dimensions for splitting
		int[] dim = new int[(int)parameters[1]];
		// different cases to find the dimensions
		// the are only a few equally weighted dimensions
		if (parameters[4]==0){
			Random rng = dimweights.getRNG();
			TreeSet<Integer> dimset = new TreeSet<Integer>();
			// keep adding new random dimensions until there are enough dimensions
			while(dimset.size()<parameters[1]){
				dimset.add(rng.nextInt(categorical.length));
			}
			// copy all values to the preset array
			Iterator<Integer> itr = dimset.iterator();
			int counter = 0;
			while (itr.hasNext()){
				dim[counter] = itr.next();
				counter++;
			}
		}
		// if there is a large set of equally weighted dimensions. Large means
		// in comparison to the number of dimensions of the training vectors
		else if (parameters[4]==1){
			// get a random permutation all numbers from 0 to mtry-1
			int[] dimset = Shuffle.randPerm(categorical.length);
			for (int i=0; i<parameters[1]; i++){
				dim[i] = dimset[i];
			}
		}
		// for the general case where all dimensions have been asigned a weight
		else {
			dim = dimweights.randPerm((int)parameters[1]);			
		}
		
		// we provide the splitting random splitting dimensions for this node
		// and all other neccessary variables to the splitting function
		// and receive the splitting dimension and the split location inside current
		return G.split(current,points,trainingset,labels,weights,categorical,dim,parameters,splitpurity,leafindex);
	}
	
	// the class which we use as a stack element for the recursive tree splitting function
	private class StackElement {
		private TreeNode<DecisionNode> current;
		private int[] points;
		private int depth;
		
		public TreeNode<DecisionNode> getNode(){
			return this.current;
		}
		
		// it should be noticed that a reference is being returned here
		// this function is actually more for setting purposes
		public int[] getPoints(){
			return this.points;
		}
		
		public int getDepth(){
			return this.depth;
		}
		
		public void setNode(TreeNode<DecisionNode> a){
			this.current = a;
		}
		
		public void setPoints(int[] points){
			this.points = points.clone();
		}
		
		public void setDepth(int depth){
			this.depth = depth;
		}
	}
	
	// as function which returns the probability of belong to all classes based on a single input points
	public double[] applyTree(final double[] point){
		TreeNode<DecisionNode> current = this.myTree;
		DecisionNode tmp = current.getData();
		while (tmp.getPoints()==null){
			if (tmp.evaluatePoint(point)){
				current = current.getLeft();
				tmp = current.getData();
			}
			else{
				current = current.getRight();
				tmp = current.getData();
			}
		}
		return tmp.getProbabilities();
	}
	
	// the applyTree function for a collection of points
	public double[][] applyTree(final double[][] points){
		// this function doesn't know the number of classes in advance
		// therefor we need to apply the tree to a single point to obtain the number
		// of classes
		double[] temp = this.applyTree(points[0]);
		double[][] probabilities = new double[points.length][temp.length];
		probabilities[0] = temp;
		for (int i=1; i<points.length; i++){
			probabilities[i] = this.applyTree(points[i]);
		}
		return probabilities;
	}
	
	public TreeNode<DecisionNode> getTree(){
		return this.myTree;
	}
	
	// a function to find which leaf in the tree a point would belong to (important for proximity calculations
	public int getLeafIndex(final double[] point){
		TreeNode<DecisionNode> current = this.myTree;
		DecisionNode tmp = current.getData();
		while (tmp.getPoints()==null){
			if (tmp.evaluatePoint(point)){
				current = current.getLeft();
				tmp = current.getData();
			}
			else{
				current = current.getRight();
				tmp = current.getData();
			}
		}
		current = null;
		int a = tmp.getLeafIndex();
		tmp = null;
		return a;
	}
	
	// a function to find which leaf in the tree a point would belong to (important for proximity calculations
	public int[] getLeafIndex(final double[][] points){
		int[] indices = new int[points.length];
		for (int i=0; i<points.length; i++){
			indices[i] = getLeafIndex(points[i]);
		}
		return indices;
	}
	
	public ArrayList<TreeNode<DecisionNode>> leaves(){
		ArrayList<TreeNode<DecisionNode>> leaf = new ArrayList<TreeNode<DecisionNode>>();
		TreeNode<DecisionNode> current = this.myTree;
		addChildren(leaf,current);
		return leaf;
	}
	
	private void addChildren(ArrayList<TreeNode<DecisionNode>> leaf, TreeNode<DecisionNode> current){
		DecisionNode tmp = current.getData();
		if (tmp.getPoints()==null){
			addChildren(leaf,current.getLeft());
			addChildren(leaf,current.getRight());
		}
		else {
			leaf.add(current);
		}
	}
	
	// return the total number of leaves this tree contains
	public int getLeafIndex(){
		return this.leafindex;
	}
	
	public int[] getLeafNeighbors(final double[] point){
		TreeNode<DecisionNode> current = this.myTree;
		DecisionNode tmp = current.getData();
		while (tmp.getPoints()==null){
			if (tmp.evaluatePoint(point)){
				current = current.getLeft();
				tmp = current.getData();
			}
			else{
				current = current.getRight();
				tmp = current.getData();
			}
		}
		return tmp.getPoints();
	}
	
	// the purpose of this function is to update the list variable with all points
	// contained in leaf nodes when using this tree for classification und the condition
	// that certain input dimension are missing
	public void neighbors(final double[] point, final int[] missing, final TreeSet<SortPair2> list, final double[] weights){
		neighbors(this.myTree,point,missing,list,weights);
	}
	
	// recursive implemenation of the algorithm proposed above
	private void neighbors(TreeNode<DecisionNode> current, final double[] point, final int[] missing, final TreeSet<SortPair2> list, final double[] weights){
		// the current node is a branching node
		if (current.getData().getPoints()==null){
			// if the current split dimension is among the missing dimensions
			if (Arrays.binarySearch(missing,current.getData().getDim())>=0){
				// in this case we need to go through both children
				neighbors(current.getLeft(),point,missing,list,weights);
				neighbors(current.getRight(),point,missing,list,weights);
			}
			// otherwise split as expected
			else {
				if (current.getData().evaluatePoint(point)){
					neighbors(current.getLeft(),point,missing,list,weights);
				}
				else{
					neighbors(current.getRight(),point,missing,list,weights);
				}
			}
		}
		// the current node is a leaf node
		else {
			int[] p = current.getData().getPoints();
			// go through all points at this leaf
			for (int i=0; i<p.length; i++){
				// add a new point to the list of points if this point isn't contained
				// otherwise we have to update the internal value to count the 
				// current leaf point
				if (!list.add(new SortPair2(p[i],weights[p[i]]))){
					list.floor(new SortPair2(p[i],0)).setOriginalIndex(list.floor(new SortPair2(p[i],0)).getOriginalIndex()+weights[p[i]]);
				}
			}
		}
	}
	
	// non-weighted version of the neighbor function from above
	public void neighbors(final double[] point, final int[] missing, final TreeSet<SortPair2> list){
		neighbors(this.myTree,point,missing,list);
	}
	
	private void neighbors(TreeNode<DecisionNode> current, final double[] point, final int[] missing, final TreeSet<SortPair2> list){
		if (current.getData().getPoints()==null){
			if (Arrays.binarySearch(missing,current.getData().getDim())>=0){
				// in this case we need to go through both children
				neighbors(current.getLeft(),point,missing,list);
				neighbors(current.getRight(),point,missing,list);
			}
			else {
				if (current.getData().evaluatePoint(point)){
					neighbors(current.getLeft(),point,missing,list);
				}
				else{
					neighbors(current.getRight(),point,missing,list);
				}
			}
		}
		else {
			int[] p = current.getData().getPoints();
			for (int i=0; i<p.length; i++){
				// add a new point to the list of points if this point isn't contained
				// otherwise we have to update the internal value
				if (!list.add(new SortPair2(p[i],1))){
					list.floor(new SortPair2(p[i],0)).setOriginalIndex(list.floor(new SortPair2(p[i],0)).getOriginalIndex()+1);
				}
			}
		}
	}
	
	// as function which returns the probability of belong to all classes based on a single input points
	// this case handles missing input dimensions
	public double[] applyTree(final double[] point, final int[] missing, int num_class){
		double[] p = new double[num_class+1];
		applyTree(this.myTree,point,missing,p);
		double[] pr = new double[num_class];
		for (int i=0; i<p.length-1; i++){
			pr[i] = p[i]/p[p.length-1];
		}
		return pr;
	}
	
	private void applyTree(TreeNode<DecisionNode> current, final double[] point, final int[] missing, double[] p){
		if (current.getData().getPoints()==null){
			if (Arrays.binarySearch(missing,current.getData().getDim())>=0){
				// in this case we need to go through both children
				applyTree(current.getLeft(),point,missing,p);
				applyTree(current.getRight(),point,missing,p);
			}
			else {
				if (current.getData().evaluatePoint(point)){
					applyTree(current.getLeft(),point,missing,p);
				}
				else{
					applyTree(current.getRight(),point,missing,p);
				}
			}
		}
		else {
			double[] pr = current.getData().getProbabilities();
			double wt = current.getData().getWeight();
			int l = pr.length;
			for (int i=0; i<l; i++){
				p[i]+=pr[i]*wt;
			}
			p[l]+=wt;
		}
	}
	
	public TreeNode<DecisionNode> getRoot(){
		return this.myTree;
	}
	
	public void setLeafIndex(int leafindex) {
		this.leafindex = leafindex;
	}
}