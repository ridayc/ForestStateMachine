package flib.algorithms.randomforest;

public class DecisionNode implements
java.io.Serializable {
	// dimension at which the current splitting/decision is made
	private int dim;
	// Indicates if the current splitting dimension if categorical or not
	private boolean categorical;
	// The value with which a comparison is made to choose which child node
	// to branch to in a decision tree
	private double splitpoint;
	// a double array containing the probality of belonging to any specific class
	// this array is only non-empty for leaf nodes
	private double[] probabilities = null;
	// list of all training points contained in this node if it's a leaf node
	private int[] points = null;
	// weight of all points in this node. Only for leaf nodes
	private double weight;
	
	// Each leaf in a decision tree has its own specific index
	// This is important to measure the "distance" between points
	// according to the tree structure (e.g. for proximities)
	// Equal to -1 if the current node is not a leaf node
	private int leafindex = -1;
	
	public boolean isLeafNode(){
		if(this.probabilities!=null) {
			return true;
		}
		else {
			return false;
		}
	}
	
	//getter functions
	public int getDim(){
		return this.dim;
	}
	
	public boolean getCategorical(){
		return this.categorical;
	}
	
	public double getSplit(){
		return this.splitpoint;
	}
	
	public double[] getProbabilities(){
		return this.probabilities;
	}
	
	public int getLeafIndex(){
		return this.leafindex;
	}
	
	public int[] getPoints(){
		return this.points;
	}
	
	public double getWeight(){
		return this.weight;
	}
	
	// setter functions
	public void setDim(int dim){
		this.dim = dim;
	}
	
	public void setCategorical(boolean cat){
		this.categorical = cat;
	}
	
	public void setSplit(double splitpoint){
		this.splitpoint = splitpoint;
	}
	
	public void setLeaf(final double[] probabilities){
		this. probabilities = probabilities.clone();
	}
	
	public void setLeafIndex(int leafindex){
		this.leafindex = leafindex;
	}
	
	public void setPoints(final int[] points){
		this.points = points.clone();
	}
	
	public void setWeight(double weight){
		this.weight = weight;
	}
	
	// This function specifies if a given point is on the right or left side of the split point
	// or if it is in the specified class if the splitting dimension is categorical
	public boolean evaluatePoint(final double[] point){
		if (this.categorical){
			if (point[this.dim]==this.splitpoint) return true;
			else return false;
		}
		else {
			if (point[this.dim]<this.splitpoint) return true;
			else return false;
		}
	}
}