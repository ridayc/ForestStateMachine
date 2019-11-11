package flib.math;

import flib.math.SortPair;

/*
This class contains an implementation of a Red-Black Tree where each node additionally contains a weight value, as well as the sum of the
weights of its left sub tree. Additionally each node contains an indexing value, so that elements log(n) random access time. This version
accepts SortPair's as an input, but this class could be adapted if other behavior is required.
*/

// the wrapper class is responsible to keep a check on the total tree weight and it's size for indexing
public class WRBTreeNode implements  java.io.Serializable{
	// color of the current node red and black are true and false
	// default color is false/black
	private boolean color;
	// sum of the weights of the left subtree
	// default sum is 0
	private double weight_left;
	// number of entries within the left subtree
	// default value is 0
	private int index;
	// the content of the node
	// weight.getValue() contains the weight of the node
	// weight.getOriginal index will typically contain an array location
	// but could contain any double value desired
	private SortPair weight;
	// left child node
	private WRBTreeNode left;
	// right child node
	private WRBTreeNode right;
	// reference to parent node
	private WRBTreeNode parent;
	
	// the default constructor is the default constructor
	public WRBTreeNode() {
	}
	
	// getter functions
	public WRBTreeNode getParent(){
		return this.parent;
	}
	
	public WRBTreeNode getLeft(){
		return this.left;
	}
	
	public WRBTreeNode getRight(){
		return this.right;
	}
	
	// return the parent's parent
	public WRBTreeNode getGrandparent(){
		if (this.parent!=null){
			return this.parent.getParent();
		}
		else return null;
	}
	
	// return the sibling node of the parent node
	public WRBTreeNode getUncle(){
		if (this.getGrandparent()==null){
			return null;
		}
		else if (this.getGrandparent().getLeft()==this.parent){
			return this.getGrandparent().getRight();
		}
		else {
			return this.getGrandparent().getLeft();
		}
	}
	
	// return the sibling node
	// case of parent==null will result in a runtime error
	public WRBTreeNode getSibling(){
		if (this==this.parent.getLeft()){
			return this.parent.getRight();
		}
		else {
			return this.parent.getLeft();
		}
	}
	
	public SortPair getWeight(){
		return this.weight;
	}
	
	public double getWeight_left(){
		return this.weight_left;
	}
	
	public int getIndex(){
		return this.index;
	}
	
	public boolean getColor(){
		return this.color;
	}
	
	
	//setter functions
	public void setParent(WRBTreeNode p){
		this.parent = p;
	}
	
	// the child setter functions also the the child's parent correctly
	public void setLeft(){
		this.left = new WRBTreeNode();
		this.left.setParent(this);
	}
	
	public void setLeft(WRBTreeNode l){
		this.left = l;
		if (left!=null){
			this.left.setParent(this);
		}
	}
	
	public void setRight(){
		this.right = new WRBTreeNode();
		this.right.setParent(this);
	}
	
	public void setRight(WRBTreeNode r){
		this.right = r;
		if (right!=null){
			this.right.setParent(this);
		}
	}
	
	public void setColor(boolean c){
		this.color = c;
	}
	
	public void setWeight(SortPair w){
		this.weight = w;
	}
	
	public void setWeight_left(double wl){
		this.weight_left = wl;
	}
	
	public void setIndex(int ind){
		this.index = ind;
	}
	
	// insert a new pair into the tree
	// a recursive insertion function
	// there will be problems with insertion, if the inserted value
	// already exists in the tree
	public void insert(SortPair w){
		// if there are no children, or no weight has been set, insert here
		if (this.weight==null){
			// the new node is red
			this.color = true;
			// the new node has two empty leaf nodes 
			this.setLeft();
			this.setRight();
			// set the new weight
			this.weight = new SortPair(w.getValue(),w.getOriginalIndex());
			// check if the tree needs to be rebalanced
			this.insert_case1();
		}
		else if (this.weight.compareTo(w)>0){
			this.weight_left+=w.getValue();
			this.index++;
			this.left.insert(w);
		}
		else if (this.weight.compareTo(w)<0){
			this.right.insert(w);
		}
	}
	
	// Insertion rebalancing functions
	private void insert_case1(){
		// we are at the root node
		if (this.parent==null){
			this.color = false;
		}
		else {
			this.insert_case2();
		}
	}
	
	private void insert_case2(){
		// in case the parent color is false there are no further worries
		// otherwise
		if (this.parent.getColor()){
			this.insert_case3();
		}
	}
	
	private void insert_case3(){
		if (this.getUncle()!=null&&this.getUncle().getColor()){
			this.parent.setColor(false);
			this.getUncle().setColor(false);
			this.getGrandparent().setColor(true);
			this.getGrandparent().insert_case1();
		}
		else {
			this.insert_case4();
		}
	}
		
	private void insert_case4(){
		// fun rotations...
		if((this==this.parent.getRight())&&(this.parent==this.getGrandparent().getLeft())){
			this.parent.rotate_left();
			this.left.insert_case5();
		}
		else if((this==this.parent.getLeft())&&(this.parent==this.getGrandparent().getRight())){
			this.parent.rotate_right();
			this.right.insert_case5();
		}
		else {
			insert_case5();
		}
	}
	
	private void insert_case5(){
		this.parent.setColor(false);
		this.getGrandparent().setColor(true);
		if (this==this.parent.getLeft()){
			this.getGrandparent().rotate_right();
		}
		else {
			this.getGrandparent().rotate_left();
		}
	}
	
	// general right and left tree rotation which contains the adjusting of the weight and index sums of the left subtrees
	private void rotate_left(){
		// the current child node
		WRBTreeNode n = this.getRight();
		// reassign indices and weights
		n.setWeight_left(this.getWeight_left()+n.getWeight_left()+this.getWeight().getValue());
		n.setIndex(this.getIndex()+n.getIndex()+1);
		// reassign all references
		// if the current node was the root, the right child node will become the new root
		if (this.getParent()!=null){
			if (this.getParent().getLeft()==this){
				this.getParent().setLeft(n);
			}
			else {
				this.getParent().setRight(n);
			}
		}
		else {
			n.setParent(null);
		}
		this.setRight(n.getLeft());
		n.setLeft(this);
	}
	
	private void rotate_right(){
		// the current child node
		WRBTreeNode n = this.getLeft();
		// reassign indices and weights... it took some time to realize this subtraction is the only solution for the weights... it took sooo long...
		this.setWeight_left(this.getWeight_left()-n.getWeight_left()-n.getWeight().getValue());
		this.setIndex(this.getIndex()-n.getIndex()-1);
		// reassign all references
		if (this.getParent()!=null){
			if (this.getParent().getLeft()==this){
				this.getParent().setLeft(n);
			}
			else {
				this.getParent().setRight(n);
			}
		}
		else {
			n.setParent(null);
		}
		this.setLeft(n.getRight());
		n.setRight(this);
	}
	
	// returns the node which contains a specific input pair
	// is null if the pair isn't in the tree
	public WRBTreeNode search(SortPair w){
		// if the current node is a leaf node then w is not contained
		if (this.weight==null){
			return null;
		}
		// compare for left and right sides
		else if (weight.compareTo(w)>0){
			return left.search(w);
		}
		else if (weight.compareTo(w)<0){
			return right.search(w);
		}
		// otherwise we have found the node we were looking for
		else {
			return this;
		}
	}
	
	// content substitution function for single children nodes
	private boolean replace_node(){
		// java... why to you force non-null initializations when variables are
		// defined in an if-else statement...
		// c is a non-leaf (if existing) child node
		WRBTreeNode c = new WRBTreeNode();
		if (right.getWeight()==null){
			c = left;
		}
		else {
			c = right;
		}
		boolean b = color;
		// copy all values from c
		weight = c.getWeight();
		WRBTreeNode t = c.getLeft();
		c.setLeft(null);
		setLeft(t);
		t = c.getRight();
		c.setRight(null);
		setRight(t);
		color = c.getColor();
		weight_left = 0;
		index = 0;
		// get rid of all those dangling pointers!!!
		c.setWeight(null);
		c.setParent(null);
		return b;
	}
	
	// remove the node containing w from the tree
	public void delete(SortPair w){
		WRBTreeNode tn = search(w);
		// we only delete if w is in the tree
		if (tn!=null){
			WRBTreeNode t = tn;
			// adjust all top tree nodes indices
			while (t.getParent()!=null){
				if (t==t.getParent().getLeft()){
					// the indices are only updated for all nodes were the path to root was a left path
					t.getParent().setWeight_left(t.getParent().getWeight_left()-w.getValue());
					t.getParent().setIndex(t.getParent().getIndex()-1);
				}
				t = t.getParent();
			}
			tn.delete();
		}
	}
	
	// once the node to be deleted has been found, apply this function
	public void delete(){
		// node contains two non-leaf children
		if ((left.getWeight()!=null)&&(right.getWeight()!=null)){
			// find the immediate predecessor
			WRBTreeNode p = left;
			while (p.getRight().getWeight()!=null){
				p = p.getRight();
			}
			// we only move the node content
			weight = p.getWeight();
			// adjust the weight of the tree after the reinsertion
			weight_left-=weight.getValue();
			index--;
			// and return to deleting the predecessor node
			p.delete();
		}
		// once we get to the final node we have to perform all balance checks...
		else {
			delete_one_child();
		}
	}
	
	// the tree might need rebalancing after we remove a node with at most one non-leaf child
	private void delete_one_child(){
		boolean b = this.replace_node();
		if (!b){
			if (color){
				color = false;
			}
			else {
				delete_case1();
			}
		}
	}
	
	// rebalancing function chain
	// this function needs to be public since it is once called on a other node
	public void delete_case1(){
		// nothing to do for the root node
		// otherwise
		if (parent!=null){
			delete_case2();
		}
	}
	
	private void delete_case2(){
		// sibling node
		WRBTreeNode s = getSibling();
		if (s.getColor()){
			parent.setColor(true);
			s.setColor(false);
			if (this==parent.getLeft()){
				this.parent.rotate_left();
			}
			else {
				this.parent.rotate_right();
			}
		}
		delete_case3();
	}
	
	private void delete_case3(){
		// sibling node
		WRBTreeNode s = getSibling();
		// if the current "family" is black then we have to do some rebalancing
		if (!parent.getColor()&&!s.getColor()&&!s.getLeft().getColor()&&!s.getRight().getColor()){
			s.setColor(true);
			parent.delete_case1();
		}
		else {
			delete_case4();
		}
	}
	
	private void delete_case4(){
		// sibling node
		WRBTreeNode s = getSibling();
		if (parent.getColor()&&!s.getColor()&&!s.getLeft().getColor()&&!s.getRight().getColor()){
			s.setColor(true);
			parent.setColor(false);
		}
		else {
			delete_case5();
		}
	}
	
	private void delete_case5(){
		// sibling node
		WRBTreeNode s = getSibling();
		if (!s.getColor()){
			if ((this==parent.getLeft())&&s.getLeft().getColor()&&!s.getRight().getColor()){
				s.setColor(true);
				s.getLeft().setColor(false);
				s.rotate_right();
			}
			else if ((this==parent.getRight())&&!s.getLeft().getColor()&&s.getRight().getColor()){
				s.setColor(true);
				s.getRight().setColor(false);
				s.rotate_left();
			}
		}
		delete_case6();
	}
	
	private void delete_case6(){
		// sibling node
		WRBTreeNode s = getSibling();
		s.setColor(parent.getColor());
		parent.setColor(false);
		if (this==parent.getLeft()){
			s.getRight().setColor(false);
			this.parent.rotate_left();
		}
		else {
			s.getLeft().setColor(false);
			this.parent.rotate_right();
		}
	}
	
	// this function is for random access from the tree element keys
	// there will be problems if ind<0 or ind> the number of elements in tree -1
	public SortPair access(int ind){
		// we search the left subtree if there are still more elements than 
		// ind elements in left subtree
		if (ind<index){
			return left.access(ind);
		}
		// search the right subtree if the ind exceed the current index
		else if(ind>index){
			// and subtract away the current index from ind
			return right.access(ind-(index+1));
		}
		// otherwise we have found the right index and can return the weight key
		else {
			return weight;
		}
	}
	
	// access a key according to the the distributions of weights in the tree
	// this is the reason to implement a weighted R-B-Tree
	// w must!!! be >=0 and < wt (the total weight of the tree)
	public SortPair weighted_access(double w){
		// similar to the standard access function
		if (w<weight_left){
			return left.weighted_access(w);
		}
		else if (w>weight_left+weight.getValue()){
			return right.weighted_access(w-(weight_left+weight.getValue()));
		}
		else {
			return weight;
		}
	}
}
			
				
				