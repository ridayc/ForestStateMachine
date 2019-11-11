package flib.math;

import flib.math.SortPair;
import flib.math.WRBTreeNode;

/*
The weighted Red-Black Tree wrapper class
*/

// the wrapper class is responsible to keep a check on
// the total tree weight and it's size for indexing
public class WRBTree implements  java.io.Serializable{
	// the root node of the tree. Some operations might make reference
	// not be the root node anymore. Then the root node needs to recursively be set
	// a simple and quick log(n) operation
	private WRBTreeNode root;
	
	public WRBTree(){
		root = new WRBTreeNode();
	}
	
	// insert a new value pair
	public void insert(SortPair w){
		root.insert(w);
		while (root.getParent()!=null){
			root = root.getParent();
		}
	}
	
	// delete a value pair
	public void delete(SortPair w){
		root.delete(w);
		while (root.getParent()!=null){
			root = root.getParent();
		}
	}
	
	// return a reference to the node containing w
	public WRBTreeNode search(SortPair w){
		return root.search(w);
	}
	
	// random access of the sorted elements
	public SortPair access(int ind){
		return root.access(ind);
	}
	
	// "random access" of elements according to their weight
	public SortPair weighted_access(double w){
		return root.weighted_access(w);
	}
	
	// return a reference to the root of the tree
	// beware that the the root reference might change with insertion and deletion!
	public WRBTreeNode getRoot(){
		return root;
	}
}
			
				
				