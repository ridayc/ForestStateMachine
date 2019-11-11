package flib.algorithms.randomforest;

// This class contains a simple binary tree structure implementation. 
// It can easily be expanded by storing a reference to the parent node
// This is (as can be seen obviously) a template class
public class TreeNode<T> implements
java.io.Serializable {
	// The data structure contained by the tree
	private T data;
	// The list of children nodes
	// Is initialized at class construction
	private TreeNode<T> left = null;
	private TreeNode<T> right = null;
	
	// Add a node to the list of child nodes
	public void setLeft(TreeNode<T> a){
		this.left = a;
	}
	
	public void setRight(TreeNode<T> a){
		this.right = a;
	}
	
	// Set the reference of the node's data object
	public void setData(final T data){
		this.data = data;
	}
	
	// get a reference to the children
	public TreeNode<T> getLeft(){
		return this.left;
	}
	
	public TreeNode<T> getRight(){
		return this.right;
	}
	
	// Get the reference to the node's data
	public T getData(){
		return this.data;
	}
}