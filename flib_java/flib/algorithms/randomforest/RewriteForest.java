package flib.algorithms.randomforest;

import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;
import flib.algorithms.randomforest.DecisionTree;
import flib.algorithms.randomforest.RandomForest;

public class RewriteForest {
	public static DecisionTree rewriteSortedTree(final DecisionTree dt, final double[][] sorted_ts){
		DecisionTree new_dt = new DecisionTree();
		TreeNode<DecisionNode> new_root = new_dt.getRoot();
		TreeNode<DecisionNode> old_root = dt.getRoot();
		new_dt.setLeafIndex(dt.getLeafIndex());
		treeRecursion(new_root,old_root,sorted_ts);
		return new_dt;
	}
	
	private static void treeRecursion(TreeNode<DecisionNode> new_node, TreeNode<DecisionNode> old_node, final double[][] sorted_ts){
		DecisionNode new_content = new DecisionNode(), old_content = old_node.getData();
		// DeepCopy of the old node
		int dim = old_content.getDim();
		new_content.setDim(dim);
		new_content.setCategorical(old_content.getCategorical());
		new_content.setLeafIndex(old_content.getLeafIndex());
		// adjust the new splitting location
		// the non categorical way
		if (!old_content.getCategorical()){
			new_content.setSplit((sorted_ts[(int)(old_content.getSplit()*sorted_ts.length)][dim]+sorted_ts[(int)(old_content.getSplit()*sorted_ts.length)+1][dim])*0.5);
		}
		// the categorical way... remain the same
		else {
			new_content.setSplit(old_content.getSplit());
		}
		if (old_content.isLeafNode()){
			new_content.setLeaf(old_content.getProbabilities().clone());
			new_content.setPoints(old_content.getPoints().clone());
			new_content.setWeight(old_content.getWeight());
			new_node.setData(new_content);
		}
		else {
			new_node.setData(new_content);
			new_node.setLeft(new TreeNode<DecisionNode>());
			new_node.setRight(new TreeNode<DecisionNode>());
			treeRecursion(new_node.getLeft(),old_node.getLeft(),sorted_ts);
			treeRecursion(new_node.getRight(),old_node.getRight(),sorted_ts);
		}
	}
	
	public static RandomForest rewriteForest(final RandomForest rf, final double[][] sorted_ts){
		RandomForest new_rf = new RandomForest();
		new_rf.copyForestContent(rf);
		int ntree = rf.getTrees().size();
		for (int i=0; i<ntree; i++){
			new_rf.getTrees().add(rewriteSortedTree(rf.getTrees().get(i),sorted_ts));
		}
		return new_rf;
	}
}