package flib.algorithms.randomforest.splitfunctions;

import java.lang.Math;
import java.util.Random;
import flib.algorithms.clustering.SpectralClustering;
import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;
import flib.algorithms.randomforest.splitfunctions.ClassificationSplit;
import flib.algorithms.randomforest.splitfunctions.SortingFunctions;
import flib.math.VectorFun;
import flib.math.RankSort;

// This class contains all classes which spectrally cluster all classes in two
// new classes based on the number of zero crossings in and between classes
// extension classes will have to implement the final splitting function
// should implement serializable since SplitFunction does
public abstract class ClusterSplit extends ClassificationSplit {
	// this method gets the class transition matrix
	// initially intended for more than two classes, it can be used in this form to also deal with only two classes
	protected double[][] transitionMatrix(final double[][] loc, int num_class, boolean cat, double m){
		// Best make sure the input weights are integer multiples.
		// Weight values smaller than one will likely lead to unexpected behavior
		
		// initialize the transition matrix
		double[][] tm = new double[num_class][num_class];
		// the location object is build of three equally sized component
		// 1. starting location in the original sorted vector and the original value along the current dimension
		// 2. classes contained at the value
		// 3. weights of the contained classes at the specific value
		int l = loc.length/3;
		// for non-categorical dimensions
		if (!cat){
			// temporary number of classes at a value
			int l0, l1;
			// temporary class indices
			int a,b;
			// temporary variable
			double c;
			// temporary total weight
			double w0,w1;
			// number of present classes at the first specific location
			l0 = loc[l].length;
			// add the weights of the items themselves to the
			// diagonal of the matrix
			for (int i=0; i<l0; i++){
				a = (int)loc[l][i];
				tm[a][a]+=loc[2*l][i]*m;
			}
			// case of a single contained class
			if (l0<=1){
				w0 = loc[2*l][0];
				a = (int)loc[l][0];
				// there is a -1 because we're looking for the number of
				// zero crossing for integer sets
				// obviously the class only transitions into itself
				tm[a][a]+=w0-1;
			}
			// otherwise if there are multiple values contained at a single value
			else {
				w0 = VectorFun.sum(loc[2*l]);
				// nested loop over all classes
				for (int j=0; j<l0; j++){
					a = (int)loc[l][j];
					for (int k=0; k<l0; k++){
						b = (int)loc[l][k];
						// probability of one class * probability of the other class
						// everything is multiplied by the number of points contained -1
						tm[a][b]+=loc[2*l][j]/w0*loc[2*l][k]/w0*(w0-1);
					}
				}
			}
			// number of present classes at a specific location
			for (int i=1; i<l; i++){
				// same steps inside each value as above
				l1 = loc[l+i].length;
				// add the weights of the items themselves to the
				// diagonal of the matrix
				for (int j=0; j<l1; j++){
					a = (int)loc[l+i][j];
					tm[a][a]+=loc[2*l+i][j]*m;
				}
				if (l1<=1){
					w1 = loc[2*l+i][0];
					a = (int)loc[l+i][0];
					tm[a][a]+=w1-1;
				}
				else {
					w1 = VectorFun.sum(loc[2*l+i]);
					for (int j=0; j<l1; j++){
						a = (int)loc[l+i][j];
						for (int k=0; k<l1; k++){
							b = (int)loc[l+i][k];
							tm[a][b]+=2*loc[2*l+i][j]/w1*loc[2*l+i][k]/w1*(w1-1);
						}
					}
				}
				// in the case of a transition
				// transition of single classes
				if ((l0<=1)&&(l1<=1)){
					a = (int)loc[l+i-1][0];
					b = (int)loc[l+i][0];
					// create a symmetric matrix
					tm[a][b]+=0.5;
					tm[b][a]+=0.5;
				}
				// transition of multiple classes
				else {
					for (int j=0; j<l0; j++){
						a = (int)loc[l+i-1][j];
						for (int k=0; k<l1; k++){
							b = (int)loc[l+i][k];
							c = loc[2*l+i-1][j]/w0*loc[2*l+i][k]/w1*0.5;
							tm[a][b]+=c;
							tm[b][a]+=c;
						}
					}
				}
				w0 = w1;
				l0 = l1;
			}
		}
		// otherwise
		// the categorical case is very similar
		else {
			int l0, l1;
			// temporary storage indices
			int a,b;
			// temporary variable
			double c;
			// temporary total weight
			double w0,w1;
			
			// store the total weights of each class
			double[] w = new double[num_class];
			for (int i=0; i<l; i++){
				for (int j=0; j<loc[l+i].length; j++){
					w[(int)loc[l+i][j]]+=loc[2*l+i][j];
				}
			}
			// sum of all weights
			w1 = VectorFun.sum(w);
			// start going through all the categories
			// number of present classes at a specific location
			for (int i=0; i<l; i++){
				l0 = loc[l+i].length;
				w0 = VectorFun.sum(loc[2*l+i]);
				// same procedure as for non-categorical
				// add the weights of the items themselves to the
				// diagonal of the matrix
				for (int j=0; j<l0; j++){
					a = (int)loc[l+i][j];
					tm[a][a]+=loc[2*l+i][j]*m;
				}
				w0 = VectorFun.sum(loc[2*l+i]);
				for (int j=0; j<l0; j++){
					a = (int)loc[l+i][j];
					for (int k=0; k<l0; k++){
						b = (int)loc[l+i][k];
						tm[a][b]+=2*loc[2*l+i][j]/w0*loc[2*l+i][k]/w0*(w0-1);
					}
				}
				// in the case of a transition
				// transition of single classes
				// this case is slightly different for categorical variables
				// here we consider all possible random transitions between categories
				// temporary weights
				double[] tw = new double[num_class];
				for (int j=0; j<l0; j++){
					a = (int)loc[l+i][j];
					tw[a] = loc[2*l+i][j];
				}
				for (int j=0; j<l0; j++){
					a = (int)loc[l+i][j];
					for (int k=0; k<num_class; k++){
						c = loc[2*l+i][j]/w0*(w[k]-tw[k])/(w1-w0)*0.5;
						tm[a][k]+=c;
						tm[k][a]+=c;
					}
				}
			}
		}
		// return a 
		return tm;
	}
	
	@Override public boolean split(TreeNode<DecisionNode> current, final int[] points, final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final int[] dim, final double[] parameters, final double[] splitpurity, int leafindex){
		// go through all proposed dimensions and based on zeros crossing
		// choose one to be the split dimension
		// randomly choose a splitting approach cluster or classic
		Random rng = new Random();
		// for the dimension with the fewest zero crossings
		double[] nzc = new double[2];
		nzc[0] = Double.MAX_VALUE;
		// get the unique labels
		int[] ul = SortingFunctions.uniqueLabels(labels,points);
		int[] ccl = new int[2];
		double val = 0;
		int ncl = 0;
		for (int i=0; i<ul.length; i++){
			if (ncl<ul[i]){
				ncl = ul[i];
			}
		}
		ncl++;
		// cluster zero-crossing based 
		if (rng.nextDouble()>parameters[5]){
			for (int i=0; i<dim.length; i++){
				// find all unique dimension values
				double[][] loc = SortingFunctions.labelList(ul,points,trainingset,weights,dim[i]);
				// find the number of classes
				int l = loc.length/3;
				int[] cc = new int[2];
				cc[0] = 0; cc[1] = 1;
				// in case there are more than two classes
				if (ncl>2){
					// calculate the transitionMatrix
					double[][] tm = transitionMatrix(loc,ncl,categorical[dim[i]],parameters[6]);
					// adjustment/noise injection for the transition matrix
					// this is to produce clusters more balanced in total weight
					double sum = 0;
					for (int k=0; k<tm.length; k++){
						for (int j=0; j<tm.length; j++){
							sum+=tm[k][j];
						}
					}
					sum = Math.sqrt(sum);
					for (int k=0; k<tm.length; k++){
						for (int j=0; j<tm.length; j++){
							tm[k][j]+=sum*parameters[7];
						}
					}
					// get the classes after clustering them into two groups
					cc = new int[tm.length];
					SpectralClustering.cluster(tm,cc);
					//SpectralClustering.cluster(SpectralClustering.convert(tm,0),cc);
					// prepare to count the zero crossings
					loc  = SortingFunctions.convertLabels(loc,cc);
				}
				// calculate the transition matrix a new... or for the first time
				double[][] tm = transitionMatrix(loc,2,categorical[dim[i]],parameters[6]);
				// find the dimension with the minimal number of zero crossings
				if (nzc[0]>tm[0][1]){
					nzc[0] = tm[0][1];
					nzc[1] = dim[i];
					ccl = cc.clone();
				}
			}
			// go and calculate the split on the dimension which had the
			// fewest zero crossings
			double[][] loc = SortingFunctions.labelList(ul,points,trainingset,weights,(int)nzc[1]);
			// labels according to two classes
			loc = SortingFunctions.convertLabels2(loc,ccl);
			// give these labels to a desired splitting function
			val = splitMethod(loc,categorical[(int)nzc[1]]);
		}
		// classical splitting
		else {
			for (int i=0; i<dim.length; i++){
				// find all unique dimension values
				double[][] loc = SortingFunctions.labelList(ul,points,trainingset,weights,dim[i]);
				double[] g = splitMethod2(loc,categorical[dim[i]],ncl);
				if (g[0]<nzc[0]){
					nzc[0] = g[0];
					nzc[1] = dim[i];
					val = g[1];
				}
			}
		}
		int counter = 0;
		if (!categorical[(int)nzc[1]]){
			for (int i=0; i<points.length; i++){
				if (trainingset[points[i]][(int)nzc[1]]<val){
					counter++;
				}
			}
		}
		else {
			for (int i=0; i<points.length; i++){
				if (trainingset[points[i]][(int)nzc[1]]==val){
					counter++;
				}
			}
		}
		if (counter==points.length||counter==0){
			this.setLeaf(current,points,labels,weights,categorical,parameters,leafindex);
			System.out.println("Non-optimal split");
			return false;
		}
		// generate the branching
		DecisionNode branch = new DecisionNode();
		branch.setDim((int)nzc[1]);
		branch.setCategorical(categorical[(int)nzc[1]]);
		branch.setSplit(val);
		current.setData(branch);
		current.setLeft(new TreeNode<DecisionNode>());
		current.setRight(new TreeNode<DecisionNode>());
		return true;
	}
	
	protected abstract double splitMethod(double[][] loc, boolean cat);
	
	protected abstract double[] splitMethod2(double[][] loc, boolean cat, int num_class);
}