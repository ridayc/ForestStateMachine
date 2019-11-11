package flib.algorithms.randomforest;

import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;
import com.joptimizer.functions.PSDQuadraticMultivariateRealFunction;
import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.LinearMultivariateRealFunction;
import com.joptimizer.optimizers.OptimizationRequest;
import com.joptimizer.optimizers.JOptimizer;
import flib.math.VectorFun;
import flib.math.SortPair2;
import flib.math.RankSort;
import flib.math.random.Shuffle;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.DecisionTree;
import flib.algorithms.randomforest.TreeNode;
import flib.algorithms.randomforest.DecisionNode;

import gurobi.*;

public class ForestFunctions {
	// function to train the weights assigned to the individual trees of a random forest
	public static double[] weightForests(final RandomForest[] rf, final double[][] tset, final double[] labels, double[] weights){
		// we're given:
		// rf: the (likely) unweighted Random Forest
		// tset: an independent traingset/test set to evaluate the trees
		// labels: true labels for the current set of points
		// we need to create the vote matrix which contains the collection of votes
		// based on the trees
		int nfor = rf.length;
		//int n = rf.getTrainingset().length;
		int n = tset.length;
		// nclass==1 will imply the usage of a regression forest
		// hopefully all regression forests set this value to 1
		int nclass = rf[0].getNumClasses();
		// vote matrix
		double[][] votes = new double[nfor][nclass*n];
		for (int i=0; i<nfor; i++){
			double[][] v = rf[i].applyForest(tset);
			for (int j=0; j<n; j++){
				for (int k=0; k<nclass; k++){
					votes[i][k*n+j] = v[j][k];
				}
			}
		}
		// we start preparing for the quadratic optimization problem for the weights
		// we need a square matrix
		double[][] V = new double[nfor][nfor];
		// we need a offset vector
		double[] t = new double[nfor];
		double r = 0;
		// generate the matrix and vector content
		for (int i=0; i<nfor; i++){
			for (int j=0; j<nfor; j++){
				// 'correlation' between forests
				for (int k=0; k<nclass*n; k++){
					V[i][j]+=votes[i][k]*votes[j][k]*weights[k%n];
				}
			}
			if (nclass>1){
				for (int k=0; k<nclass; k++){
					for (int l=0; l<n; l++){
						// 'correlation' with the labels
						if (labels[l]==k){
							t[i]-=votes[i][k*n+l]*weights[l];
						}
					}
				}
			}
			else {
				for (int k=0; k<n; k++){
					// 'correlation' with the labels
					t[i]-=votes[i][k]*labels[k]*weights[k];
				}
			}
		}
		// now we're ready for the quadratic optimization
		PSDQuadraticMultivariateRealFunction objectiveFunction = new PSDQuadraticMultivariateRealFunction(V,t,n);
		// equalities
		double[][] A = new double[1][nfor];
		VectorFun.addi(A[0],1);
		double[] b = new double[1];
		b[0] = 1;
		// inequalities
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[nfor];
		for (int i=0; i<nfor; i++){
			double[] temp = new double[nfor];
			temp[i] = -1;
			inequalities[i] = new LinearMultivariateRealFunction(temp.clone(),0);
		}
		double[] iw = VectorFun.add(new double[nfor],1./nfor);
		// optimization problem
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		//System.out.println("Initial weight sum: "+Double.toString(VectorFun.sum(iw)));
		or.setInitialPoint(iw);
		or.setFi(inequalities);
		or.setA(A);
		or.setB(b);
		or.setToleranceFeas(1.E-8);
		or.setTolerance(1.E-8);
		//optimization
		JOptimizer opt = new JOptimizer();
		opt.setOptimizationRequest(or);
		int returnCode = 0;
		try {
			returnCode = opt.optimize();
		}
		catch (Throwable thr){
			System.out.println("Problem with the quadratic optimization");
			thr.printStackTrace();
		}
		//System.out.println(Integer.toString(returnCode));
		double[] w = opt.getOptimizationResponse().getSolution();
		//System.out.println(Arrays.toString(w));
		return w;
	}
	
	// function to train the weights assigned to the individual trees of a random forest
	public static double weightTrees(RandomForest rf, final double[][] tset, final double[] labels, double[] weights){
		// we're given:
		// rf: the (likely) unweighted Random Forest
		// tset: an independent traingset/test set to evaluate the trees
		// labels: true labels for the current set of points
		// we need to create the vote matrix which contains the collection of votes
		// based on the trees
		int ntree = rf.getNtree();
		//int n = rf.getTrainingset().length;
		int n = tset.length;
		// nclass==1 will imply the usage of a regression forest
		// hopefully all regression forests set this value to 1
		int nclass = rf.getNumClasses();
		// vote matrix
		double[][] votes = new double[ntree][nclass*n];
		for (int i=0; i<ntree; i++){
			for (int j=0; j<n; j++){
				double[] v = rf.getTrees().get(i).applyTree(tset[j]);
				for (int k=0; k<nclass; k++){
					votes[i][k*n+j] = v[k];
				}
			}
		}
		// we start preparing for the quadratic optimization problem for the weights
		// we need a square matrix
		double[][] V = new double[ntree][ntree];
		// we need a offset vector
		double[] t = new double[ntree];
		double r = 0;
		// generate the matrix and vector content
		for (int i=0; i<ntree; i++){
			for (int j=0; j<ntree; j++){
				// 'correlation' between trees
				for (int k=0; k<nclass*n; k++){
					V[i][j]+=votes[i][k]*votes[j][k]*weights[k%n];
				}
			}
			if (nclass>1){
				for (int k=0; k<nclass; k++){
					for (int l=0; l<n; l++){
						// 'correlation' with the labels
						if (labels[l]==k){
							t[i]-=votes[i][k*n+l]*weights[l];
						}
					}
				}
			}
			else {
				for (int k=0; k<n; k++){
					// 'correlation' with the labels
					t[i]-=votes[i][k]*labels[k]*weights[k];
				}
			}
		}
		// now we're ready for the quadratic optimization
		PSDQuadraticMultivariateRealFunction objectiveFunction = new PSDQuadraticMultivariateRealFunction(V,t,n);
		// equalities
		double[][] A = new double[1][ntree];
		VectorFun.addi(A[0],1);
		double[] b = new double[1];
		b[0] = 1;
		// inequalities
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[ntree];
		for (int i=0; i<ntree; i++){
			double[] temp = new double[ntree];
			temp[i] = -1;
			inequalities[i] = new LinearMultivariateRealFunction(temp.clone(),0);
		}
		double[] iw = rf.getTree_weights().clone();
		// optimization problem
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		//System.out.println("Initial weight sum: "+Double.toString(VectorFun.sum(iw)));
		or.setInitialPoint(iw);
		or.setFi(inequalities);
		or.setA(A);
		or.setB(b);
		or.setToleranceFeas(1.E-8);
		or.setTolerance(1.E-8);
		//optimization
		JOptimizer opt = new JOptimizer();
		opt.setOptimizationRequest(or);
		int returnCode = 0;
		try {
			returnCode = opt.optimize();
		}
		catch (Throwable thr){
			System.out.println("Problem with the quadratic optimization");
			thr.printStackTrace();
		}
		//System.out.println(Integer.toString(returnCode));
		double[] w = opt.getOptimizationResponse().getSolution();
		//System.out.println(Arrays.toString(w));
		rf.setTree_weights(w);
		return returnCode;
	}
	
	public static double weightTrees3(RandomForest rf, final double[][] tset, final double[] labels, final double[] weights){
		// we're given:
		// rf: the (likely) unweighted Random Forest
		// tset: an independent traingset/test set to evaluate the trees
		// labels: true labels for the current set of points
		// we need to create the vote matrix which contains the collection of votes
		// based on the trees
		int ntree = rf.getNtree();
		//int n = rf.getTrainingset().length;
		int n = tset.length;
		// nclass==1 will imply the usage of a regression forest
		// hopefully all regression forests set this value to 1
		int nclass = rf.getNumClasses();
		int[] ts = rf.getTreeSizes();
		int nl = VectorFun.sum(ts);
		int l = nclass*nl;
		double[] w = rf.getTree_weights();
		int[] offset = VectorFun.sub(VectorFun.cumsum(ts),ts);
		// vote matrix
		double[][] V = new double[l][l];
		double[] t = new double[l];
		double[][] votes = rf.applyForest(tset);
		int[][] loc = rf.getLeafIndices(tset);
		double[] initial_weights = new double[l];
		// go through all points
		if (nclass>1){
			for (int i=0; i<n; i++){
				for (int c=0; c<nclass; c++){
					for (int j=0; j<ntree; j++){
						int x = nclass*(offset[j]+loc[i][j])+c;
						if ((int)labels[i]==c){
							t[x]-=w[j]*weights[i];
						}
						for (int k=0; k<ntree; k++){
							int y = nclass*(offset[k]+loc[i][k])+c;
							V[x][y]+=w[j]*w[k]*weights[i];
						}
					}
				}
			}
		}
		else {
			for (int i=0; i<n; i++){
				for (int c=0; c<nclass; c++){
					for (int j=0; j<ntree; j++){
						int x = nclass*(offset[j]+loc[i][j])+c;
						if ((int)labels[i]==c){
							t[x]-=w[j]*labels[i]*weights[i];
						}
						for (int k=0; k<ntree; k++){
							int y = nclass*(offset[k]+loc[i][k])+c;
							V[x][y]+=w[j]*w[k]*weights[i];
						}
					}
				}
			}
		}
		//System.out.println(Arrays.toString(t));
		//initial votes
		double[] iw = new double[l];
		for (int i=0; i<ntree; i++){
			ArrayList<TreeNode<DecisionNode>> leaf = rf.getTrees().get(i).leaves();
			for (int j=0; j<ts[i]; j++){
				double[] v2 = leaf.get(ts[i]-1-j).getData().getProbabilities();
				for (int c=0; c<nclass; c++){
					int x = nclass*(offset[i]+j)+c;
					iw[x] = v2[c];
				}
			}
		}
		//System.out.println(Arrays.toString(iw));
		// now we're ready for the quadratic optimization
		PSDQuadraticMultivariateRealFunction objectiveFunction = new PSDQuadraticMultivariateRealFunction(V,t,n);
		// equalities
		double[][] A = new double[0][0];
		double[] b = new double[0];
		if (nclass>1){
			A = new double[nl][l];
			for (int i=0; i<nl; i++){
				for (int c=0; c<nclass; c++){
					A[i][nclass*i+c] = 1;
				}
			}
			b = VectorFun.add(new double[nl],1);
		}
		// inequalities
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[0];
		if (nclass>1){
			inequalities = new ConvexMultivariateRealFunction[l];
			for (int i=0; i<l; i++){
				double[] temp = new double[l];
				temp[i] = -1;
				inequalities[i] = new LinearMultivariateRealFunction(temp.clone(),-1.E-8);
			}
		}
		// optimization problem
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		//System.out.println("Initial weight sum: "+Double.toString(VectorFun.sum(iw)));
		or.setInitialPoint(iw);
		or.setFi(inequalities);
		or.setA(A);
		or.setB(b);
		or.setToleranceFeas(1.E-8);
		or.setTolerance(1.E-8);
		//optimization
		JOptimizer opt = new JOptimizer();
		opt.setOptimizationRequest(or);
		int returnCode = 0;
		try {
			returnCode = opt.optimize();
		}
		catch (Throwable thr){
			System.out.println("Problem with the quadratic optimization");
			thr.printStackTrace();
		}
		//System.out.println(Integer.toString(returnCode));
		double[] res = opt.getOptimizationResponse().getSolution();
		//System.out.println(Arrays.toString(w));
		for (int i=0; i<ntree; i++){
			ArrayList<TreeNode<DecisionNode>> leaf = rf.getTrees().get(i).leaves();
			for (int j=0; j<ts[i]; j++){
				//System.out.println(Integer.toString(leaf.get(ts[i]-1-j).getData().getLeafIndex()));
				double[] v2 = leaf.get(ts[i]-1-j).getData().getProbabilities();
				for (int c=0; c<nclass; c++){
					int x = nclass*(offset[i]+j)+c;
					v2[c] = res[x];
				}
			}
		}
		return returnCode;
	}
	
	public static void weightTrees4(RandomForest rf, final double[][] tset, final double[] labels, final double[] weights, int maxit){
		// we're given:
		// rf: the (likely) unweighted Random Forest
		// tset: an independent traingset/test set to evaluate the trees
		// labels: true labels for the current set of points
		// we need to create the vote matrix which contains the collection of votes
		// based on the trees
		int ntree = rf.getNtree();
		//int n = rf.getTrainingset().length;
		int n = tset.length;
		// nclass==1 will imply the usage of a regression forest
		// hopefully all regression forests set this value to 1
		int nclass = rf.getNumClasses();
		int[] ts = rf.getTreeSizes();
		int nl = VectorFun.sum(ts);
		int l = nclass*nl;
		double[] w = rf.getTree_weights();
		int[] offset = VectorFun.sub(VectorFun.cumsum(ts),ts);
		double[] t = new double[l];
		double[][] votes = rf.applyForest(tset);
		int[][] loc = rf.getLeafIndices(tset);
		//initial votes
		double[] iw = new double[l];
		double[] iwa = new double[l];
		for (int i=0; i<ntree; i++){
			ArrayList<TreeNode<DecisionNode>> leaf = rf.getTrees().get(i).leaves();
			for (int j=0; j<ts[i]; j++){
				double[] v2 = leaf.get(ts[i]-1-j).getData().getProbabilities();
				for (int c=0; c<nclass; c++){
					int x = nclass*(offset[i]+j)+c;
					iw[x] = v2[c];
				}
			}
		}
		// run multiple times
		for (int iter=0; iter<maxit; iter++){
			int nt = (int)((1-1./Math.exp(1))*n);
			int[] rp = Shuffle.randPerm(n);
			// set up the matrix problem
			ArrayList<TreeSet<SortPair2>> mat = new ArrayList<TreeSet<SortPair2>>();
			for (int i=0; i<l; i++){
				mat.add(new TreeSet<SortPair2>());
			}
			// gurobi setup
			try {
				double lower_bound = 0;
				double upper_bound = 1;
				if (nclass==1){
					lower_bound = VectorFun.min(iw)[0];
					upper_bound = VectorFun.max(iw)[0];
				}
				GRBEnv env = new GRBEnv("qp.log");
				GRBModel model = new GRBModel(env);
				// variable setup
				GRBVar[] vars = new GRBVar[l];
				for (int i=0; i<l; i++){
					vars[i] = model.addVar(lower_bound,upper_bound,iw[i],GRB.CONTINUOUS,"");
				}
				model.update();
				// prepare the symmetric system matrix
				// go through all points
				// first round to find and increment all sparse locations
				for (int i2=0; i2<nt; i2++){
					int i = rp[i2];
					for (int c=0; c<nclass; c++){
						for (int j=0; j<ntree; j++){
							int x = nclass*(offset[j]+loc[i][j])+c;
							if (nclass>1){
								if ((int)labels[i]==c){
									t[x]-=2*w[j]*weights[i];
								}
							}
							else {
								t[x]-=2*w[j]*labels[i]*weights[i];
							}
							for (int k=0; k<ntree; k++){
								int y = nclass*(offset[k]+loc[i][k])+c;
								SortPair2 sp = new SortPair2(y,w[j]*w[k]*weights[i]);
								if (!mat.get(x).add(sp)){
									mat.get(x).floor(sp).setOriginalIndex(mat.get(x).floor(sp).getOriginalIndex()+w[j]*w[k]*weights[i]);
								}
							}
						}
					}
				}
				if (nclass>1){
					// vote sum constraints
					for (int i=0; i<nl; i++){
						GRBLinExpr expr = new GRBLinExpr();
						for (int c=0; c<nclass; c++){
							expr.addTerm(1,vars[i*nclass+c]);
						}
						model.addConstr(expr,GRB.EQUAL,1,"");
					}
					// add all matrix values
				}
				// set the objective... aka the input matrix and input vector
				GRBQuadExpr obj = new GRBQuadExpr();
				for (int i=0; i<l; i++){
					// linear terms
					if (t[i]!=0){
						obj.addTerm(t[i],vars[i]);
					}
					// go through all non-zero values of the matrix
					Iterator<SortPair2> it = mat.get(i).iterator();
					while (it.hasNext()){
						SortPair2 sp = it.next();
						obj.addTerm(sp.getOriginalIndex(),vars[i],vars[(int)sp.getValue()]);
					}
				}
				model.setObjective(obj);
				// Solve
				model.optimize();
				if (model.get(GRB.IntAttr.Status)!=GRB.Status.OPTIMAL) {
					System.out.println("Didn't get the optimal solution");
				}
				for (int i=0; i<l; i++){
					iwa[i]+=vars[i].get(GRB.DoubleAttr.X);
				}
				model.dispose();
				env.dispose();
			}
			catch (GRBException e) {
				System.out.println("Error code: " + e.getErrorCode() + ". " +e.getMessage());
				e.printStackTrace();
			}
		}
		VectorFun.multi(iwa,1./maxit);
		//System.out.println(Arrays.toString(iwa));
		for (int i=0; i<ntree; i++){
			ArrayList<TreeNode<DecisionNode>> leaf = rf.getTrees().get(i).leaves();
			for (int j=0; j<ts[i]; j++){
				//System.out.println(Integer.toString(leaf.get(ts[i]-1-j).getData().getLeafIndex()));
				double[] v2 = leaf.get(ts[i]-1-j).getData().getProbabilities();
				for (int c=0; c<nclass; c++){
					int x = nclass*(offset[i]+j)+c;
					v2[c] = iwa[x];
				}
			}
		}
	}
	
	public static int[] weightTrees2(RandomForest rf, final double[][] tset, final double[] labels, double[] initial_weights, int maxit){
		int[] gap = new int[maxit];
		int ntree = rf.getNtree();
		int n = tset.length;
		int dim = tset[0].length;
		int nclass = rf.getNumClasses();
		double[] iw = initial_weights.clone();
		double[][][] vt = new double[ntree][n][nclass];
		Random rng = new Random();
		double tol = 1.E-3;
		for (int i=0; i<ntree; i++){
			for (int j=0; j<n; j++){
				vt[i][j] = rf.getTrees().get(i).applyTree(tset[j]).clone();
			}
		}
		// each iteration goes through all trees and all points
		for (int it=0; it<maxit; it++){
			int[] rp = Shuffle.randPerm(ntree);
			double[][] votes = new double[n][nclass];
			// votes for all points for the current weight
			// helps stabilize the algorithm
			for (int i=0; i<ntree; i++){
				for (int j=0; j<n; j++){
					VectorFun.addi(votes[j],VectorFun.mult(vt[i][j],iw[i]));
				}
			}
			// cycle through all trees
			for (int i=0; i<ntree; i++){
				// adjust the weights of the leftover set
				double w = 1./(1-iw[rp[i]]);
				if(Double.isInfinite(w)){
					System.out.println("One weight to rule them all...");
				}
				double[] intervals = new double[2*n];
				// interval for each point
				for (int j=0; j<n; j++){
					double[] v1 =vt[rp[i]][j];
					double[] v2 = VectorFun.mult(VectorFun.sub(votes[j],VectorFun.mult(v1,iw[rp[i]])),w);
					/*
					if (VectorFun.sum(v1)>1+tol||VectorFun.sum(v1)<1-tol||VectorFun.sum(v2)>1+tol||VectorFun.sum(v2)<1-tol){
						System.out.println("Imprecision");
					System.out.println("v1: "+Arrays.toString(v1));
					System.out.println("v2: "+Arrays.toString(v2));
					}
					*/
					// initial interval for this point
					double s1 = 0;
					double s2 = 1;
					int l = (int)labels[j];
					// go through all classes to see how the single and total votes compare
					for (int k=0; k<nclass; k++){
						// checking against all other classes
						if (k!=l){
							// find the current intersection point
							double al = v1[l]-v2[l];
							double ak = v1[k]-v2[k];
							double its = (v2[k]-v2[l])/(al-ak);
							// case where the lines are parallel
							if(Double.isNaN(its)||Math.abs(al-ak)<tol){
								//System.out.println("parallel lines");
								if (v1[l]<=v1[k]){
									s1 = 1.1;
									s2 = -0.1;
									break;
								}
							}
							else {
								// if the label curve is steeper we look on the right of the intersect
								if (al>ak){
									if (its>=s1){
										s1 = its;
									}
								}
								// take all values larger than the intersect
								else {
									if (its<=s2){
										s2 = its;
									}
								}
							}
						}
					}
					// intervals are disregarded if the starting point is larger than the end point
					if (s1>=s2){
						s1 = 1.1;
						s2 = -0.1;
					}
					// list the start and the stop points of all intervals
					intervals[2*j] = s1;
					intervals[2*j+1] = s2;
				}
				//System.out.println(Arrays.toString(intervals));
				// sort all the interval start and end points
				RankSort rs = new RankSort(intervals);
				double[] s = rs.getSorted();
				int[] o = rs.getRank();
				// count all unique locations between 0 and 1
				int count = 0;
				if (s[0]>=0){
					count = 1;
				}
				for (int j=1; j<s.length; j++){
					if(s[j]>s[j-1]&&s[j]>=0&&s[j]<=1){
						count++;
					}
				}
				// number of points in the interval right after this starting point
				int[] counter = new int[count];
				// location where the interval starts
				double[] loc = new double[count];
				count = -1;
				if (s[0]>=0&&s[0]<=1){
					count = 0;
					loc[count] = s[0];
					// if starting event
					if (o[0]%2==0){
						counter[count]++;
					}
					// if stopping event... which shouldn't be possible to reach at this point
					// except maybe if the is a start and stop at zero...
					else {
						counter[count]--;
					}
				}
				// go through all interval points
				for (int j=1; j<s.length; j++){
					//ignore all points below 0
					if (s[j]>=0&&s[j]<=1){
						if (s[j]>s[j-1]){
							count++;
							if (count>0){
								// copy the previous count value
								counter[count] = counter[count-1];
							}
						}
						loc[count] = s[j];
						// if starting event
						if (o[j]%2==0){
							counter[count]++;
						}
						// if stopping event
						else {
							counter[count]--;
						}
					}
				}
				/*
				System.out.println("Intervals: "+Arrays.toString(intervals));
				System.out.println("sorted: "+Arrays.toString(s));
				System.out.println("rank: "+Arrays.toString(o));
				System.out.println("counter: "+Arrays.toString(counter));
				System.out.println("loc: "+Arrays.toString(loc));
				//System.out.println(Arrays.toString(counter));
				//System.out.println(Arrays.toString(loc));
				*/
				// find all intervals which have a maximum number of points
				ArrayList<Integer> a = new ArrayList<Integer>();
				int max = 0;
				for (int j=0; j<counter.length; j++){
					if (counter[j]==max){
						a.add(j);
					}
					if (counter[j]>max){
						a = new ArrayList<Integer>();
						a.add(j);
						max = counter[j];
					}
				}
				// new weight for this tree 
				double w2 = 0;
				if (a.size()>0){
					// choose a random interval start from the max interval starts
					int b = a.get(rng.nextInt(a.size()));
					if (b==loc.length-1){
						// this will still cause problems at the moment
						System.out.println("My name is... prepare to die");
						w2 = 1;
					}
					else {
						w2 = loc[b]+rng.nextDouble()*(loc[b+1]-loc[b]);
						//System.out.println(Double.toString(w2));
					}
					//gap[it] = n-counter[b];
					gap[it] = n-max;
				}
				else {
					System.out.println(Arrays.toString(intervals));
				}
				double ww = (1.-w2)/(1.-iw[rp[i]]);
				// adjust the votes for all points
				for (int j=0; j<n; j++){
					// subtract the old votes for this point
					VectorFun.subi(votes[j],VectorFun.mult(vt[rp[i]][j],iw[rp[i]]));
					// readjust the total votes according to the lost weight
					VectorFun.multi(votes[j],ww);
					// add the effect of this tree to this point's vote
					VectorFun.addi(votes[j],VectorFun.mult(vt[rp[i]][j],w2));
				}
				// adjust all the tree weights
				VectorFun.multi(iw,ww);
				iw[rp[i]] = w2;
				//System.out.println("weights: "+Arrays.toString(iw));
				//System.out.println("weight sum: "+Double.toString(VectorFun.sum(w2)));
			}
			
		}
		rf.setTree_weights(iw);
		return gap;
	}
	
	public static int removeTrees(RandomForest rf, double quantile){
		int ntree = rf.getNtree();
		int n = rf.getTrainingset().length;
		// get a sorted list of the tree weights
		RankSort rs = new RankSort(rf.getTree_weights());
		int[] o = rs.getRank();
		double[] s = rs.getSorted();
		// start sum the weights
		int ind = 0;
		double w = 0;
		for (int i=0; i<ntree; i++){
			w+=s[i];
			if (w>quantile){
				break;
			}
			else {
				ind++;
			}
		}
		// start removing trees
		if (ind>0){
			double[] temp_weight = new double[ntree-ind];
			int[][] temp_samp = new int[ntree-ind][n];
			int[][] samp = rf.getSamples();
			ArrayList<DecisionTree> forest = rf.getTrees();
			ArrayList<DecisionTree> temp_forest = new ArrayList<DecisionTree>();
			for (int i=0; i<ntree-ind; i++){
				temp_weight[i] = s[ntree-1-i];
				temp_samp[i] = samp[o[ntree-1-i]].clone();
				temp_forest.add(forest.get(o[ntree-1-i]));
			}
			rf.setTree_weights(VectorFun.div(temp_weight,VectorFun.sum(temp_weight)));
			rf.setSamples(temp_samp);
			rf.setTrees(temp_forest);
			rf.setNtree(rf.getTrees().size());
		}
		return rf.getNtree();
	}
	
	public static int removeTrees(RandomForest rf, int ind){
		int ntree = rf.getNtree();
		int n = rf.getTrainingset().length;
		// get a sorted list of the tree weights
		RankSort rs = new RankSort(rf.getTree_weights());
		int[] o = rs.getRank();
		double[] s = rs.getSorted();
		// start sum the weights
		// start removing trees
		if (ind>ntree){
			ind = ntree;
		}
		if (ind>0){
			double[] temp_weight = new double[ind];
			int[][] temp_samp = new int[ind][n];
			int[][] samp = rf.getSamples();
			ArrayList<DecisionTree> forest = rf.getTrees();
			ArrayList<DecisionTree> temp_forest = new ArrayList<DecisionTree>();
			for (int i=0; i<ind; i++){
				temp_weight[i] = s[ntree-1-i];
				temp_samp[i] = samp[o[ntree-1-i]].clone();
				temp_forest.add(forest.get(o[ntree-1-i]));
			}
			rf.setTree_weights(VectorFun.div(temp_weight,VectorFun.sum(temp_weight)));
			rf.setSamples(temp_samp);
			rf.setTrees(temp_forest);
			rf.setNtree(rf.getTrees().size());
		}
		return rf.getNtree();
	}
	
	public static RandomForest mergeForests(final RandomForest[] rfs){
		int l = rfs.length;
		// copy the first forest as a reference
		RandomForest rf = new RandomForest();
		// prepare for problems if the forests weren't trained on the same data sets
		rf.copyForestContent(rfs[0]);
		int ntree = 0;
		for (int i=0; i<l; i++){
			ntree+=rfs[i].getNtree();
		}
		double[] temp_weight = VectorFun.add(new double[ntree],1./ntree);
		int[][] temp_samp = new int[ntree][];
		ArrayList<DecisionTree> temp_forest = new ArrayList<DecisionTree>();
		int count = 0;
		for (int i=0; i<l; i++){
			for (int j=0; j<rfs[i].getNtree(); j++){
				temp_samp[count+j] = rfs[i].getSamples()[j];
				// temp_weight[count+j] = rfs[i].getTree_weights()[j];
				temp_forest.add(rfs[i].getTrees().get(j));
			}
			count+=rfs[i].getNtree();
		}
		rf.setTree_weights(VectorFun.div(temp_weight,VectorFun.sum(temp_weight)));
		rf.setSamples(temp_samp);
		rf.setTrees(temp_forest);
		rf.setNtree(rf.getTrees().size());
		return rf;
	}
	
	public static void mergeForests(final RandomForest[] rfs, RandomForest rf){
		int l = rfs.length;
		// prepare for problems if the forests weren't trained on the same data sets
		rf.copyForestContent(rfs[0]);
		int ntree = 0;
		for (int i=0; i<l; i++){
			ntree+=rfs[i].getNtree();
		}
		double[] temp_weight = VectorFun.add(new double[ntree],1./ntree);
		int[][] temp_samp = new int[ntree][];
		ArrayList<DecisionTree> temp_forest = new ArrayList<DecisionTree>();
		int count = 0;
		for (int i=0; i<l; i++){
			for (int j=0; j<rfs[i].getNtree(); j++){
				temp_samp[count+j] = rfs[i].getSamples()[j];
				// temp_weight[count+j] = rfs[i].getTree_weights()[j];
				temp_forest.add(rfs[i].getTrees().get(j));
			}
			count+=rfs[i].getNtree();
		}
		rf.setTree_weights(VectorFun.div(temp_weight,VectorFun.sum(temp_weight)));
		rf.setSamples(temp_samp);
		rf.setTrees(temp_forest);
		rf.setNtree(rf.getTrees().size());
	}
	
	public static double[][] randomizedSet(final double[][] set, final boolean[] categorical, int times, double noise){
		int n = set.length;
		int dim = set[0].length;
		double[][] trainingset = new double[n*times][dim];
		// Gaussian halfwidth for the rank noise
		Random rng = new Random();
		// go through all dimensions
		for (int i=0; i<dim; i++){
			// don't put noise into categorical dimensions
			if (!categorical[i]){
				double[] val = new double[n];
				for (int j=0; j<n; j++){
					val[j] = set[j][i];
				}
				// sort the input dimension
				RankSort rs = new RankSort(val);
				double[] s = rs.getSorted();
				int[] o = rs.getRank();
				for (int j=0; j<times; j++){
					for (int k=0; k<n; k++){
						int r = (int)Math.round((rng.nextGaussian()*noise+k));
						while(0>r||r>=n){
							r = (int)Math.round((rng.nextGaussian()*noise+k));
						}
						trainingset[o[k]+j*n][i] = s[r];
					}
				}
			}
			else {
				for (int j=0; j<times; j++){
					for (int k=0; k<n; k++){
						if (rng.nextDouble()<noise/n){
							trainingset[k+j*n][i] = set[rng.nextInt(n)][i];
						}
						else {
							trainingset[k+j*n][i] = set[k][i];
						}
					}
				}
			}
		}
		return trainingset;
	}
}		