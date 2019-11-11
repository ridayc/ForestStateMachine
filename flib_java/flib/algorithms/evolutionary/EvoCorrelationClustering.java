package flib.algorithms.evolutionary;

import java.lang.Math;
import java.util.Random;
import flib.math.VectorFun;
import flib.math.VectorAccess;
import flib.math.random.Shuffle;

public class EvoCorrelationClustering {
	// number of clusters in all sets
	private int k;
	// number of sets
	private int n;
	// block matrix of correlation matrices between sets
	// the first dimension is the location in the correlation block
	// the second dimension is the location in the blockmatrix
	private double[][] CM;
	// mutationrate for each individual
	private double mutrate;
	// strong mutation rate for each individual
	private double mutrate2;
	// the number of iterations where if the fittest individual has the same fittness 
	// the population except the fittest is replaced
	private int eliminationrate;
	// number of times the fitness has stayed the same
	private int counter;
	// previous maximum fitness value
	private double prev;
	// number of individuals in the population
	private int size;
	// the new and the old populations
	private int[][] pop_old, pop_new;
	// fitness of the population
	private double[] pop_fitness;
	// fittest individual of the population
	private double[] fittest;
	// fitness value of the fittest individual of the population throughout all iterations
	private double[] progess;
	
	public EvoCorrelationClustering(final double[][] CM, double mutrate, double mutrate2, int eliminationrate, int size, int numit){
		this.k = (int)Math.sqrt(CM[0].length);
		// size has to be a multiple of 4
		if (size%4!=0){
			size+=4-(size%4);
		}
		this.n = (int)Math.sqrt(CM.length);
		this.CM = new double[n*n][k*k];
		for (int i=0; i<n*n; i++){
			this.CM[i] = CM[i].clone();
		}
		// the mutation rates should be between 0 and 1
		this.mutrate = mutrate;
		this.mutrate = mutrate2;
		this.eliminationrate = eliminationrate;
		this.size = size;
		this.generatePopulation();
		this.counter = 0;
		this.prev = 0;
		this.iterate(numit);
	}
	
	private EvoCorrelationClustering(final int[][] pop, final double[][] CM, double mutrate, double mutrate2, int numit){
		this.k = (int)Math.sqrt(CM[0].length);
		// size has to be a multiple of 4
		this.n = (int)Math.sqrt(CM.length);
		this.CM = new double[n*n][k*k];
		for (int i=0; i<n*n; i++){
			this.CM[i] = CM[i].clone();
		}
		// the mutation rates should be between 0 and 1
		this.mutrate = mutrate;
		this.mutrate = mutrate2;
		this.size = pop.length;
		this.pop_new = new int[this.size][this.n*this.k];
		this.pop_old = new int[this.size/2][this.n*this.k];
		for (int i=0; i<this.size; i++){
			this.pop_new[i] = pop[i].clone();
		}
		this.iterate(numit);
	}
	
	/*public static EvoCorrelationClustering hierarchicalECC(final double[][] CM, double mutrate, double mutrate2, int size, int numit ){
		//return new EvoCorrelationClustering(new double[1][1], 1, 1, 2, 2);
	}*/
	
	private double iterate(){
		this.selection();
		this.reproduce();
		this.mutate();
		this.pop_fitness = this.fitness();
		return VectorFun.max(this.pop_fitness)[0];
	}
	
	public void iterate(int numit){
		this.fittest = new double[numit];
		for (int i=0; i<numit; i++){
			this.fittest[i] = this.iterate();
			if (this.fittest[i]==this.prev){
				counter++;
			}
			else {
				counter = 0;
			}
			this.prev = this.fittest[i];
			if (this.counter>=this.eliminationrate){
				this.counter = 0;
				this.elimination();
			}
		}
	}
	
	// the fitness function for individuals
	private double fitness(final int[] ind){
		double val=0;
		// val is the sum of all the correlation values of the population string
		// which are n repeated sequences of nc length
		// from all the sets crossing the corresponding string values
		for (int i=0; i<this.n; i++){
			for (int j=i+1; j<this.n; j++){
				for (int l=0; l<this.k; l++){
					val+=CM[i+j*n][ind[i*k+l]+k*ind[j*k+l]];
				}
			}
		}
		return val;
	}
	
	// the fitness of the whole population
	public double[] fitness(){
		double[] val = new double[this.size];
		for (int i=0; i<this.size; i++){
			val[i] = this.fitness(this.pop_new[i]);
		}
		return val;
	}
	
	private int[] generate(){
		int[] temp = new int[this.k];
		int[] ind = new int[this.k*this.n];
		for (int i=0; i<this.n; i++){
			temp = Shuffle.randPerm(this.k);
			for (int j=0; j<this.k; j++){
				ind[i*k+j] = temp[j];
			}
		}
		localSearch(ind);
		return ind;
	}	
	
	private void generatePopulation(){
		// initiate the new population
		this.pop_new = new int[this.size][this.n*this.k];
		// each indivual is built up of random permutations of the integers from 0 to k
		int[] temp = new int[this.k];
		for (int i=0; i<this.size; i++){
			this.pop_new[i] = generate();
		}
		this.pop_fitness = this.fitness();
		// The old population contains individuals which survive for reproduction
		this.pop_old = new int[this.size/2][this.n*this.k];
	}
	
	private void localSearch(int[] ind){
		// initial fitness
		double fit = fitness(ind);
		double fit_old = fit;
		// temporary fitness change
		double f, ft;
		// if the individual is locally optimal
		boolean nonopt = true;
		// randomized order to go through all sets
		int[] order = new int[this.n];
		// randomized order to go through all clusters
		int[] order2 = new int[this.k];
		// randomized order to go through all clusters... again
		int[] order3 = new int[this.k-1];
		int a,b, c;
		while (nonopt){
			order = Shuffle.randPerm(this.n);
			// we only look at flips of two clusters in each set
			// this has complexity k^2 instead of k!
			// go through all sets
			for (int i=0; i<this.n; i++){
				// go through each cluster
				order2 = Shuffle.randPerm(this.k);
				// the largest change in fitness among the following clusters
				ft = 0;
				// clusters which should be swapped
				a = 0;
				b = 0;
				for (int j=0; j<this.k; j++){
					// vs each other cluster
					order3 = Shuffle.randPerm(this.k-1);
					for (int l=0; l<this.k-1; l++){
						// fitness change based on the current value
						f = 0;
						for (int m=0; m<this.n; m++){
							// we don't look compare the set against itself
							if(order[i]!=m){
								// yeah, these are quite a few ugly lines of nested referencing...
								// check the loss of fitness from the old value location
								f-=CM[m+order[i]*n][ind[order[i]*k+order2[j]]*k+ind[m*k+order2[j]]];
								f-=CM[m+order[i]*n][ind[order[i]*k+order2[order3[l]+1]]*k+ind[m*k+order2[order3[l]+1]]];
								// check the gain of fitness from the new value location
								f+=CM[m+order[i]*n][ind[order[i]*k+order2[j]]*k+ind[m*k+order2[order3[l]+1]]];
								f+=CM[m+order[i]*n][ind[order[i]*k+order2[order3[l]+1]]*k+ind[m*k+order2[j]]];
							}
						}
						// if the new fittness surpasses the old surplus take not of this 
						if (f>ft){
							a = order2[j];
							b = order2[order3[l]+1];
						}
					}
				}
				// we take the swap of variables which provides the greatest fitness surplus
				fit+=ft;
				c = ind[order[i]*k+a];
				ind[order[i]*k+a] = ind[order[i]*k+b];
				ind[order[i]*k+b] = c;
			}
			if (fit==fit_old){
				nonopt = false;
			}
		}
	}
	
	// we only mutate offspring at their production
	// after reproduction we have the order:
	// parent, parent, child, child, parent, parent, child, etc.
	private void mutate(){
		// random number generator
		Random r = new Random();
		// integer array for shuffled values
		int[] temp = new int[this.k];
		// temporary variables
		double a;
		int b;
		// go through all individuals in the population
		for (int i=0; i<this.size; i++){
			b = i%4;
			if (b==2||b==3){
				// for each indivual go through all sets
				for (int j=0; j<this.n; j++){
					// for each string check if it should be mutated
					// check if a strong mutation should occur
					// a strong mutatation reorders a whole set
					a = r.nextDouble();
					if (a<this.mutrate2){
						temp = Shuffle.randPerm(this.k);
						for (int l=0; j<this.k; l++){
							this.pop_new[i][j*k+l] = temp[l];
						}
					}
					// a weak mutation swaps two cluster values with one another
					else if (a<this.mutrate){
						temp = Shuffle.randPerm(this.k);
						b = this.pop_new[i][j*k+temp[0]];
						this.pop_new[i][j*k+temp[0]] = this.pop_new[i][j*k+temp[1]];
						this.pop_new[i][j*k+temp[1]] = b;
					}
				}
				// find locally optimized versions after mutation
				localSearch(this.pop_new[i]);
			}
		}
	}
	
	private void selection(){
		int[] temp = Shuffle.randPerm(this.size);
		// compare two neighboring individuals according to the shuffled order
		// put the fitter indivudual into the old population
		for (int i=0; i<this.size/2; i++){
			if (this.pop_fitness[temp[2*i+1]]>this.pop_fitness[temp[2*i]]){
				this.pop_old[i] = this.pop_new[temp[2*i+1]].clone();
			}
			else {
				this.pop_old[i] = this.pop_new[temp[2*i]].clone();
			}
		}
		// return the highest fitness value
	}
	
	private int hamiltonDistance(final int[] a, final int[] b){
		int hd = 0;
		for (int i=0; i<a.length; i++){
			if (a[i]==b[i]){
				hd++;
			}
		}
		return hd;
	}
	
	private void reproduce(){
		// random number generator
		Random r = new Random();
		for (int i=0; i<this.size/4; i++){
			// keep the old individuals
			this.pop_new[i*4] = this.pop_old[i*2].clone();
			this.pop_new[i*4+1] = this.pop_old[i*2+1].clone();
			// mix to old indivuals and generate two offspring
			for (int j=0; j<this.n; j++){
				// first child
				if (r.nextBoolean()){
					for (int l=0; l<this.k; l++){
						this.pop_new[i*4+2][j*k+l] = this.pop_old[i*2][j*k+l];
					}
				}
				else {
					for (int l=0; l<this.k; l++){
						this.pop_new[i*4+2][j*k+l] = this.pop_old[i*2+1][j*k+l];
					}
				}
				// second child
				if (r.nextBoolean()){
					for (int l=0; l<this.k; l++){
						this.pop_new[i*4+3][j*k+l] = this.pop_old[i*2][j*k+l];
					}
				}
				else {
					for (int l=0; l<this.k; l++){
						this.pop_new[i*4+3][j*k+l] = this.pop_old[i*2+1][j*k+l];
					}
				}
			}
		}
	}
	
	private void elimination(){
		int loc = (int)(VectorFun.max(this.pop_fitness)[1]);
		this.pop_new[0] = this.pop_new[loc].clone();
		for (int i=1; i<this.size; i++){
			this.pop_new[i] = generate();
		}
		this.pop_fitness = this.fitness();
	}
	
	// getter functions
	public double[] getHistory(){
		return this.fittest.clone();
	}
	
	public int[] getFittest(){
		int loc = (int)(VectorFun.max(this.pop_fitness)[1]);
		return this.pop_new[loc].clone();
	}
	
	public int[][] getPopulation(){
		int[][] pop = new int[this.size][this.k*this.n];
		for (int i=0; i<this.size; i++){
			pop[i] = this.pop_new[i].clone();
		}
		return pop;
	}
}