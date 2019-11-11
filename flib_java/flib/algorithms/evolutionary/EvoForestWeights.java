package flib.algorithms.evolutionary;

import java.util.Random;
import java.util.ArrayList;
import java.lang.Math;
import flib.algorithms.randomforest.RandomForest;
import flib.algorithms.randomforest.splitfunctions.SplitFunction;
import flib.math.VectorFun;
import flib.math.random.Shuffle;

public class EvoForestWeights {
	private ArrayList<RandomForest> population;
	private int pop_size, n, dim,ntree, counter, reset;
	private double mutation_rate;
	private double[] labels, weights, fitness, avg_fitness, max_fitness, splitpurity, parameters;
	private double[][] trainingset;
	private boolean[] categorical;
	private SplitFunction G;
	
	public EvoForestWeights(final double[][] trainingset, final double[] labels, final double[] weights, final boolean[] categorical, final double[] dimweights, final double[] parameters, final double[] splitpurity, final SplitFunction G, int ntree, int pop_size, double mutation_rate, int numit, int reset){
		this.dim = trainingset[0].length;
		this.n = trainingset.length;
		this.trainingset = new double[n][this.dim];
		for (int i=0; i<n; i++){
			this.trainingset[i] = trainingset[i].clone();
		}
		this.categorical = categorical.clone();
		this.labels = labels.clone();
		this.weights = weights.clone();
		this.ntree = ntree;
		this.parameters = parameters.clone();
		this.splitpurity = splitpurity.clone();
		this.pop_size = (pop_size/4)*4;
		this.mutation_rate = mutation_rate;
		this.reset = reset;
		this.avg_fitness = new double[numit];
		this.max_fitness = new double[numit];
		this.fitness = new double[this.pop_size];
		this.population = new ArrayList<RandomForest>();
		for (int i=0; i<this.pop_size; i++){
			this.population.add(generateForest());
			this.fitness[i] = this.population.get(i).outOfBagError()[0];
		}
		this.avg_fitness[0] = VectorFun.sum(this.fitness)/this.pop_size;
		this.max_fitness[0] = VectorFun.min(this.fitness)[0];
		this.counter = 0;
		for (int i=1; i<numit; i++){
			iterate(i);
		}
	}
	
	public RandomForest generateForest(){
		Random rng = new Random();
		double[] dimweights = new double[dim];
		for (int i=0; i<dim; i++){
			dimweights[i] = rng.nextDouble();
		}
		double[] p = parameters.clone();
		// randomly set mtry
		p[1] = rng.nextInt(this.dim)+1;
		return (new RandomForest(this.trainingset,this.labels,this.weights,this.categorical,dimweights,p,this.splitpurity,this.G,this.ntree));
	}
	
	public RandomForest generateForest(int mtry, double[] dimweights){
		double[] p = parameters.clone();
		p[1] = mtry;
		return (new RandomForest(this.trainingset,this.labels,this.weights,this.categorical,dimweights,p,this.splitpurity,this.G,this.ntree));
	}
	
	public void iterate(int it){
		Random rng = new Random();
		int[] fit_ind = new int[this.pop_size/2];
		int[] unfit_ind = new int[this.pop_size/2];
		int[] randdim = Shuffle.randPerm(pop_size);
		// selection
		for (int i=0; i<this.pop_size/2; i++){
			if (this.fitness[randdim[2*i]]<this.fitness[randdim[2*i+1]]){
				fit_ind[i] = randdim[2*i];
				unfit_ind[i] = randdim[2*i+1];
			}
			else {
				fit_ind[i] = randdim[2*i+1];
				unfit_ind[i] = randdim[2*i];
			}
		}
		int mtry;
		double[] dimweights = new double[this.dim];
		// crossover reproduction+mutation
		for (int i=0; i<this.pop_size/4; i++){
			for (int k=0; k<2; k++){
				double[] dw0 = this.population.get(fit_ind[2*i]).getDW();
				double[] dw1 = this.population.get(fit_ind[2*i+1]).getDW();
				for (int j=0; j<this.dim; j++){
					if (rng.nextInt(2)==0){
						dimweights[j] = dw0[j];
					}
					else {
						dimweights[j] = dw1[j];
					}
					if (rng.nextDouble()<=this.mutation_rate){
						// smaller
						if (rng.nextBoolean()){
							dimweights[j] = rng.nextDouble()*dimweights[j];
						}
						// larger
						else {
							dimweights[j] = rng.nextDouble()*(1-dimweights[j])+dimweights[j];
						}
					}
				}
				mtry = population.get(fit_ind[2*i+rng.nextInt(2)]).getMtry();
				if (rng.nextDouble()<=this.mutation_rate){
					// smaller
					if (rng.nextBoolean()){
						mtry = rng.nextInt(mtry)+1;
					}
					// larger
					else {
						mtry = rng.nextInt(this.dim-mtry+1)+mtry;
					}
				}
				this.population.set(unfit_ind[2*i+k],generateForest(mtry,dimweights));
			}
			// have the parent forests be recalculated
			//this.population.set(fit_ind[2*i],generateForest(this.population.get(fit_ind[2*i]).getMtry(),this.population.get(fit_ind[2*i]).getDimWeights().clone()));
			//this.population.set(fit_ind[2*i+1],generateForest(this.population.get(fit_ind[2*i+1]).getMtry(),this.population.get(fit_ind[2*i+1]).getDimWeights().clone()));
		}
		// calculate the new fitness values
		for (int i=0; i<this.pop_size; i++){
			this.fitness[i] = this.population.get(i).outOfBagError()[0];
		}
		this.avg_fitness[it] = VectorFun.sum(this.fitness)/this.pop_size;
		double[] min = VectorFun.min(this.fitness);
		this.max_fitness[it] = min[0];
		// replace the population except for the fittest indivual if the average
		// population fitness did not increase the last round
		if (this.max_fitness[it]==this.max_fitness[it-1]){
			this.counter++;
			if (this.counter>this.reset){
				this.counter = 0;
				for (int i=0; i<this.pop_size; i++){
					if (i!=(int)min[1]){
						this.population.set(i,generateForest());
					}
				}
				this.avg_fitness[it] = VectorFun.sum(this.fitness)/this.pop_size;
				this.max_fitness[it] = VectorFun.min(this.fitness)[0];
			}
		}
		else {
			this.counter = 0;
		}
	}
	
	public ArrayList<RandomForest> getPopulation(){
		return this.population;
	}
	
	public double[] getFitness(){
		return this.fitness;
	}
	
	public double[] getAvgFitness(){
		return this.avg_fitness;
	}
	
	public double[] getMaxFitness(){
		return this.max_fitness;
	}
}