package flib.algorithms.clustering;

import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Random;
import java.lang.Math;
import flib.math.FractionSortPair;
import flib.math.VectorFun;
import flib.math.random.Shuffle;
import ij.IJ;


public class GibbsClusterMatching{
	// total number of clusters over all sets
	private int tnc;
	// target number of clusters
	private int nct;
	// number of sets
	private int d;
	// number of pixels
	private int n;
	// mapping of each initial cluster to the new target clusters
	private int[] nc;
	// the initial clusters
	private double[][] iniset;
	// the collection of pixels
	private ArrayList<TreeSet<FractionSortPair>> x;
	// target cluster distribution per pixel
	private int[][] y;
	// log sum per pixel
	private double[] logsum;
	// current total logsum
	private double cls;
	// probabilty epsilon
	private double epsilon;
	// number of iterations
	private int numit;
	// step rejection factor
	private double beta;
	// target cluster each cluster belongs to
	private int[] tc;
	// the inverse... which set each cluster belongs to
	private int[] tci;
	// number of clusters each target cluster contains
	private int[] nt;
	// log sum history
	double[] hist;
	
	
	
	public GibbsClusterMatching(final double[][] iniset, int nct, double epsilon, int numit, double beta){
		// variable initialization
		this.nct = nct;
		this.numit = numit;
		this.epsilon = epsilon+1;
		this.beta = beta;
		this.tnc = 0;
		this.d = iniset.length;
		this.n = iniset[0].length;
		this.iniset = new double[this.d][this.n];
		for (int i=0; i<this.d; i++){
			this.iniset[i] = iniset[i].clone();
		}
		this.x = new ArrayList<TreeSet<FractionSortPair>>();
		this.y = new int[this.n][this.nct];
		this.nc = new int[this.d];
		int[] nc2 = new int[this.d];
		// go through all set and count the number of clusters
		for (int i=0; i<this.d; i++){
			// the cluster number of this image are higher than
			this.nc[i] = this.tnc;
			// number of clusters in all sets together
			this.tnc+=(int)(VectorFun.max(this.iniset[i])[0])+1;
			nc2[i] = this.tnc;
		}
		this.tc = new int[this.tnc];
		this.nt = new int[this.nct];
		this.tci = new int[this.tnc];
		// random initialization of the initial clusters to the target clusters
		// these will be evenly distributed over the target clusters
		int[] order = Shuffle.randPerm(this.tnc);
		int count = 0;
		for (int i=0; i<this.tnc; i++){
			this.tc[order[i]] = i%this.nct;
			// how many clusters a target cluster contains
			this.nt[i%this.nct]++;
			// find out in which set the cluster is in
			if (nc2[count]<=i){
				count++;
			}
			this.tci[i] = count;
		}
		// go through all pixels and increase the counters for each target cluster
		// containing that pixel
		for (int i=0; i<this.d; i++){
			for (int j=0; j<this.n; j++){
				this.y[j][this.tc[(int)this.iniset[i][j]+this.nc[i]]]++;
			}
		}
		this.logsum = new double[this.n];
		// now we come to a slightly more complicated part (actually... not):
		// each pixel has a sort tree which is ordered according to is pixel cluster
		// fraction
		for (int i=0; i<this.n; i++){
			this.x.add(new TreeSet<FractionSortPair>());
			for (int j=0; j<this.nct; j++){
				this.x.get(i).add(new FractionSortPair(y[i][j],this.nt[j],j));
				// initial log sum values
				this.logsum[i]+=Math.log(this.epsilon-(float)y[i][j]/this.nt[j]);
			}
		}
		// for each pixel we calculate the log sum value after finding 
		// the maximum probability and adjusting the current log sum value
		double p;
		for (int i=0; i<this.n; i++){
			p = this.x.get(i).last().getFraction();
			this.logsum[i]-=Math.log(this.epsilon-p);
			this.logsum[i]+=Math.log(p);
		}
		this.cls = VectorFun.sum(this.logsum);
		this.hist = new double[numit];
		// now we are ready to iterate the alogrithm
		for (int i=0; i<this.numit; i++){
			randomIterate();
			this.hist[i] = this.cls;
		}
	}
	
	private void randomIterate(){
		// we begin the iteration by choosing a random cluster
		Random rand = new Random();
		int a = rand.nextInt(this.tnc);
		// old cluster location
		int o = this.tc[a];
		// new random target cluster
		int b = rand.nextInt(this.nct-1);
		if (b>=o){
			b++;
		}
		if (this.nt[o]<=1){
			return;
		}
		// temporarily update all pixel related values
		double temp;
		int a2;
		for (int i=0; i<this.n; i++){
			// check for each point if it belongs the initial cluster which is changing is target cluster
			a2 = (int)this.iniset[this.tci[a]][i]+this.nc[this.tci[a]];
			// fix the probailities to find the new probability maximum
			temp = this.x.get(i).last().getFraction();
			this.logsum[i]-=Math.log(temp);
			this.logsum[i]+=Math.log(this.epsilon-temp);
			// change the values in the old location
			this.logsum[i]-=Math.log(this.epsilon-(float)this.y[i][o]/this.nt[o]);
			this.x.get(i).remove(new FractionSortPair(this.y[i][o],this.nt[o],o));
			if (a2==a){
				this.y[i][o]-=1;
			}
			this.x.get(i).add(new FractionSortPair(this.y[i][o],(this.nt[o]-1),o));
			//this.x.get(i).add(new FractionSortPair(this.y[i][o],(this.nt[o]),o));
			this.logsum[i]+=Math.log(this.epsilon-(float)this.y[i][o]/(this.nt[o]-1));
			// change values in the new location
			this.logsum[i]-=Math.log(this.epsilon-(float)this.y[i][b]/this.nt[b]);
			this.x.get(i).remove(new FractionSortPair(this.y[i][b],this.nt[b],b));
			if (a2==a){
				this.y[i][b]+=1;
			}
			this.x.get(i).add(new FractionSortPair(this.y[i][b],(this.nt[b]+1),b));
			//this.x.get(i).add(new FractionSortPair(this.y[i][b],(this.nt[b]),b));
			this.logsum[i]+=Math.log(this.epsilon-(float)this.y[i][b]/(this.nt[b]+1));
			// look for the new maximum value
			temp = this.x.get(i).last().getFraction();
			this.logsum[i]-=Math.log(this.epsilon-temp);
			this.logsum[i]+=Math.log(temp);
		}
		if ((VectorFun.sum(this.logsum)>this.cls)||(rand.nextDouble()>this.beta)){
			// everything stays the same except
			this.cls = VectorFun.sum(this.logsum);
			this.nt[o]-=1;
			this.nt[b]+=1;
			this.tc[a] = b;
		}
		else {
			// we need to revert all values to their previous state...
			for (int i=0; i<this.n; i++){
				// check for each point if it belongs the initial cluster which is changing is target cluster
				a2 = (int)this.iniset[this.tci[a]][i]+this.nc[this.tci[a]];
				// fix the probailities to find the new probability maximum
				temp = this.x.get(i).last().getFraction();
				this.logsum[i]-=Math.log(temp);
				this.logsum[i]+=Math.log(this.epsilon-temp);
				// change the values in the old location
				this.logsum[i]-=Math.log(this.epsilon-(float)this.y[i][o]/(this.nt[o]-1));
				this.x.get(i).remove(new FractionSortPair(this.y[i][o],(this.nt[o]-1),o));
				if (a2==a){
					this.y[i][o]+=1;
				}
				this.x.get(i).add(new FractionSortPair(this.y[i][o],this.nt[o],o));
				this.logsum[i]+=Math.log(this.epsilon-(float)this.y[i][o]/this.nt[o]);
				// change values in the new location
				this.logsum[i]-=Math.log(this.epsilon-(float)this.y[i][b]/(this.nt[b]+1));
				this.x.get(i).remove(new FractionSortPair(this.y[i][b],(this.nt[b]+1),b));
				if (a2==a){
					this.y[i][b]-=1;
				}
				this.x.get(i).add(new FractionSortPair(this.y[i][b],this.nt[b],b));
				this.logsum[i]+=Math.log(this.epsilon-(float)this.y[i][b]/this.nt[b]);
				// look for the new maximum value
				temp = this.x.get(i).last().getFraction();
				this.logsum[i]-=Math.log(this.epsilon-temp);
				this.logsum[i]+=Math.log(temp);
			}
		}
	}
	
	public double[][] getFractions(){
		double[][] temp = new double[this.nct][this.n];
		for (int i=0; i<this.n; i++){
			for (int j=0; j<this.nct; j++){
				temp[j][i] = (double)y[i][j]/this.nt[j];
			}
		}
		return temp;
	}
	
	public double[] getHistory(){
		return this.hist;
	}
	
	public ArrayList<TreeSet<FractionSortPair>> getTrees(){
		return this.x;
	}
}	