package flib.neurons.spiking;

import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Random;
import java.lang.Math;
import ij.IJ;
import flib.math.VectorFun;
import flib.neurons.spiking.SynapticEvent;

public class SingleThresholdUnit implements
java.io.Serializable {
	// synaptic weighting of incoming signals
	private double[] weights_exc, weights_inh;
	// list which weights are fixed ( at the moment only for excitatory connections)
	private boolean[] fixed;
	// list of past firing inputs and input time for each synapse
	private double[] input_exc, input_inh, in_time_exc, in_time_inh;
	// synaptic targets and origins
	private int[][] targets, syn_loc, syn_orig_exc, syn_orig_inh;
	// synaptic delays
	private double[][] delay;
	// modules targeted by this neuron
	private int[] target_modules;
	// neuron type
	private boolean type;
	// learning factors
	// learning rate decay
	private double alpha_t_exc, alpha_t_inh1, alpha_t_inh2;
	private double alpha_fact_exc, alpha_fact_inh;
	// target interspike intervals
	private double target_rate_exc, target_rate_inh;
	// amplitude learning
	private double amp_fact_exc, amp_fact_inh;
	// refractory period
	private double refrac;
	// threshold
	private double threshold;
	// decay constant
	private double decay_rate;
	// input amplification
	private double amp_exc, amp_inh;
	// number of spike events for rate
	private double spike_frame;
	// time of last observation
	private double time;
	// neuron potential
	private double pot;
	// last refractory period beginning
	private double last_refrac;
	// hyperpolarization after activation
	private double hyper;
	// history variables for different purposes
	private double pot_hist_exc, pot_hist_inh1, pot_hist_inh2, pot_t_exc, pot_t_inh, amp_t;
	private double[] decay_exc, decay_inh1, decay_inh2;
	// weight adjustment factors
	private double w_exc, w_inh;
	// estimation of the current interspike times
	private double rate, last_spike;
	// neuron contains an identity generator to prevent event collisions
	private Random rng;
	// weight balancing exponents
	private double weight_fact_exc, weight_fact_inh;
	
	public SingleThresholdUnit(final double[] weights_exc, final double[] weights_inh, final boolean[] fixed, final int[] target_modules, final int[][] targets, final int[][] syn_loc, final int[][] syn_orig_exc, final int[][] syn_orig_inh, final double[][] delay, final double[] learning, final double[] rate, boolean type, double amp_exc, double amp_inh, double refrac, double threshold, double decay_rate, double hyper, double spike_frame, double weight_fact_exc, double weight_fact_inh){
		// variable initialization
		this.weights_exc = weights_exc.clone();
		this.input_exc = new double[weights_exc.length];
		this.in_time_exc = new double[weights_exc.length];
		this.decay_exc = new double[weights_exc.length];
		this.weights_inh = weights_inh.clone();
		this.input_inh = new double[weights_inh.length];
		this.in_time_inh = new double[weights_inh.length];
		this.decay_inh1 = new double[weights_inh.length];
		this.decay_inh2 = new double[weights_inh.length];
		this.fixed = fixed.clone();
		this.targets = new int[targets.length][];
		for (int i=0; i<this.targets.length; i++){
			this.targets[i] = targets[i].clone();
		}
		this.syn_loc = new int[syn_loc.length][];
		for (int i=0; i<this.syn_loc.length; i++){
			this.syn_loc[i] = syn_loc[i].clone();
		}
		this.syn_orig_exc = new int[syn_orig_exc.length][];
		this.syn_orig_inh = new int[syn_orig_inh.length][];
		for (int i=0; i<this.syn_orig_exc.length; i++){
			this.syn_orig_exc[i] = syn_orig_exc[i].clone();
		}
		for (int i=0; i<this.syn_orig_inh.length; i++){
			this.syn_orig_inh[i] = syn_orig_inh[i].clone();
		}
		this.delay = new double[delay.length][];
		for (int i=0; i<this.delay.length; i++){
			this.delay[i] = delay[i].clone();
		}
		this.target_modules = target_modules.clone();
		this.type = type;
		this.alpha_t_exc = learning[0];
		this.alpha_fact_exc = learning[1];
		this.alpha_t_inh1 = learning[2];
		this.alpha_t_inh2 = learning[3];
		this.alpha_fact_inh = learning[4];
		this.target_rate_exc = rate[0];
		this.target_rate_inh = rate[1];
		this.amp_fact_exc = rate[2];
		this.amp_fact_inh = rate[3];
		this.amp_exc = amp_exc;
		this.amp_inh = amp_inh;
		this.refrac = refrac;
		this.threshold = threshold;
		this.decay_rate = decay_rate;
		this.hyper = hyper;
		this.time = 0;
		this.pot = 0;
		this.last_refrac = -refrac;
		this.w_exc = 1;
		this.w_inh = 1;
		// some initial value in case the rate is never set
		this.rate = 1;
		this.spike_frame = spike_frame;
		this.last_spike = 0;
		this.rng = new Random();
		this.weight_fact_exc = weight_fact_exc;
		this.weight_fact_inh = weight_fact_inh;
	}
	
	public boolean synapticInput(boolean type, int loc, double t){
		boolean fired = false;
		// check if we're in a refractory period. If yes, do nothing
		if ((t-last_refrac)<refrac){
			return fired;
		}
		// update the potential decay
		pot*=Math.exp(-(t-time)*decay_rate);
		// case of excitatory input
		if (type){
			double dt = t-amp_t;
			// update the current synapse weight (if it's not fixed)
			if(!fixed[loc]){
				// time difference to the last input spike
				double delta_t = t-pot_t_exc;
				// update the effect of previous spikes of this cell according to time
				pot_hist_exc*=Math.exp(-delta_t*alpha_t_exc);
				// factor to add to the current weight
				double a = -alpha_fact_exc*pot_hist_exc*Math.exp(-weight_fact_exc*weights_exc[loc]/w_exc);
				if (a<-weights_exc[loc]){
					a = -weights_exc[loc];
				}
				// sum of all weights
				w_exc+=a;
				// adjustment of the current individual weight
				weights_exc[loc]+=a;
				// update the trace for the input events at this synapse
				decay_exc[loc] = decay_exc[loc]*Math.exp(-(t-in_time_exc[loc])*alpha_t_exc)+1;
				// update the synapse time
				in_time_exc[loc] = t;
				// adjust the current potential
				pot+=amp_exc*weights_exc[loc]/w_exc;
				// update the last input spike time
				pot_t_exc = t;
				// adjustment of the amplification factor
				amp_exc+=amp_fact_exc*(target_rate_exc*dt-amp_exc/w_exc*weights_exc[loc]);
			}
			else {
				// adjust the current potential
				pot+=weights_exc[loc];
				// adjustment of the amplification factor
				amp_exc+=amp_fact_exc*(target_rate_exc*dt-weights_exc[loc]);
			}
			amp_t = t;
			if (pot>=threshold){
				this.fire(t);
				fired = true;
			}
		}
		else {
			// time difference to the last input spike
			double delta_t = t-pot_t_inh;
			// update the effect of previous spikes according to time
			pot_hist_inh1*=Math.exp(-delta_t*alpha_t_inh1);
			pot_hist_inh2*=Math.exp(-delta_t*alpha_t_inh2);
			// factor to add to the current weight
			double a = alpha_fact_inh*(pot_hist_inh1-pot_hist_inh2)*Math.exp(-weight_fact_inh*weights_inh[loc]/w_inh);
			if (a<-weights_inh[loc]){
				a = -weights_inh[loc];
			}
			// sum of all weights
			w_inh+=a;
			// adjustment of the current individual weight
			weights_inh[loc]+=a;
			// update the trace for the input events at this synapse
			decay_inh1[loc] = decay_inh1[loc]*Math.exp(-(t-in_time_inh[loc])*alpha_t_inh1)+1;
			decay_inh2[loc] = decay_inh2[loc]*Math.exp(-(t-in_time_inh[loc])*alpha_t_inh2)+1;
			// update the synapse time
			in_time_inh[loc] = t;
			// adjust the current potential
			pot-=amp_inh/w_inh*weights_inh[loc];
			// update the last input spike time
			pot_t_inh = t;
			// adjustment of the amplification factor
			amp_inh+=amp_fact_inh*(target_rate_inh*delta_t-amp_inh/w_inh*weights_inh[loc]);
		}
		// update the current neuron time
		time = t;
		return fired;
	}
	
	public void refractoryPeriod(double t){
		pot = -hyper;
		time = t;
	}		
	
	public void fire(double t){
		// update 
		// adjust all weights
		for (int i=0; i<weights_exc.length; i++){
			if (!fixed[i]){
				// adjust the decay factor
				decay_exc[i]*=Math.exp(-(t-in_time_exc[i])*alpha_t_exc);
				// last decay time is now
				in_time_exc[i] = t;
				double a = alpha_fact_exc*decay_exc[i];
				// sum of all weights
				w_exc+=a;
				// adjustment of the current individual weight
				weights_exc[i]+=a;
			}
		}
		// update the excitatory post synaptic trace
		pot_hist_exc = pot_hist_exc*Math.exp(-(t-pot_t_exc))+1;
		pot_t_exc = t;
		// all the same for inhibition
		for (int i=0; i<weights_inh.length; i++){
			// adjust the decay factor
			decay_inh1[i]*=Math.exp(-(t-in_time_inh[i])*alpha_t_inh1);
			decay_inh2[i]*=Math.exp(-(t-in_time_inh[i])*alpha_t_inh2);
			// last decay time is now
			in_time_inh[i] = t;
			double a = alpha_fact_inh*(decay_inh2[i]-decay_inh1[i]);
			if (a<-weights_inh[i]){
				a = -weights_inh[i];
			}
			// sum of all weights
			w_inh+=a;
			// adjustment of the current individual weight
			weights_inh[i]+=a;
		}
		// update the inhibitory post synaptic trace
		pot_hist_inh1 = pot_hist_inh1*Math.exp(-(t-pot_t_inh)*alpha_t_inh1)+1;
		pot_hist_inh2 = pot_hist_inh2*Math.exp(-(t-pot_t_inh)*alpha_t_inh2)+1;
		pot_t_inh = t;
		// renormalize all weights
		double a = 0;
		for (int i=0; i<weights_exc.length; i++){
			if(!fixed[i]){
				a+=weights_exc[i];
			}
		}
		if (Math.abs(a-w_exc)>1e-6&&weights_exc.length>0){
			IJ.log("there was a problem in the weight normalization");
		}
		w_exc = a;
		for (int i=0; i<weights_exc.length; i++){
			if (!fixed[i]){
				weights_exc[i]/=w_exc;
			}
		}
		w_exc = 1;
		a = 0;
		for (int i=0; i<weights_inh.length; i++){
			a+=weights_inh[i];
		}
		if (Math.abs(a-w_inh)>1e-6&&weights_inh.length>0){
			IJ.log("there was a problem in the weight normalization");
		}
		w_inh = a;
		for (int i=0; i<weights_inh.length; i++){
			weights_inh[i]/=w_inh;
		}
		w_inh = 1;
	}
	
	public void project(double t, TreeSet<SynapticEvent> eventlist, int module_number, int neuron_number){
		// estimation of the interspike interval
		rate = (rate*spike_frame+t-last_spike)/(1+spike_frame);
		last_spike = t;
		// update the last refractory time
		last_refrac = t;
		// send to refractory event
		if(!eventlist.add(new SynapticEvent(t+refrac,rng.nextDouble(),2,module_number, neuron_number,0))){
			IJ.log("a refractory event was lost...");
		}
		int a;
		if (type){
			a = 0;
		}
		else {
			a = 1;
		}
		for (int i=0; i<targets.length; i++){
			for (int j=0; j<targets[i].length; j++){
				if(!eventlist.add(new SynapticEvent(t+delay[i][j],rng.nextDouble(),a,target_modules[i],targets[i][j],syn_loc[i][j]))){
					IJ.log("a synaptic transmission event was lost...");
				}
			}
		}
	}
	
	public double getRate(){
		return this.rate;
	}
	
	public double getAmp_exc(){
		return this.amp_exc;
	}
	
	public double getAmp_inh(){
		return this.amp_inh;
	}
	
		public double[] getWeights_exc(){
		return this.weights_exc.clone();
	}
	
	public double[] getWeights_inh(){
		return this.weights_inh.clone();
	}
	
	public boolean[] getFixed(){
		return this.fixed.clone();
	}
	
	public int[][] getTargets(){
		int temp[][] = new int[targets.length][];
		for (int i=0; i<targets.length; i++){
			temp[i] = targets[i].clone();
		}
		return temp;
	}
	
	public int[][] getSynapseLocations(){
		int temp[][] = new int[syn_loc.length][];
		for (int i=0; i<syn_loc.length; i++){
			temp[i] = syn_loc[i].clone();
		}
		return temp;
	}
	
	public int[][] getSynapseOrigins_exc(){
		int temp[][] = new int[syn_orig_exc.length][];
		for (int i=0; i<syn_orig_exc.length; i++){
			temp[i] = syn_orig_exc[i].clone();
		}
		return temp;
	}
	
	public int[][] getSynapseOrigins_inh(){
		int temp[][] = new int[syn_orig_inh.length][];
		for (int i=0; i<syn_orig_inh.length; i++){
			temp[i] = syn_orig_inh[i].clone();
		}
		return temp;
	}
	
	public boolean getType(){
		return this.type;
	}
	
	public double[] getInput_exc(){
		return this.input_exc.clone();
	}
	
	public double[] getInput_inh(){
		return this.input_inh.clone();
	}
	
	public double getRefrac(){
		return this.refrac;
	}
	
	public double getHyper(){
		return this.hyper;
	}
	
	public double getPot(){
		return this.pot;
	}
	
	public double getThreshold(){
		return this.threshold;
	}
	
	public double getTime(){
		return this.time;
	}
	
	public double getSpikeFrame(){
		return this.spike_frame;
	}
	
	public double getDecayRate(){
		return this.decay_rate;
	}
	
	public void setAlpha_exc(double alpha_exc){
		this.alpha_fact_exc = alpha_exc;
	}
	
	public void setAlpha_inh(double alpha_inh){
		this.alpha_fact_inh = alpha_inh;
	}
	
	public void setRate(double rate){
		this.rate = rate;
	}
	
	public void setAmp_fact_exc(double amp_fact_exc){
		this.amp_fact_exc = amp_fact_exc;
	}
	
	public void setAmp_fact_inh(double amp_fact_inh){
		this.amp_fact_inh = amp_fact_inh;
	}
	
	public void setSpikeFrame(double spike_frame){
		this.spike_frame = spike_frame;
	}
}