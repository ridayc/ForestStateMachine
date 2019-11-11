package flib.neurons;

import java.util.ArrayList;
import java.util.Random;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.math.VectorFun;
import flib.math.random.Sample;
import flib.datastructures.TypeConversion;
import flib.neurons.SingleThresholdUnit;

public class NetworkModule implements 
java.io.Serializable {
	
	// list of all the neurons in this module
	// does it matter if we store the neuron list inside or outside of the module?
	private ArrayList<SingleThresholdUnit> neurons;
	// width, height and depth of the neuron field
	private int w,h,d_exc, d_inh;
	// list of target modules
	private int[] target_modules;
	// list of target compartments in the target modules
	
	
	public NetworkModule(int w, int h, int d_exc, int d_inh, final int[] target_modules){
		this.w = w;
		this.h = h;
		this.d_exc = d_exc;
		this.d_inh = d_inh;
		this.target_modules = target_modules.clone();		
	}
	
	public void connectToModule(int num_module, int target_module, final int[] target_compartment, final NetworkModule M, final double[] syn_connectivity, final double[] geo_connectivity, int[][] exc_weight_counter, int[][] inh_weight_counter, ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> syn_loc, ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> syn_orig_exc,ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> syn_orig_inh){
		
		// synaptic connectivity
		// syn_connectivity: contains average numbers of synapses by type neurons
		// formed on other type neurons
		// syn_connectivity[0]: average number of synapses formed by an excitatory 
		// neuron onto excitatory neurons
		// syn_connectivity[1]: average number of synapses formed by an excitatory 
		// neuron onto inhibitory neurons
		// syn_connectivity[2]: average number of synapses formed by an inhibitory 
		// neuron onto excitatory neurons
		// syn_connectivity[3]: average number of synapses formed by an inhibitory 
		// neuron onto inhibitory neurons
	
		// geometric connectivity
		// geo_connectivity[0]: cutoff radius off for excitory synapses
		// geo_connectivity[1]: cutoff radius off for inhibitory synapses
		// geo_connectivity[2]: wrap around boundaries or not
		// geo_connectivity[3]: space filling factor of the neighborhood spiral
		// geo_connectivity[4]: radial spacing of the disk
		// geo_connectivity[5]: fixed excitatory input weights
		// geo_connectivity[6]: respacing of the input-output neuron spacing in x-direction
		// geo_connectivity[7]: respacing of the input-output neuron spacing in y-direction
		
		// some parameters for different use
		int x,y,x2,y2;
		int xm = w/2;
		int ym = h/2;
		int w2 = M.getWidth();
		int h2 = M.getHeight();
		int d_exc2 = M.getD_exc();
		int d_inh2 = M.getD_inh();
		int xm2 = w2/2;
		int ym2 = h2/2;
		Random rng = new Random();
		// first prepare the excitatory neurons
		// prepare the target spiral shape coordinates
		if (geo_connectivity[0]>=0){
			int[][] coord = NeighborhoodSample.circleCoord(geo_connectivity[0]);
			double[] r = new double[coord.length];
			double r_sum = 0;
			for (int j=0; j<r.length; j++){
				r[j] =Math.sqrt(coord[j][0]*coord[j][0]+coord[j][1]*coord[j][1])*geo_connectivity[4];
				if (r[j]<=1){
					r[j] = 1;
				}
				r[j] = Math.pow(r[j],geo_connectivity[3]);
				r_sum+=r[j];
			}
			// prepare the probabilities of certain synapse types
			double p_exc = syn_connectivity[0]/(r_sum*d_exc2);
			double p_inh = syn_connectivity[1]/(r_sum*d_inh2);
			// go over all excitatory neurons
			for (int i=0; i<this.w*this.h*this.d_exc; i++){
				x = (i%(w*h))%w;
				y = (i%(w*h))/w;
				x2 = (int)((x-xm)*geo_connectivity[6]+xm2);
				y2 = (int)((y-ym)*geo_connectivity[7]+ym2);
				// go over all potential excitable targets
				int[] temp = NeighborhoodSample.shapeNeighbor2d(coord,w2,h2,x2,y2,(int)geo_connectivity[2]);
				for (int j=0; j<temp.length; j++){
					if (temp[j]!=-1){
						// for all excitatory layers in the target module
						for (int k=0; k<d_exc2; k++){
							// if the threshold is not exceeded we form a synapse
							if(rng.nextDouble()<p_exc*r[j]){
								int loc = w2*h2*k+temp[j];
								// neurons should not target themselves...
								if ((this.target_modules[target_module]!=num_module)||(loc!=i)){
									// add this current target to the general target list
									linkNeurons(syn_loc,syn_orig_exc,exc_weight_counter,i,loc,target_module,num_module,target_compartment[0]);
								}
							}
						}
						// for all inhibitory layers in the target module
						for (int k=d_exc2; k<d_exc2+d_inh2; k++){
							// if the threshold is not exceeded we form a synapse
							if(rng.nextDouble()<p_inh*r[j]){
								int loc = w2*h2*k+temp[j];
								linkNeurons(syn_loc,syn_orig_exc,exc_weight_counter,i,loc,target_module,num_module,target_compartment[1]);
							}
						}
					}
				}
			}
			
			// inhibitory neurons targeting
			// prepare the target spiral shape coordinates
			coord = NeighborhoodSample.circleCoord(geo_connectivity[1]);
			r = new double[coord.length];
			r_sum = 0;
			for (int j=0; j<r.length; j++){
				r[j] =Math.sqrt(coord[j][0]*coord[j][0]+coord[j][1]*coord[j][1])*geo_connectivity[4];
				if (r[j]<=1){
					r[j] = 1;
				}
				r[j] = Math.pow(r[j],geo_connectivity[3]);
				r_sum+=r[j];
			}
			// prepare the probabilities of certain synapse types
			p_exc = syn_connectivity[2]/(r_sum*d_exc2);
			p_inh = syn_connectivity[3]/(r_sum*d_inh2);
			// go over all inhibitory neurons
			for (int i=this.w*this.h*this.d_exc; i<this.w*this.h*(this.d_exc+this.d_inh); i++){
				x = (i%(w*h))%w;
				y = (i%(w*h))/w;
				x2 = (int)((x-xm)*geo_connectivity[6]+xm2);
				y2 = (int)((y-ym)*geo_connectivity[7]+ym2);
				// go over all potential inhibitable targets
				int[] temp = NeighborhoodSample.shapeNeighbor2d(coord,w2,h2,x2,y2,(int)geo_connectivity[2]);
				for (int j=0; j<temp.length; j++){
					if (temp[j]!=-1){
						// for all excitatory layers in the target module
						for (int k=0; k<d_exc2; k++){
							// if the threshold is not exceeded we form a synapse
							if(rng.nextDouble()<p_exc*r[j]){
								int loc = w2*h2*k+temp[j];
								linkNeurons(syn_loc,syn_orig_inh,inh_weight_counter,i,loc,target_module,num_module,target_compartment[2]);
							}
						}
						// for all inihibitory layers in the target module
						for (int k=d_exc2; k<d_exc2+d_inh2; k++){
							// if the threshold is not exceeded we form a synapse
							if(rng.nextDouble()<p_inh*r[j]){
								int loc = w2*h2*k+temp[j];
								// neurons should not target themselves...
								if ((this.target_modules[target_module]!=num_module)||(loc!=i)){
									linkNeurons(syn_loc,syn_orig_inh,inh_weight_counter,i,loc,target_module,num_module,target_compartment[3]);
								}
							}
						}
					}
				}
			}
		}
		// all to all connectivity
		else {
			double p_exc = syn_connectivity[0]/(w2*h2*d_exc2);
			double p_inh = syn_connectivity[1]/(w2*h2*d_inh2);
			// go over all excitatory neurons
			for (int i=0; i<this.w*this.h*this.d_exc; i++){
				for (int j=0; j<w2*h2; j++){
					// for all excitatory layers in the target module
					for (int k=0; k<d_exc2; k++){
						// if the threshold is not exceeded we form a synapse
						if(rng.nextDouble()<p_exc){
							int loc = w2*h2*k+j;
							// neurons should not target themselves...
							if ((this.target_modules[target_module]!=num_module)||(loc!=i)){
								linkNeurons(syn_loc,syn_orig_exc,exc_weight_counter,i,loc,target_module,num_module,target_compartment[0]);
							}
						}
					}
					// for all inhibitory layers in the target module
					for (int k=d_exc2; k<d_exc2+d_inh2; k++){
						// if the threshold is not exceeded we form a synapse
						if(rng.nextDouble()<p_inh){
							int loc = w2*h2*k+j;
							linkNeurons(syn_loc,syn_orig_exc,exc_weight_counter,i,loc,target_module,num_module,target_compartment[1]);
						}
					}
				}
			}
			
			// inhibitory neurons targeting
			// prepare the probabilities of certain synapse types
			p_exc = syn_connectivity[2]/(w2*h2*d_exc2);
			p_inh = syn_connectivity[3]/(w2*h2*d_inh2);
			// go over all inhibitory neurons
			for (int i=this.w*this.h*this.d_exc; i<this.w*this.h*(this.d_exc+this.d_inh); i++){
				// go over all potential excitatory targets
				for (int j=0; j<w2*h2; j++){
					// for all excitatory layers in the target module
					for (int k=0; k<d_exc2; k++){
						// if the threshold is not exceeded we for a synapse
						if(rng.nextDouble()<p_exc){
							int loc = w2*h2*k+j;
							linkNeurons(syn_loc,syn_orig_inh,inh_weight_counter,i,loc,target_module,num_module,target_compartment[2]);
						}
					}
					// for all inihibitory layers in the target module
					for (int k=d_exc2; k<d_exc2+d_inh2; k++){
						// if the threshold is not exceeded we for a synapse
						if(rng.nextDouble()<p_inh){
							int loc = w2*h2*k+j;
							// neurons should not target themselves...
							if ((this.target_modules[target_module]!=num_module)||(loc!=i)){
								linkNeurons(syn_loc,syn_orig_inh,inh_weight_counter,i,loc,target_module,num_module,target_compartment[3]);
							}
						}
					}
				}
			}
		}
	}
	
	private void linkNeurons(ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> sl,ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> so,int[][] wc,int i, int loc, int tm, int mn, int tc){
		sl.get(i).get(tm).add(new ArrayList<Integer>());
		int s = sl.get(i).get(tm).size()-1;
		sl.get(i).get(tm).get(s).add(loc);
		sl.get(i).get(tm).get(s).add(tc);
		sl.get(i).get(tm).get(s).add(wc[loc][tc]);
		so.get(loc).get(tc).add(new ArrayList<Integer>());
		so.get(loc).get(tc).get(wc[loc][tc]).add(mn);
		so.get(loc).get(tc).get(wc[loc][tc]).add(i);
		wc[loc][tc]++;
	}
	
	// this functions generates the neurons in this module. Before running this
	// all neuron targets should have been prepared with the connectToModule
	// function on all target modules
	public void generateNeurons(final double[][] neuron_param, final double[][][] learning_param, final int[][] exc_weight_counter, final int[][] inh_weight_counter, ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> syn_loc,ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> syn_orig_exc,ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> syn_orig_inh){
		// neuron parameters
		// neuron_param[0]: excitatory neuron excitory synapse learning rate
		// neuron_param[1]: excitatory neuron inhibitory synapse learning rate
		// neuron_param[2]: inhibitory neuron excitory synapse learning rate
		// neuron_param[3]: inhibitory neuron inhibitory synapse learning rate
		// neuron_param[0]: excitatory neuron signal amplification on excitatory neurons
		// neuron_param[1]: excitatory neuron signal amplification on inhibitory neurons
		// neuron_param[2]: inhibitory neuron signal amplification on excitatory neurons
		// neuron_param[3]: inhibitory neuron signal amplification on inhibitory neurons
		// neuron_param[8]: excitatory neuron excitatory adaption rate
		// neuron_param[9]: excitatory neuron first inhibitory adaption rate
		// neuron_param[10]: excitatory neuron second inhibitory adaption rate
		// neuron_param[11]: inihibitory neuron excitatory adaption rate
		// neuron_param[12]: inhibitory neuron first inhibitory adaption rate
		// neuron_param[13]: inhibitory neuron second inhibitory adaption rate
		// neuron_param[14]: excitatory neuron update weight
		// neuron_param[15]: inhibitory neuron update weight
		// neuron_param[16]: excitatory neuron firing adaptation rate
		// neuron_param[17]: inhibitory neuron firing adaptation rate
		// neuron_param[18]: maximum excitatory firing rate
		// neuron_param[19]: maximum inhibitory firing rate
		// neuron_param[20]: neuron weight initialization
		// neuron_param[21]: excitatory input threshold
		// neuron_param[22]: inhibitory input threshold
		// neuron_param[23]: excitatory time constant
		// neuron_param[24]: inhibitory time constant
		// neuron_param[25]: excitatory excitatory weight balance factor
		// neuron_param[26]: excitatory inhibitory weight balance factor
		// neuron_param[27]: inhibitory excitatory weight balance factor
		// neuron_param[28]: inhibitory inhibitory weight balance factor
		// neuron_param[29]: excitatory neuron excitatory learning transition
		// neuron_param[30]: excitatory neuron inhibitory learning transition
		// neuron_param[31]: excitatory neuron excitatory target firing input
		// neuron_param[32]: excitatory neuron inhibitory target firing
		// neuron_param[33]: inhibitory neuron excitatory learning transition
		// neuron_param[34]: inhibitory neuron inhibitory learning transition
		// neuron_param[35]: inhibitory neuron excitatory target firing input
		// neuron_param[36]: inhibitory neuron inhibitory target firing
		// neuron_param[37]: excitatory neuron background firing
		// neuron_param[38]: excitatory neuron background learning
		// neuron_param[39]: excitatory neuron noise target
		// neuron_param[40]: inhibitory neuron inhibitory background firing
		// neuron_param[41]: inhibitory neuron  background learning
		// neuron_param[42]: inhibitory neuron noise target
		
		// random number generator
		Random rng = new Random();
		// straight forward neuron setup
		this.neurons = new ArrayList<SingleThresholdUnit>();
		// loop over all neurons in this module
		int a = this.w*this.h*this.d_exc;
		for (int i=0; i<this.w*this.h*(this.d_exc+this.d_inh); i++){
			// preparation of the neuron targets and target locations
			int[][][] s = (int[][][])TypeConversion.cpArrayList(syn_loc.get(i),3,"[I");
			// preparation of the initial normalized weight vectors
			double[][] weights_exc = new double[exc_weight_counter[i].length][];
			for (int j=0; j<exc_weight_counter[i].length; j++){
				//weights_exc[j] = VectorFun.mult(Sample.randomUnitVectorL1(exc_weight_counter[i][j],rng),1);
				weights_exc[j] = VectorFun.add(new double[exc_weight_counter[i][j]],1./exc_weight_counter[i][j]);
			}
			double[][] weights_inh = new double[inh_weight_counter[i].length][];
			for (int j=0; j<inh_weight_counter[i].length; j++){
				//weights_inh[j] = VectorFun.mult(Sample.randomUnitVectorL1(inh_weight_counter[i][j],rng),1);
				weights_inh[j] = VectorFun.add(new double[inh_weight_counter[i][j]],1./inh_weight_counter[i][j]);
			}
			int[][][] o_exc = (int[][][])TypeConversion.cpArrayList(syn_orig_exc.get(i),3,"[I");
			int[][][] o_inh = (int[][][])TypeConversion.cpArrayList(syn_orig_inh.get(i),3,"[I");
			if (i<a){
				this.neurons.add(new SingleThresholdUnit(weights_exc,weights_inh,neuron_param[0],neuron_param[1],learning_param[0], learning_param[1],neuron_param[4][0],true,this.target_modules,s,o_exc,o_inh,neuron_param[4][1],neuron_param[4][2],neuron_param[4][3],neuron_param[4][4]));
			}
			else {
				this.neurons.add(new SingleThresholdUnit(weights_exc,weights_inh,neuron_param[2],neuron_param[3],learning_param[2], learning_param[3],neuron_param[5][0],false,this.target_modules,s,o_exc,o_inh,neuron_param[5][1],neuron_param[5][2],neuron_param[5][3],neuron_param[5][4]));
			}
		}
	}
	
	public void forcedUpdate(final double[] input, final int[] input_loc,final ArrayList<ArrayList<SingleThresholdUnit>> list){
		for (int i=0; i<input.length; i++){
			this.neurons.get(input_loc[i]).setRate(input[i]);
			this.neurons.get(input_loc[i]).project(list);
		}
	}
	
	public void forcedUpdate(final double[] input,final ArrayList<ArrayList<SingleThresholdUnit>> list){
		int[] temp = new int[input.length];
		for (int i=0; i<input.length; i++){
			temp[i] = i;
		}
		forcedUpdate(input,temp,list);
	}
	
	// getter functions
	public int getWidth(){
		return this.w;
	}
	
	public int getHeight(){
		return this.h;
	}
	
	public int getD_exc(){
		return this.d_exc;
	}
	
	public int getD_inh(){
		return this.d_inh;
	}
	
	public ArrayList<SingleThresholdUnit> getNeurons(){
		return this.neurons;
	}
}		