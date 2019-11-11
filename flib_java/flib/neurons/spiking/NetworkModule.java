package flib.neurons.spiking;

import java.util.ArrayList;
import java.util.Random;
import flib.math.distributions.LogNormal;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.math.VectorFun;
import flib.math.random.Sample;
import flib.neurons.spiking.SingleThresholdUnit;

public class NetworkModule implements 
java.io.Serializable {
	
	// list of all the neurons in this module
	// does it matter if we store the neuron list inside or outside of the module?
	private ArrayList<SingleThresholdUnit> neurons;
	// width, height and depth of the neuron field
	private int w,h,d_exc, d_inh;
	// list of target modules
	private int[] target_modules;
	
	
	public NetworkModule(int w, int h, int d_exc, int d_inh, final int[] target_modules){
		this.w = w;
		this.h = h;
		this.d_exc = d_exc;
		this.d_inh = d_inh;
		this.target_modules = target_modules.clone();
	}
	
	public void connectToModule(int num_module, int target_module, final NetworkModule M, final double[] syn_connectivity, final double[] geo_connectivity, final int[] exc_weight_counter, final int[] inh_weight_counter, ArrayList<ArrayList<ArrayList<Integer>>> targets, ArrayList<ArrayList<ArrayList<Integer>>> syn_loc, ArrayList<ArrayList<ArrayList<Double>>> delay, ArrayList<ArrayList<ArrayList<Integer>>> syn_orig_exc,ArrayList<ArrayList<ArrayList<Integer>>> syn_orig_inh,ArrayList<ArrayList<Double>> fixed_weights){
		
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
		// geo_connectivity[4]: radial spacing of the spiral
		// geo_connectivity[5]: fixed excitatory input weights
		// geo_connectivity[6]: respacing of the input-output neuron spacing in x-direction
		// geo_connectivity[7]: respacing of the input-output neuron spacing in y-direction
		// geo_connectivity[8]: excitatory delay time between modules
		// geo_connectivity[9]: excitatory distance delay factor
		// geo_connectivity[10]: excitatory delay standard deviation
		// geo_connectivity[11]: inhibitory delay time between modules
		// geo_connectivity[12]: inhibitory distance delay factor between modules
		// geo_connectivity[13]: inhibitory delay standard deviation
		
		// some parameters for different use
		LogNormal lnrm = new LogNormal(1,1);
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
			double[] dist = new double[coord.length];
			double r_sum = 0;
			for (int j=0; j<dist.length; j++){
				// pixel distance to circle center
				dist[j] =Math.sqrt(coord[j][0]*coord[j][0]+coord[j][1]*coord[j][1])*geo_connectivity[4];
				if (dist[j]<=1){
					r[j] = 1;
				}
				// probability weighting depending on the distance to the center
				r[j] = Math.pow(dist[j],geo_connectivity[3]);
				r_sum+=r[j];
			}
			// prepare the probabilities of certain synapse types
			double p_exc = syn_connectivity[0]/(r_sum*d_exc2);
			double p_inh = syn_connectivity[1]/(r_sum*d_inh2);
			// go over all excitatory neurons
			for (int i=0; i<this.w*this.h*this.d_exc; i++){
				x = (i%(w*h))%w;
				y = (i%(w*h))/w;
				// coordinate system recentering
				x2 = (int)((x-xm)*geo_connectivity[6]+xm2);
				y2 = (int)((y-ym)*geo_connectivity[7]+ym2);
				// go over all potential excitatory targets
				int[] temp = NeighborhoodSample.shapeNeighbor2d(coord,w2,h2,x2,y2,(int)geo_connectivity[2]);
				for (int j=0; j<temp.length; j++){
					// if the coordinate exists
					if (temp[j]!=-1){
						// for all excitatory layers in the target module
						for (int k=0; k<d_exc2; k++){
							// if the threshold is not exceeded we for a synapse
							if(rng.nextDouble()<p_exc*r[j]){
								int loc = w2*h2*k+temp[j];
								// neurons should not target themselves...
								if ((this.target_modules[target_module]!=num_module)||(loc!=i)){
									targets.get(i).get(target_module).add(loc);
									syn_loc.get(i).get(target_module).add(exc_weight_counter[loc]);
									delay.get(i).get(target_module).add(lnrm.generate(geo_connectivity[8]+geo_connectivity[9]*dist[j],geo_connectivity[10]));
									syn_orig_exc.get(loc).add(new ArrayList<Integer>());
									syn_orig_exc.get(loc).get(exc_weight_counter[loc]).add(num_module);
									syn_orig_exc.get(loc).get(exc_weight_counter[loc]).add(i);
									fixed_weights.get(loc).add(geo_connectivity[5]);
									exc_weight_counter[loc]++;
								}
							}
						}
						// for all inhibitory layers in the target module
						for (int k=d_exc2; k<d_exc2+d_inh2; k++){
							// if the threshold is not exceeded we for a synapse
							if(rng.nextDouble()<p_inh*r[j]){
								int loc = w2*h2*k+temp[j];
								targets.get(i).get(target_module).add(loc);
								syn_loc.get(i).get(target_module).add(exc_weight_counter[loc]);
								delay.get(i).get(target_module).add(lnrm.generate(geo_connectivity[8]+geo_connectivity[9]*dist[j],geo_connectivity[10]));
								syn_orig_exc.get(loc).add(new ArrayList<Integer>());
								syn_orig_exc.get(loc).get(exc_weight_counter[loc]).add(num_module);
								syn_orig_exc.get(loc).get(exc_weight_counter[loc]).add(i);
								fixed_weights.get(loc).add(geo_connectivity[5]);
								exc_weight_counter[loc]++;
							}
						}
					}
				}
			}
			
			// inhibitory neurons targeting
			// prepare the target spiral shape coordinates
			coord = NeighborhoodSample.circleCoord(geo_connectivity[1]);
			r = new double[coord.length];
			dist = new double[coord.length];
			r_sum = 0;
			for (int j=0; j<dist.length; j++){
				dist[j] =Math.sqrt(coord[j][0]*coord[j][0]+coord[j][1]*coord[j][1])*geo_connectivity[4];
				if (dist[j]<=1){
					r[j] = 1;
				}
				r[j] = Math.pow(dist[j],geo_connectivity[3]);
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
				// go over all potential excitatory targets
				int[] temp = NeighborhoodSample.shapeNeighbor2d(coord,w2,h2,x2,y2,(int)geo_connectivity[2]);
				for (int j=0; j<temp.length; j++){
					if (temp[j]!=-1){
						// for all excitatory layers in the target module
						for (int k=0; k<d_exc2; k++){
							// if the threshold is not exceeded we for a synapse
							if(rng.nextDouble()<p_exc*r[j]){
								int loc = w2*h2*k+temp[j];
								targets.get(i).get(target_module).add(loc);
								syn_loc.get(i).get(target_module).add(inh_weight_counter[loc]);
								delay.get(i).get(target_module).add(lnrm.generate(geo_connectivity[11]+geo_connectivity[12]*dist[j],geo_connectivity[13]));
								syn_orig_inh.get(loc).add(new ArrayList<Integer>());
								syn_orig_inh.get(loc).get(inh_weight_counter[loc]).add(num_module);
								syn_orig_inh.get(loc).get(inh_weight_counter[loc]).add(i);
								inh_weight_counter[loc]++;
							}
						}
						// for all inihibitory layers in the target module
						for (int k=d_exc2; k<d_exc2+d_inh2; k++){
							// if the threshold is not exceeded we for a synapse
							if(rng.nextDouble()<p_inh*r[j]){
								int loc = w2*h2*k+temp[j];
								// neurons should not target themselves...
								if ((this.target_modules[target_module]!=num_module)||(loc!=i)){
									targets.get(i).get(target_module).add(loc);
									syn_loc.get(i).get(target_module).add(inh_weight_counter[loc]);
									delay.get(i).get(target_module).add(lnrm.generate(geo_connectivity[11]+geo_connectivity[12]*dist[j],geo_connectivity[13]));
									syn_orig_inh.get(loc).add(new ArrayList<Integer>());
									syn_orig_inh.get(loc).get(inh_weight_counter[loc]).add(num_module);
									syn_orig_inh.get(loc).get(inh_weight_counter[loc]).add(i);
									inh_weight_counter[loc]++;
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
						// if the threshold is not exceeded we for a synapse
						if(rng.nextDouble()<p_exc){
							int loc = w2*h2*k+j;
							// neurons should not target themselves...
							if ((this.target_modules[target_module]!=num_module)||(loc!=i)){
								targets.get(i).get(target_module).add(loc);
								syn_loc.get(i).get(target_module).add(exc_weight_counter[loc]);
								delay.get(i).get(target_module).add(lnrm.generate(geo_connectivity[8],geo_connectivity[10]));
								syn_orig_exc.get(loc).add(new ArrayList<Integer>());
								syn_orig_exc.get(loc).get(exc_weight_counter[loc]).add(num_module);
								syn_orig_exc.get(loc).get(exc_weight_counter[loc]).add(i);
								fixed_weights.get(loc).add(geo_connectivity[5]);
								exc_weight_counter[loc]++;
							}
						}
					}
					// for all inhibitory layers in the target module
					for (int k=d_exc2; k<d_exc2+d_inh2; k++){
						// if the threshold is not exceeded we for a synapse
						if(rng.nextDouble()<p_inh){
							int loc = w2*h2*k+j;
							targets.get(i).get(target_module).add(loc);
							syn_loc.get(i).get(target_module).add(exc_weight_counter[loc]);
							delay.get(i).get(target_module).add(lnrm.generate(geo_connectivity[8],geo_connectivity[10]));
							syn_orig_exc.get(loc).add(new ArrayList<Integer>());
							syn_orig_exc.get(loc).get(exc_weight_counter[loc]).add(num_module);
							syn_orig_exc.get(loc).get(exc_weight_counter[loc]).add(i);
							fixed_weights.get(loc).add(geo_connectivity[5]);
							exc_weight_counter[loc]++;
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
							targets.get(i).get(target_module).add(loc);
							syn_loc.get(i).get(target_module).add(inh_weight_counter[loc]);
							delay.get(i).get(target_module).add(lnrm.generate(geo_connectivity[11],geo_connectivity[13]));
							syn_orig_inh.get(loc).add(new ArrayList<Integer>());
							syn_orig_inh.get(loc).get(inh_weight_counter[loc]).add(num_module);
							syn_orig_inh.get(loc).get(inh_weight_counter[loc]).add(i);
							inh_weight_counter[loc]++;
						}
					}
					// for all inihibitory layers in the target module
					for (int k=d_exc2; k<d_exc2+d_inh2; k++){
						// if the threshold is not exceeded we for a synapse
						if(rng.nextDouble()<p_inh){
							int loc = w2*h2*k+j;
							// neurons should not target themselves...
							if ((this.target_modules[target_module]!=num_module)||(loc!=i)){
								targets.get(i).get(target_module).add(loc);
								syn_loc.get(i).get(target_module).add(inh_weight_counter[loc]);
								delay.get(i).get(target_module).add(lnrm.generate(geo_connectivity[11],geo_connectivity[13]));
								syn_orig_inh.get(loc).add(new ArrayList<Integer>());
								syn_orig_inh.get(loc).get(inh_weight_counter[loc]).add(num_module);
								syn_orig_inh.get(loc).get(inh_weight_counter[loc]).add(i);
								inh_weight_counter[loc]++;
							}
						}
					}
				}
			}
		}
	}
	
	// this functions generates the neurons in this module. Before running this
	// all neuron targets should have been prepared with the connectToModule
	// function on all target modules
	public void generateNeurons(final double[] neuron_param, final int init, final int[] exc_weight_counter, final int[] inh_weight_counter, ArrayList<ArrayList<ArrayList<Integer>>> targets, ArrayList<ArrayList<ArrayList<Integer>>> syn_loc, ArrayList<ArrayList<ArrayList<Double>>> delay, ArrayList<ArrayList<ArrayList<Integer>>> syn_orig_exc,ArrayList<ArrayList<ArrayList<Integer>>> syn_orig_inh,ArrayList<ArrayList<Double>> fixed_weights){
		// neuron parameters
		// neuron_param[0]: excitatory neuron excitory synapse learning rate
		// neuron_param[1]: excitatory neuron excitory synapse learning factor
		// neuron_param[2]: excitatory neuron first inhibitory synapse learning rate
		// neuron_param[3]: excitatory neuron second inhibitory synapse learning rate
		// neuron_param[4]: excitatory neuron inhibitory synapse learning factor
		// neuron_param[5]: inhibitory neuron excitory synapse learning rate
		// neuron_param[6]: inhibitory neuron excitory synapse learning factor
		// neuron_param[7]: inhibitory neuron first inhibitory synapse learning rate
		// neuron_param[8]: inhibitory neuron second inhibitory synapse learning rate
		// neuron_param[9]: excitatory neuron inhibitory synapse learning factor
		// neuron_param[10]: excitatory target excitatory input
		// neuron_param[11]: excitatory target inhibitory input
		// neuron_param[12]: excitatory excitatory amplitude learning factor
		// neuron_param[13]: excitatory inhibtitory amplitude learning factor
		// neuron_param[14]: inhibitory target excitatory input
		// neuron_param[15]: inhibitory target inhibitory input
		// neuron_param[16]: inhibitory excitatory amplitude learning factor
		// neuron_param[17]: inhibitory inhibtitory amplitude learning factor
		// neuron_param[18]: initial excitatory excitatory amplitude
		// neuron_param[19]: initial excitatory inhibtitory amplitude
		// neuron_param[20]: initial inhibtitory excitatory amplitude
		// neuron_param[21]: initial inhibtitory inhibtitory amplitude
		// neuron_param[22]: excitatory neuron refractory length
		// neuron_param[23]: inhibitory neuron refractory length
		// neuron_param[24]: excitatory neuron firing threshold
		// neuron_param[25]: inhibitory neuron firing threshold
		// neuron_param[26]: excitatory neuron decay rate
		// neuron_param[27]: inhibitory neuron decay rate
		// neuron_param[28]: excitatory neuron hyperpolization value
		// neuron_param[29]: inhibitory neuron hyperpolization value
		// neuron_param[30]: neuron interspike interval averaging factor
		// neuron_param[31]: neuron excitatory excitatory weight based learning adaptation
		// neuron_param[32]: neuron excitatory inhibitory weight based learning adaptation
		// neuron_param[33]: neuron inhibitory excitatory weight based learning adaptation
		// neuron_param[34]: neuron inhibitory inhibitory weight based learning adaptation
		
		// random number generator
		Random rng = new Random();
		// straight forward neuron setup
		this.neurons = new ArrayList<SingleThresholdUnit>();
		// loop over all excitatory neurons in this module
		for (int i=0; i<this.w*this.h*this.d_exc; i++){
			double[] learning = new double[5];
			for (int j=0; j<5; j++){
				learning[j] = neuron_param[j];
			}
			double[] rate = new double[4];
			for (int j=0; j<4; j++){
				rate[j] = neuron_param[j+2*5];
			}
			// preparation of the neuron targets and target locations
			int[][] t = new int[targets.get(i).size()][];
			int[][] s = new int[targets.get(i).size()][];
			double[][] d = new double[targets.get(i).size()][];
			for (int j=0; j<t.length; j++){
				t[j] = new int[targets.get(i).get(j).size()];
				s[j] = new int[targets.get(i).get(j).size()];
				d[j] = new double[targets.get(i).get(j).size()];
				for (int k=0; k<t[j].length; k++){
					t[j][k] = targets.get(i).get(j).get(k);
					s[j][k] = syn_loc.get(i).get(j).get(k);
					d[j][k] = delay.get(i).get(j).get(k);
				}
			}
			int counter = 0;
			boolean[] weights_fixed = new boolean[fixed_weights.get(i).size()];
			for (int j=0; j<weights_fixed.length; j++){
				weights_fixed[j] = (fixed_weights.get(i).get(j)>0);
				if (weights_fixed[j]){
					counter++;
				}
			}
			// preparation of the initial normalized weight vectors
			double[] weights_exc, weights_inh;
			if (false){
				weights_exc = VectorFun.add(new double[exc_weight_counter[i]],1/exc_weight_counter[i]);
				for (int j=0; j<weights_fixed.length; j++){
					if (weights_fixed[j]){
						weights_exc[j] = fixed_weights.get(i).get(j);
					}
				}
				weights_inh = VectorFun.add(new double[inh_weight_counter[i]],1/inh_weight_counter[i]);
			}
			else {
				weights_exc = new double[exc_weight_counter[i]];
				double[] temp1 = VectorFun.mult(Sample.randomUnitVectorL1(exc_weight_counter[i]-counter,rng),1);
				int counter2 = 0;
				for (int j=0; j<weights_fixed.length; j++){
					if (weights_fixed[j]){
						weights_exc[j] = fixed_weights.get(i).get(j);
					}
					else {
						weights_exc[j] = temp1[counter2];
						counter2++;
					}
				}
				weights_inh = VectorFun.mult(Sample.randomUnitVectorL1(inh_weight_counter[i],rng),1);
			}
			int[][] o_exc = new int[syn_orig_exc.get(i).size()][2];
			for (int j=0; j<syn_orig_exc.get(i).size(); j++){
				o_exc[j][0] = syn_orig_exc.get(i).get(j).get(0);
				o_exc[j][1] = syn_orig_exc.get(i).get(j).get(1);
			}
			int[][] o_inh = new int[syn_orig_inh.get(i).size()][2];
			for (int j=0; j<syn_orig_inh.get(i).size(); j++){
				o_inh[j][0] = syn_orig_inh.get(i).get(j).get(0);
				o_inh[j][1] = syn_orig_inh.get(i).get(j).get(1);
			}
			this.neurons.add(new SingleThresholdUnit(weights_exc,weights_inh,weights_fixed,target_modules,t,s,o_exc,o_inh,d,learning,rate,true,neuron_param[18],neuron_param[19],neuron_param[22],neuron_param[24],neuron_param[26],neuron_param[28],neuron_param[30],neuron_param[31],neuron_param[32]));
		}
		// loop over all inhibitory neurons in this module
		for (int i=this.w*this.h*this.d_exc; i<this.w*this.h*(this.d_exc+this.d_inh); i++){
			double[] learning = new double[5];
			for (int j=0; j<5; j++){
				learning[j] = neuron_param[j+5];
			}
			double[] rate = new double[4];
			for (int j=0; j<4; j++){
				rate[j] = neuron_param[j+2*5+4];
			}
			// preparation of the neuron targets and target locations
			int[][] t = new int[targets.get(i).size()][];
			int[][] s = new int[targets.get(i).size()][];
			double[][] d = new double[targets.get(i).size()][];
			for (int j=0; j<t.length; j++){
				t[j] = new int[targets.get(i).get(j).size()];
				s[j] = new int[targets.get(i).get(j).size()];
				d[j] = new double[targets.get(i).get(j).size()];
				for (int k=0; k<t[j].length; k++){
					t[j][k] = targets.get(i).get(j).get(k);
					s[j][k] = syn_loc.get(i).get(j).get(k);
					d[j][k] = delay.get(i).get(j).get(k);
				}
			}
			int counter = 0;
			boolean[] weights_fixed = new boolean[fixed_weights.get(i).size()];
			for (int j=0; j<weights_fixed.length; j++){
				weights_fixed[j] = (fixed_weights.get(i).get(j)>0);
				if (weights_fixed[j]){
					counter++;
				}
			}
						// preparation of the initial normalized weight vectors
			double[] weights_exc, weights_inh;
			if (false){
				weights_exc = VectorFun.add(new double[exc_weight_counter[i]],1/exc_weight_counter[i]);
				for (int j=0; j<weights_fixed.length; j++){
					if (weights_fixed[j]){
						weights_exc[j] = fixed_weights.get(i).get(j);
					}
				}
				weights_inh = VectorFun.add(new double[inh_weight_counter[i]],1/inh_weight_counter[i]);
			}
			else {
				weights_exc = new double[exc_weight_counter[i]];
				double[] temp1 = VectorFun.mult(Sample.randomUnitVectorL1(exc_weight_counter[i]-counter,rng),1);
				int counter2 = 0;
				for (int j=0; j<weights_fixed.length; j++){
					if (weights_fixed[j]){
						weights_exc[j] = fixed_weights.get(i).get(j);
					}
					else {
						weights_exc[j] = temp1[counter2];
						counter2++;
					}
				}
				weights_inh = VectorFun.mult(Sample.randomUnitVectorL1(inh_weight_counter[i],rng),1);
			}
			int[][] o_exc = new int[syn_orig_exc.get(i).size()][2];
			for (int j=0; j<syn_orig_exc.get(i).size(); j++){
				o_exc[j][0] = syn_orig_exc.get(i).get(j).get(0);
				o_exc[j][1] = syn_orig_exc.get(i).get(j).get(1);
			}
			int[][] o_inh = new int[syn_orig_inh.get(i).size()][2];
			for (int j=0; j<syn_orig_inh.get(i).size(); j++){
				o_inh[j][0] = syn_orig_inh.get(i).get(j).get(0);
				o_inh[j][1] = syn_orig_inh.get(i).get(j).get(1);
			}
			this.neurons.add(new SingleThresholdUnit(weights_exc,weights_inh,weights_fixed,target_modules,t,s,o_exc,o_inh,d,learning,rate,false,neuron_param[20],neuron_param[21],neuron_param[23],neuron_param[25],neuron_param[27],neuron_param[29],neuron_param[30],neuron_param[33],neuron_param[34]));
		}
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