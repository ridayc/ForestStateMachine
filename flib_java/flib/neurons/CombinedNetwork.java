package flib.neurons;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.TreeSet;
import java.util.Iterator;
import ij.ImagePlus;
import flib.neurons.SingleThresholdUnit;
import flib.neurons.NetworkModule;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.SortPair;
import flib.datastructures.TypeConversion;
import flib.ij.stack.StackOperations;

public class CombinedNetwork implements 
java.io.Serializable {
	
	// we have all types of private variables to store for the sake of later transparency
	// list of all neurons in all modules
	private ArrayList<ArrayList<SingleThresholdUnit>> neurons;
	// list of all modules in the network
	private ArrayList<NetworkModule> modules;
	// update probaility
	private double[] update_prob;
	// alternatively the update tree for non-random updating
	private TreeSet<SortPair> update_list;
	// current time for update list
	private double time;
	// average firing rates of individual neurons
	private double[][] avg_rates;
	// average firing rate variance of individual neurons
	private double[][] avg_variance;
	// window for rate averaging
	private double avg_window;
	// Module neuron variables
	private double[][][] neuron_param;
	// learning parameters for all modules
	private double[][][][] learning_param;
	// inter-module synapse connectivity values
	private double[][][] syn_connectivity;
	// inter-module geometric connectivity
	private double[][][] geo_connectivity;
	// module size parameters
	private int[][] module_size;
	// target module numbers of each module
	private int[][] module_targets;
	// target compartments for each module target
	private int[][][] target_compartments;
	// lookup lists
	private int[] neuron_module;
	private int[] neuron_loc;
	
	public CombinedNetwork(final int[][] module_size, final int[][] module_targets, final int[][][] target_compartments, final double[][][] neuron_param, final double[][][][] learning_param, final double[][][] syn_connectivity, final double[][][] geo_connectivity, double avg_window){
		// module_size: contains the dimensions of each module
		// module_size[i][0]: module width
		// module_size[i][1]: module height
		// module_size[i][2]: module number of excitatory layers
		// module_size[i][3]: module number of inhibitory layers
		// module_size[i][4]: number of excitatory compartments of the excitatory population
		// module_size[i][5]: number of inhibitory compartments of the excitatory population
		// module_size[i][6]: number of excitatory compartments of the inhibitory population
		// module_size[i][7]: number of inhibitory compartments of the inhibitory population
		
		// number of modules
		int m = module_size.length;
		// Initialize all Modules
		this.modules = new ArrayList<NetworkModule>();
		this.module_size = (int[][])TypeConversion.copyMultiArrayObject(module_size);
		this.module_targets = (int[][])TypeConversion.copyMultiArrayObject(module_targets);
		this.target_compartments = (int[][][])TypeConversion.copyMultiArrayObject(target_compartments);
		for (int i=0; i<m; i++){
			this.modules.add(new NetworkModule(this.module_size[i][0],this.module_size[i][1],this.module_size[i][2],this.module_size[i][3], this.module_targets[i]));
		}
		// copy all other inputs
		this.neuron_param = (double[][][])TypeConversion.copyMultiArrayObject(neuron_param);
		this.learning_param = (double[][][][])TypeConversion.copyMultiArrayObject(learning_param);
		this.syn_connectivity = (double[][][])TypeConversion.copyMultiArrayObject(syn_connectivity);
		this.geo_connectivity = (double[][][])TypeConversion.copyMultiArrayObject(geo_connectivity);
		
		// preparation of variables for the module connecting
		int[][][] exc_weight_counter = new int[m][][];
		int[][][] inh_weight_counter = new int[m][][];
		ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>> syn_loc = new ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>>();
		ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>> syn_orig_exc = new ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>>();
		ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>> syn_orig_inh = new ArrayList<ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>>();
		for (int i=0; i<m; i++){
			int a = this.module_size[i][0]*this.module_size[i][1]*this.module_size[i][2];
			int b = this.module_size[i][0]*this.module_size[i][1]*this.module_size[i][3];
			int num_neuron = a+b;
			exc_weight_counter[i] = new int[num_neuron][];
			inh_weight_counter[i] = new int[num_neuron][];
			for (int j=0; j<a; j++){
				exc_weight_counter[i][j] = new int[module_size[i][4]];
				inh_weight_counter[i][j] = new int[module_size[i][5]];
			}
			for (int j=a; j<num_neuron; j++){
				exc_weight_counter[i][j] = new int[module_size[i][6]];
				inh_weight_counter[i][j] = new int[module_size[i][7]];
			}
			syn_loc.add(new ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>());
			syn_orig_exc.add(new ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>());
			syn_orig_inh.add(new ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>());
			for (int j=0; j<num_neuron; j++){
				syn_loc.get(i).add(new ArrayList<ArrayList<ArrayList<Integer>>>());
				syn_orig_exc.get(i).add(new ArrayList<ArrayList<ArrayList<Integer>>>());
				syn_orig_inh.get(i).add(new ArrayList<ArrayList<ArrayList<Integer>>>());
				for (int k=0; k<this.module_targets[i].length; k++){
					syn_loc.get(i).get(j).add(new ArrayList<ArrayList<Integer>>());
				}
				if (j<a){
					for (int k=0; k<this.module_size[i][4]; k++){
						syn_orig_exc.get(i).get(j).add(new ArrayList<ArrayList<Integer>>());
					}
					for (int k=0; k<this.module_size[i][5]; k++){
						syn_orig_inh.get(i).get(j).add(new ArrayList<ArrayList<Integer>>());
					}
				}
				else {
					for (int k=0; k<this.module_size[i][6]; k++){
						syn_orig_exc.get(i).get(j).add(new ArrayList<ArrayList<Integer>>());
					}
					for (int k=0; k<this.module_size[i][7]; k++){
						syn_orig_inh.get(i).get(j).add(new ArrayList<ArrayList<Integer>>());
					}
				}
			}
		}
		
		// now we explicitly create the synaptic targets for all the neurons in all different modules
		for (int i=0; i<m; i++){
			for (int j=0; j<this.module_targets[i].length; j++){
				this.modules.get(i).connectToModule(i,j,this.target_compartments[i][j],this.modules.get(this.module_targets[i][j]),this.syn_connectivity[i][j],this.geo_connectivity[i][j],exc_weight_counter[this.module_targets[i][j]],inh_weight_counter[this.module_targets[i][j]],syn_loc.get(i),syn_orig_exc.get(this.module_targets[i][j]),syn_orig_inh.get(this.module_targets[i][j]));
			}
		}
		
		// based on above connectivity we generate the neurons in all modules
		// add the module neuron lists to the global neuron list
		this.neurons = new ArrayList<ArrayList<SingleThresholdUnit>>();
		for (int i=0; i<m; i++){
			this.modules.get(i).generateNeurons(this.neuron_param[i],this.learning_param[i],exc_weight_counter[i],inh_weight_counter[i],syn_loc.get(i),syn_orig_exc.get(i),syn_orig_inh.get(i));
			this.neurons.add(this.modules.get(i).getNeurons());
		}
		
		// setting up the neuron update probabilities, and where these are located according to module
		int n = 0;
		for (int i=0; i<m; i++){
			n+=this.neurons.get(i).size();
		}
		this.update_prob = new double[n];
		this.neuron_module = new int[n];
		this.neuron_loc = new int[n];
		int counter = 0;
		for (int i=0; i<m; i++){
			for (int j=0; j<this.neurons.get(i).size(); j++){
				this.neuron_module[counter] = i;
				this.neuron_loc[counter] = j;
				this.update_prob[counter] = this.neurons.get(i).get(j).getUpdate();
				counter++;
			}
		}
		// normalization of the update vector
		update_prob = VectorFun.cumsum(update_prob);
		VectorFun.normi(update_prob);
		
		//update list initialization
		update_list = new TreeSet<SortPair>();
		time = 0;
		for (int i=0; i<neuron_module.length; i++){
			update_list.add(new SortPair(time+1./neurons.get(neuron_module[i]).get(neuron_loc[i]).getUpdate(),(double)i));
		}
		
		// average neuron firing rates
		this.avg_window = avg_window;
		this.avg_rates = new double[m][];
		this.avg_variance = new double[m][];
		for (int i=0; i<m; i++){
			this.avg_rates[i] = new double[neurons.get(i).size()];
			this.avg_variance[i] = new double[neurons.get(i).size()];
		}
	}
	
	// iterate through n randomly choosen neurons in the population and update their weights as well
	public void iterate_free(int n, final Random rng){
		for (int i=0; i<n; i++){
			double r = rng.nextDouble();
			int a = Arrays.binarySearch(update_prob,r);
			if (a<0){
				a = -(a+1);
			}
			neurons.get(neuron_module[a]).get(neuron_loc[a]).updateWeightsFireWire();
			neurons.get(neuron_module[a]).get(neuron_loc[a]).getFrequency();
			double b = neurons.get(neuron_module[a]).get(neuron_loc[a]).project(neurons);
			avg_rates[neuron_module[a]][neuron_loc[a]] = (avg_rates[neuron_module[a]][neuron_loc[a]]*avg_window+b)/(avg_window+1);
			avg_variance[neuron_module[a]][neuron_loc[a]] = (avg_variance[neuron_module[a]][neuron_loc[a]]*avg_window+(b-avg_rates[neuron_module[a]][neuron_loc[a]])*(b-avg_rates[neuron_module[a]][neuron_loc[a]]))/(avg_window+1);
		}
	}
	
	// iterate through n randomly choosen neurons in the population and update their weights as well
	public void iterate_fixed(int n, final Random rng){
		int counter = 0;
		int counter2 = 0;
		SortPair b;
		double a = 0;
		int c;
		if (n>0){
			b = update_list.first();
			a = b.getValue();
		}
		Iterator<SortPair> itr;
		while (counter<n){
			itr = update_list.iterator();
			while (itr.hasNext()&&counter<n){
				b = itr.next();
				if(a==b.getValue()){
					c = (int)b.getOriginalIndex();
					neurons.get(neuron_module[c]).get(neuron_loc[c]).updateWeightsFireWire();
					neurons.get(neuron_module[c]).get(neuron_loc[c]).getFrequency();
					counter2++;
					counter++;
				}
				else {
					time = b.getValue();
					break;
				}
			}
			while(counter2>0){
				b = update_list.first();
				c = (int)b.getOriginalIndex();
				update_list.remove(b);
				double d = neurons.get(neuron_module[c]).get(neuron_loc[c]).project(neurons);
				update_list.add(new SortPair(a+1./neurons.get(neuron_module[c]).get(neuron_loc[c]).getUpdate(),(double)c));
				counter2--;
				avg_rates[neuron_module[c]][neuron_loc[c]] = (avg_rates[neuron_module[c]][neuron_loc[c]]*avg_window+d)/(avg_window+1);
				avg_variance[neuron_module[c]][neuron_loc[c]] = (avg_variance[neuron_module[c]][neuron_loc[c]]*avg_window+(d-avg_rates[neuron_module[c]][neuron_loc[c]])*(d-avg_rates[neuron_module[c]][neuron_loc[c]]))/(avg_window+1);
			}
			a = time;
		}
	}
	
	public void reset(){
		for (int i=0; i<neurons.size(); i++){
			this.avg_rates[i] = new double[neurons.get(i).size()];
			this.avg_variance[i] = new double[neurons.get(i).size()];
			for (int j=0; j<neurons.get(i).size(); j++){
				neurons.get(i).get(j).setRate(0);
				for (int k=0; k<neurons.get(i).get(j).getWeights_exc().length; k++){
					for (int l=0; l<neurons.get(i).get(j).getWeights_exc()[k].length; l++){
						neurons.get(i).get(j).updateInput(true,0,k,l);
					}
				}
				for (int k=0; k<neurons.get(i).get(j).getWeights_inh().length; k++){
					for (int l=0; l<neurons.get(i).get(j).getWeights_inh()[k].length; l++){
						neurons.get(i).get(j).updateInput(false,0,k,l);
					}
				}
			}
		}
		// reinitiate the update list
		update_list = new TreeSet<SortPair>();
		time = 0;
		for (int i=0; i<neuron_module.length; i++){
			update_list.add(new SortPair(time+1./neurons.get(neuron_module[i]).get(neuron_loc[i]).getUpdate(),(double)i));
		}
	}
	
	public ArrayList<ArrayList<ImagePlus>> run(final ArrayList<ImagePlus> imp_in, int numit, final int[] update_modules, final int[] update_rate, int free_epoch, int fixed_epoch, int output, int reset, int[] current_im){
		// imp: is an ArrayList which contains the input frames for the driver modules
		// numit: total number of iterations for the run
		// update_modules: which modules are to be driven with external input
		// update_rate: rate at which the individual driver modules are updated
		// epoch: number of neuron updates until an iteration is counted
		// output: how frequently all the modules should be written to output
		// current_im: frame number at which the individual modules should start receiving input from
		
		// Random number generator
		Random rng = new Random();
		// prepare the output ArrayList
		ArrayList<ArrayList<ImagePlus>> out = new ArrayList<ArrayList<ImagePlus>>();
		for (int i=0; i<neurons.size(); i++){
			out.add(new ArrayList<ImagePlus>());
		}
		for (int i=0; i<numit; i++){
			// resetting
			if (i%reset==0&&i>0){
				reset();
			}
			// updating
			for (int j=0; j<update_modules.length; j++){
				if (i%update_rate[j]==0){
					current_im[j]%=imp_in.get(j).getNSlices();
					double[] x = VectorConv.float2double((float[])(imp_in.get(j).getImageStack().getProcessor(current_im[j]+1).convertToFloat().getPixels()));
					this.modules.get(update_modules[j]).forcedUpdate(x,neurons);
					current_im[j]++;
				}
			}
			// update in a fixed order depending on firing time
			iterate_fixed(fixed_epoch,rng);
			// update random individual neurons
			iterate_free(free_epoch,rng);
			// if it's time to produce output, then do that
			if (i%output==0){
				for (int j=0; j<neurons.size(); j++){
					out.get(j).add(getNeuronStates(j));
				}
			}
		}
		return out;
	}		
	
	public int[][][][] weightHistogram(int numbin){
		int[][][][] histo = new int[neurons.size()][4][][];
		for (int i=0; i<neurons.size(); i++){
			int a, b, c;
			a = 0;
			// all excitatory neurons
			if (modules.get(i).getD_exc()>0){
				a = modules.get(i).getWidth()*modules.get(i).getHeight()*modules.get(i).getD_exc();
				b = neurons.get(i).get(0).getCompartmentNumber_exc();
				c = neurons.get(i).get(0).getCompartmentNumber_inh();
				histo[i][0] = new int[b][numbin];
				histo[i][1] = new int[c][numbin];
				for (int j=0; j<a; j++){
					// excitatory synapses
					histoComponent(histo[i][0],neurons.get(i).get(j).getWeights_exc());
					// inhibitory synapses
					histoComponent(histo[i][1],neurons.get(i).get(j).getWeights_inh());
				}
			}
			// all inhibitory neurons
			if (modules.get(i).getD_inh()>0){
				b = neurons.get(i).get(a).getCompartmentNumber_exc();
				c = neurons.get(i).get(a).getCompartmentNumber_inh();
				histo[i][2] = new int[b][numbin];
				histo[i][3] = new int[c][numbin];
				for (int j=a; j<neurons.get(i).size(); j++){
					// excitatory synapses
					histoComponent(histo[i][2],neurons.get(i).get(j).getWeights_exc());
					// inhibitory synapses
					histoComponent(histo[i][3],neurons.get(i).get(j).getWeights_inh());
				}
			}
		}
		return histo;
	}
	
	private void histoComponent(int[][] histo, final double[][] temp){
		int numbin = histo[0].length;
		for (int i=0; i<temp.length; i++){
			for (int j=0; j<temp[i].length; j++){
				if (temp[i][j]>=1){
					histo[i][numbin-1]++;
				}
				else {
					histo[i][(int)(temp[i][j]*numbin)]++;
				}
			}
		}
	}
	
	public int[][][][] outHistogram(int numbin){
		int[][][][] histo = new int[neurons.size()][4][][];
		for (int i=0; i<neurons.size(); i++){
			int a, b, c;
			a = 0;
			// all excitatory neurons
			if (modules.get(i).getD_exc()>0){
				a = modules.get(i).getWidth()*modules.get(i).getHeight()*modules.get(i).getD_exc();
				b = module_targets[i].length;
				histo[i][0] = new int[b][numbin];
				histo[i][2] = new int[b][numbin];
				for (int j=0; j<a; j++){
					ArrayList<ArrayList<Double>> temp = new ArrayList<ArrayList<Double>>();
					ArrayList<ArrayList<Double>> temp2 = new ArrayList<ArrayList<Double>>();
					for (int k=0; k<neurons.get(i).get(j).getSynapseLocations().length; k++){
						temp.add(new ArrayList<Double>());
						temp2.add(new ArrayList<Double>());
						for (int l=0; l<neurons.get(i).get(j).getSynapseLocations()[k].length; l++){
							int[] d = neurons.get(i).get(j).getSynapseLocations()[k][l];
							int t = module_targets[i][k];
							if (d[0]<modules.get(t).getWidth()*modules.get(t).getHeight()*modules.get(t).getD_exc()){
								temp.get(k).add(neurons.get(t).get(d[0]).getWeights_exc()[d[1]][d[2]]);
							}
							else {
								temp2.get(k).add(neurons.get(t).get(d[0]).getWeights_exc()[d[1]][d[2]]);
							}
						}
					}
					double[][] vec = new double[temp.size()][];
					for (int k=0; k<vec.length; k++){
						vec[k] = new double[temp.get(k).size()];
						for (int l=0; l<temp.get(k).size(); l++){
							vec[k][l] = temp.get(k).get(l);
						}
					}
					for (int k=0; k<vec.length; k++){
						if (vec[k].length>0){
							VectorFun.multi(vec[k],1./VectorFun.max(vec[k])[0]);
						}
					}
					histoComponent(histo[i][0],vec);
					vec = new double[temp2.size()][];
					for (int k=0; k<vec.length; k++){
						vec[k] = new double[temp2.get(k).size()];
						for (int l=0; l<temp2.get(k).size(); l++){
							vec[k][l] = temp2.get(k).get(l);
						}
					}
					for (int k=0; k<vec.length; k++){
						if (vec[k].length>0){
							VectorFun.multi(vec[k],1./VectorFun.max(vec[k])[0]);
						}
					}
					histoComponent(histo[i][2],vec);
				}
			}
			// all inhibitory neurons
			if (modules.get(i).getD_inh()>0){
				b = module_targets[i].length;
				histo[i][1] = new int[b][numbin];
				histo[i][3] = new int[b][numbin];
				for (int j=a; j<neurons.get(i).size(); j++){
					ArrayList<ArrayList<Double>> temp = new ArrayList<ArrayList<Double>>();
					ArrayList<ArrayList<Double>> temp2 = new ArrayList<ArrayList<Double>>();
					for (int k=0; k<neurons.get(i).get(j).getSynapseLocations().length; k++){
						temp.add(new ArrayList<Double>());
						temp2.add(new ArrayList<Double>());
						for (int l=0; l<neurons.get(i).get(j).getSynapseLocations()[k].length; l++){
							int[] d = neurons.get(i).get(j).getSynapseLocations()[k][l];
							int t = module_targets[i][k];
							if (d[0]<modules.get(t).getWidth()*modules.get(t).getHeight()*modules.get(t).getD_exc()){
								temp.get(k).add(neurons.get(t).get(d[0]).getWeights_inh()[d[1]][d[2]]);
							}
							else {
								temp2.get(k).add(neurons.get(t).get(d[0]).getWeights_inh()[d[1]][d[2]]);
							}
						}
					}
					double[][] vec = new double[temp.size()][];
					for (int k=0; k<vec.length; k++){
						vec[k] = new double[temp.get(k).size()];
						for (int l=0; l<temp.get(k).size(); l++){
							vec[k][l] = temp.get(k).get(l);
						}
					}
					for (int k=0; k<vec.length; k++){
						if (vec[k].length>0){
							VectorFun.multi(vec[k],1./VectorFun.max(vec[k])[0]);
						}
					}
					histoComponent(histo[i][1],vec);
					vec = new double[temp2.size()][];
					for (int k=0; k<vec.length; k++){
						vec[k] = new double[temp2.get(k).size()];
						for (int l=0; l<temp2.get(k).size(); l++){
							vec[k][l] = temp2.get(k).get(l);
						}
					}
					for (int k=0; k<vec.length; k++){
						if (vec[k].length>0){
							VectorFun.multi(vec[k],1./VectorFun.max(vec[k])[0]);
						}
					}
					histoComponent(histo[i][3],vec);
				}
			}
		}
		return histo;
	}
	
	public int[][][][] inHistogram(int numbin){
		int[][][][] histo = new int[neurons.size()][4][][];
		for (int i=0; i<neurons.size(); i++){
			int a, b, c;
			a = 0;
			// all excitatory neurons
			if (modules.get(i).getD_exc()>0){
				a = modules.get(i).getWidth()*modules.get(i).getHeight()*modules.get(i).getD_exc();
				b = neurons.get(i).get(0).getCompartmentNumber_exc();
				c = neurons.get(i).get(0).getCompartmentNumber_inh();
				histo[i][0] = new int[b][numbin];
				histo[i][1] = new int[c][numbin];
				for (int j=0; j<a; j++){
					double[][] temp = new double[neurons.get(i).get(j).getWeights_exc().length][];
					for (int k=0; k<temp.length; k++){
						temp[k] = neurons.get(i).get(j).getWeights_exc()[k].clone();
						if (temp[k].length>0){
							VectorFun.multi(temp[k],1./VectorFun.max(temp[k])[0]);
						}
					}
					// excitatory synapses
					histoComponent(histo[i][0],temp);
					temp = new double[neurons.get(i).get(j).getWeights_inh().length][];
					for (int k=0; k<temp.length; k++){
						temp[k] = neurons.get(i).get(j).getWeights_inh()[k].clone();
						if (temp[k].length>0){
							VectorFun.multi(temp[k],1./VectorFun.max(temp[k])[0]);
						}
					}
					// inhibitory synapses
					histoComponent(histo[i][1],temp);
				}
			}
			// all inhibitory neurons
			if (modules.get(i).getD_inh()>0){
				b = neurons.get(i).get(a).getCompartmentNumber_exc();
				c = neurons.get(i).get(a).getCompartmentNumber_inh();
				histo[i][2] = new int[b][numbin];
				histo[i][3] = new int[c][numbin];
				for (int j=a; j<neurons.get(i).size(); j++){
					double[][] temp = new double[neurons.get(i).get(j).getWeights_exc().length][];
					for (int k=0; k<temp.length; k++){
						temp[k] = neurons.get(i).get(j).getWeights_exc()[k].clone();
						VectorFun.multi(temp[k],1./VectorFun.max(temp[k])[0]);
					}
					// excitatory synapses
					histoComponent(histo[i][2],temp);
					temp = new double[neurons.get(i).get(j).getWeights_inh().length][];
					for (int k=0; k<temp.length; k++){
						temp[k] = neurons.get(i).get(j).getWeights_inh()[k].clone();
						if (temp[k].length>0){
							VectorFun.multi(temp[k],1./VectorFun.max(temp[k])[0]);
						}
					}
					// inhibitory synapses
					histoComponent(histo[i][3],temp);
				}
			}
		}
		return histo;
	}
	
	// getter functions
	
	public ArrayList<ArrayList<SingleThresholdUnit>> getNeurons(){
		return this.neurons;
	}
	
	public ArrayList<NetworkModule> getModules(){
		return this.modules;
	}
	
	public ImagePlus getNeuronStates(int module_number){
		int w = this.module_size[module_number][0];
		int h = this.module_size[module_number][1];
		int d = this.module_size[module_number][2]+this.module_size[module_number][3];
		return StackOperations.convert2Stack(avg_rates[module_number],w,h,d);
	}
	
	public ImagePlus getNeuronStd(int module_number){
		int w = this.module_size[module_number][0];
		int h = this.module_size[module_number][1];
		int d = this.module_size[module_number][2]+this.module_size[module_number][3];
		return StackOperations.convert2Stack(VectorFun.sqrt(avg_variance[module_number]),w,h,d);
	}
	
	public ImagePlus getAvg_exc_exc(int module_number, int compartment){
		int w = this.module_size[module_number][0];
		int h = this.module_size[module_number][1];
		int d = this.module_size[module_number][2];
		int a = module_number;
		double[] x = new double[w*h*d];
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			x[i] = neurons.get(module_number).get(i).getAvg_exc()[compartment];
		}
		return StackOperations.convert2Stack(x,w,h,d);
	}
	
	public ImagePlus getAvg_exc_inh(int module_number, int compartment){
		int w = this.module_size[module_number][0];
		int h = this.module_size[module_number][1];
		int d = this.module_size[module_number][2];
		int a = module_number;
		double[] x = new double[w*h*d];
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			x[i] = neurons.get(module_number).get(i).getAvg_inh()[compartment];
		}
		return StackOperations.convert2Stack(x,w,h,d);
	}
	
	public ImagePlus getAvg_inh_exc(int module_number, int compartment){
		int w = this.module_size[module_number][0];
		int h = this.module_size[module_number][1];
		int d = this.module_size[module_number][3];
		int a = module_number;
		double[] x = new double[w*h*d];
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			x[i-this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]] = neurons.get(module_number).get(i).getAvg_exc()[compartment];
		}
		return StackOperations.convert2Stack(x,w,h,d);
	}
	
	public ImagePlus getAvg_inh_inh(int module_number, int compartment){
		int w = this.module_size[module_number][0];
		int h = this.module_size[module_number][1];
		int d = this.module_size[module_number][3];
		int a = module_number;
		double[] x = new double[w*h*d];
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			x[i-this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]] = neurons.get(module_number).get(i).getAvg_inh()[compartment];
		}
		return StackOperations.convert2Stack(x,w,h,d);
	}
	
	public TreeSet<SortPair> getUpdateList(){
		return this.update_list;
	}
	
	// setter functions
	// learning rates
	
	public void setAlpha_exc_exc(double alpha){
		for (int i=0; i<neurons.size(); i++){
			setAlpha_exc_exc(alpha,i);
		}
	}
	
	public void setAlpha_exc_exc(double alpha, int module_number){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setAlpha_exc(alpha);
		}
	}
	
	public void setAlpha_exc_exc(double alpha, int module_number, int compartment){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setAlpha_exc(alpha,compartment);
		}
	}
	
	public void setAlpha_exc_inh(double alpha){
		for (int i=0; i<neurons.size(); i++){
			setAlpha_exc_inh(alpha,i);
		}
	}
	
	public void setAlpha_exc_inh(double alpha, int module_number){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setAlpha_inh(alpha);
		}
	}
	
	public void setAlpha_exc_inh(double alpha, int module_number, int compartment){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setAlpha_inh(alpha,compartment);
		}
	}
	
	public void setAlpha_inh_exc(double alpha){
		for (int i=0; i<neurons.size(); i++){
			setAlpha_inh_exc(alpha,i);
		}
	}
	
	public void setAlpha_inh_exc(double alpha, int module_number){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setAlpha_exc(alpha);
		}
	}
	
	public void setAlpha_inh_exc(double alpha, int module_number, int compartment){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setAlpha_exc(alpha,compartment);
		}
	}
	
	public void setAlpha_inh_inh(double alpha){
		for (int i=0; i<neurons.size(); i++){
			setAlpha_inh_inh(alpha,i);
		}
	}
	
	public void setAlpha_inh_inh(double alpha, int module_number){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setAlpha_inh(alpha);
		}
	}
	
	public void setAlpha_inh_inh(double alpha, int module_number, int compartment){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setAlpha_inh(alpha,compartment);
		}
	}
	
	// shut off all learning
	public void setAlpha(){
		setAlpha_exc_exc(0);
		setAlpha_exc_inh(0);
		setAlpha_inh_exc(0);
		setAlpha_inh_inh(0);
	}
	
	public void setDelta_exc_exc(double delta, int module_number, int compartment){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setDelta_exc(delta,compartment);
		}
	}
	
	public void setDelta_exc_exc(double delta, int module_number){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setDelta_exc(delta);
		}
	}
	
	public void setDelta_exc_exc(double delta){
		for (int i=0; i<neurons.size(); i++){
			setDelta_exc_exc(delta,i);
		}
	}
	
	public void setDelta_exc_inh(double delta, int module_number, int compartment){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setDelta_inh(delta,compartment);
		}
	}
	
	public void setDelta_exc_inh(double delta, int module_number){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setDelta_inh(delta);
		}
	}
	
	public void setDelta_exc_inh(double delta){
		for (int i=0; i<neurons.size(); i++){
			setDelta_exc_inh(delta,i);
		}
	}
	
	public void setDelta_inh_exc(double delta, int module_number, int compartment){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setDelta_exc(delta,compartment);
		}
	}
	
	public void setDelta_inh_exc(double delta, int module_number){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setDelta_exc(delta);
		}
	}
	
	public void setDelta_inh_exc(double delta){
		for (int i=0; i<neurons.size(); i++){
			setDelta_inh_exc(delta,i);
		}
	}
	
	public void setDelta_inh_inh(double delta, int module_number, int compartment){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setDelta_inh(delta,compartment);
		}
	}
	
	public void setDelta_inh_inh(double delta, int module_number){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setDelta_inh(delta);
		}
	}
	
	public void setDelta_inh_inh(double delta){
		for (int i=0; i<neurons.size(); i++){
			setDelta_inh_inh(delta,i);
		}
	}
	
	public void setLearning_exc_exc(double val, int module_number, int compartment, int type){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setLearning_exc(val,compartment,type);
		}
	}
	
	public void setLearning_inh_exc(double val, int module_number, int compartment, int type){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setLearning_exc(val,compartment,type);
		}
	}
	
	public void setLearning_exc_inh(double val, int module_number, int compartment, int type){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setLearning_inh(val,compartment,type);
		}
	}
	
	public void setLearning_inh_inh(double val, int module_number, int compartment, int type){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setLearning_inh(val,compartment,type);
		}
	}
	
	// shut off all learning
	public void setDelta(){
		setDelta_exc_exc(0);
		setDelta_exc_inh(0);
		setDelta_inh_exc(0);
		setDelta_inh_inh(0);
	}
	
	public void setAverage_window(int window){
		this.avg_window = window;
	}
}