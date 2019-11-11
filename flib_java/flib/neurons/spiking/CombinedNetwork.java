package flib.neurons.spiking;

import java.util.ArrayList;
import java.util.TreeSet;
import java.util.Iterator;
import java.util.Arrays;
import java.util.Random;
import ij.ImagePlus;
import ij.IJ;
import flib.math.distributions.LogNormal;
import flib.neurons.spiking.SynapticEvent;
import flib.neurons.spiking.SingleThresholdUnit;
import flib.neurons.spiking.NetworkModule;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.ij.stack.StackOperations;

public class CombinedNetwork implements 
java.io.Serializable {
	
	// we have all types of private variables to store for the sake of later transparency
	// list of all neurons in all modules
	private ArrayList<ArrayList<SingleThresholdUnit>> neurons;
	// list of all modules in the network
	private ArrayList<NetworkModule> modules;
	// list of all queued event
	private TreeSet<SynapticEvent> eventlist;
	// global time of the simulation
	private double time;
	// Module neuron variables
	private double[][] neuron_param;
	// inter-module synapse connectivity values
	private double[][][] syn_connectivity;
	// inter-module geometric connectivity
	private double[][][] geo_connectivity;
	// module size parameters
	private int[][] module_size;
	// target module numbers of each module
	private int[][] module_targets;
	// lookup lists
	private int[] neuron_module;
	private int[] neuron_loc;
	private Random rng;
	
	public CombinedNetwork(final int[][] module_size, final int[][] module_targets, final double[][] neuron_param, final double[][][] syn_connectivity, final double[][][] geo_connectivity){
		// module_size: contains the dimensions of each module
		// module_size[i][0]: module width
		// module_size[i][1]: module height
		// module_size[i][2]: module number of excitatory layers
		// module_size[i][3]: module number of inhibitory layers
		
		// number of modules
		int m = module_size.length;
		// copy the module parameters
		this.module_size = new int[m][4];
		this.module_targets = new int[m][];
		// Initialize all Modules
		this.modules = new ArrayList<NetworkModule>();
		for (int i=0; i<m; i++){
			this.module_size[i] = module_size[i].clone();
			this.module_targets[i] = module_targets[i].clone();
			this.modules.add(new NetworkModule(this.module_size[i][0],this.module_size[i][1],this.module_size[i][2],this.module_size[i][3], this.module_targets[i]));
		}
		
		// copy all other inputs
		this.neuron_param = new double[m][];
		this.syn_connectivity = new double[m][][];
		this.geo_connectivity = new double[m][][];
		for (int i=0; i<m; i++){
			this.neuron_param[i] = neuron_param[i].clone();
			this.syn_connectivity[i] = new double[this.module_targets[i].length][];
			this.geo_connectivity[i] = new double[this.module_targets[i].length][];
			for (int j=0; j<this.module_targets[i].length; j++){
				this.syn_connectivity[i][j] = syn_connectivity[i][j].clone();
				this.geo_connectivity[i][j] = geo_connectivity[i][j].clone();
			}
		}
		
		// preparation of variables for the module connecting
		int[][] exc_weight_counter = new int[m][];
		int[][] inh_weight_counter = new int[m][];
		ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> targets = new ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>();
		ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> syn_loc = new ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>();
		ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> delay = new ArrayList<ArrayList<ArrayList<ArrayList<Double>>>>();
		ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> syn_orig_exc = new ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>();
		ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>> syn_orig_inh = new ArrayList<ArrayList<ArrayList<ArrayList<Integer>>>>();
		ArrayList<ArrayList<ArrayList<Double>>> fixed_weights = new ArrayList<ArrayList<ArrayList<Double>>>();
		for (int i=0; i<m; i++){
			int num_neuron = this.module_size[i][0]*this.module_size[i][1]*(this.module_size[i][2]+this.module_size[i][3]);
			exc_weight_counter[i] = new int[num_neuron];
			inh_weight_counter[i] = new int[num_neuron];
			targets.add(new ArrayList<ArrayList<ArrayList<Integer>>>());
			syn_loc.add(new ArrayList<ArrayList<ArrayList<Integer>>>());
			delay.add(new ArrayList<ArrayList<ArrayList<Double>>>());
			syn_orig_exc.add(new ArrayList<ArrayList<ArrayList<Integer>>>());
			syn_orig_inh.add(new ArrayList<ArrayList<ArrayList<Integer>>>());
			fixed_weights.add(new ArrayList<ArrayList<Double>>());
			for (int j=0; j<num_neuron; j++){
				targets.get(i).add(new ArrayList<ArrayList<Integer>>());
				syn_loc.get(i).add(new ArrayList<ArrayList<Integer>>());
				delay.get(i).add(new ArrayList<ArrayList<Double>>());
				syn_orig_exc.get(i).add(new ArrayList<ArrayList<Integer>>());
				syn_orig_inh.get(i).add(new ArrayList<ArrayList<Integer>>());
				fixed_weights.get(i).add(new ArrayList<Double>());
				for (int k=0; k<this.module_targets[i].length; k++){
					targets.get(i).get(j).add(new ArrayList<Integer>());
					syn_loc.get(i).get(j).add(new ArrayList<Integer>());
					delay.get(i).get(j).add(new ArrayList<Double>());
				}
			}
		}
		
		// now we explicitly create the synaptic targets for all the neurons in all different modules
		for (int i=0; i<m; i++){
			for (int j=0; j<this.module_targets[i].length; j++){
				this.modules.get(i).connectToModule(i,j,this.modules.get(this.module_targets[i][j]),this.syn_connectivity[i][j],this.geo_connectivity[i][j],exc_weight_counter[this.module_targets[i][j]],inh_weight_counter[this.module_targets[i][j]],targets.get(i),syn_loc.get(i),delay.get(i),syn_orig_exc.get(this.module_targets[i][j]),syn_orig_inh.get(this.module_targets[i][j]),fixed_weights.get(this.module_targets[i][j]));
			}
		}
		
		// based on above connectivity we generate the neurons in all modules
		// add the module neuron lists to the global neuron list
		this.neurons = new ArrayList<ArrayList<SingleThresholdUnit>>();
		for (int i=0; i<m; i++){
			this.modules.get(i).generateNeurons(this.neuron_param[i],0,exc_weight_counter[i],inh_weight_counter[i],targets.get(i),syn_loc.get(i),delay.get(i),syn_orig_exc.get(i),syn_orig_inh.get(i),fixed_weights.get(i));
			this.neurons.add(this.modules.get(i).getNeurons());
		}
		
		// initialize the event list
		this.eventlist = new TreeSet<SynapticEvent>();
		this.rng = new Random();
	}
	
	public void update_input(int module_number, final double[] input_rates){
		// remove all previous forced spikings from the queue
		Iterator<SynapticEvent> itr = eventlist.iterator();
		//IJ.log("Number of events before update: "+String.format("%d",eventlist.size()));
		int a = 0;
		int b = 0;
		while(itr.hasNext()){
			SynapticEvent current = new SynapticEvent(itr.next());
			if ((current.type==3)&&(current.target_module==module_number)){
				itr.remove();
			}
			/*else if (current.type==0||current.type==1){
				a++;
			}
			else if (current.type==2){
				b++;
			}*/
		}
		/*IJ.log("Number of events after event removal: "+String.format("%d",eventlist.size()));
		IJ.log("Current time: "+String.format("%.2f",time));
		if (!eventlist.isEmpty()){
			IJ.log("Last time: "+String.format("%.2f",eventlist.last().time));
		}*/
		// for each neuron in this module create an event which causes repeated
		// firing events at input rate intervals
		for (int i=0; i<neurons.get(module_number).size(); i++){
			if(!eventlist.add(new SynapticEvent(time-Math.log(1-rng.nextDouble())/(input_rates[i]+1e-6),rng.nextDouble(),3,module_number,i,1/(input_rates[i]+1e-6)))){
				IJ.log("an update event was lost...");
			}
		}
		//IJ.log("Number of synaptic events in the queue: "+String.format("%d",a));
		//IJ.log("Number of refractory events in the queue: "+String.format("%d",b));
		//IJ.log("Time frame of the coming events: "+String.format("%.2f",eventlist.last().time-time));
	}
		
		
	
	public void iterate(double duration){
		double target_time = time+duration;
		while(!eventlist.isEmpty()){
			SynapticEvent current = new SynapticEvent(eventlist.first());
			if (current.time<time){
				IJ.log("event timing is off...");
			}
			if(current.time>=target_time){
				break;
			}
			time = current.time;
			switch (current.type) {
				// excitatory firing
				case 0: if(neurons.get(current.target_module).get(current.target_neuron).synapticInput(true,(int)(current.target_synapse),time)){
						neurons.get(current.target_module).get(current.target_neuron).project(time,eventlist,current.target_module,current.target_neuron);
					}
					//IJ.log("firing event 0");
					break;
				// inhibitory firing
				case 1: neurons.get(current.target_module).get(current.target_neuron).synapticInput(false,(int)(current.target_synapse),time);
					//IJ.log("firing event 1");
					break;
				// refractory period
				case 2: neurons.get(current.target_module).get(current.target_neuron).refractoryPeriod(time);
					//IJ.log("firing event 2");
					break;
				// forced input firing... hope these events don't get lost!
				case 3: neurons.get(current.target_module).get(current.target_neuron).project(time,eventlist,current.target_module,current.target_neuron);
					if(!eventlist.add(new SynapticEvent(time-Math.log(1-rng.nextDouble())*current.target_synapse,rng.nextDouble(),3,current.target_module,current.target_neuron,current.target_synapse))){
						IJ.log("an update event was lost...");
					}
					//IJ.log("firing event 3");
					break;
				default: break;
			}
			//IJ.log("at time"+String.format("%f",time));
			// remove the event from the list
			eventlist.remove(current);
		}
	}
	
	public ArrayList<ArrayList<ImagePlus>> run(final ArrayList<ImagePlus> imp, int numit, final int[] update_modules, final int[] update_rate, double duration, int output, int[] current_im){
		// imp: is an ArrayList which contains the input frames for the driver modules
		// numit: total number of iterations for the run
		// update_modules: which modules are to be driven with external input
		// update_rate: rate at which the individual driver modules are updated
		// duration: minimum amount of time that needs to pass by for an iteration count
		// output: how frequently all the modules should be written to output
		// current_im: frame number at which the individual modules should start receiving input from
		
		// prepare the output ArrayList
		ArrayList<ArrayList<ImagePlus>> out = new ArrayList<ArrayList<ImagePlus>>();
		for (int i=0; i<neurons.size(); i++){
			out.add(new ArrayList<ImagePlus>());
		}
		
		for (int i=0; i<numit; i++){
			// updating
			for (int j=0; j<update_modules.length; j++){
				if (i%update_rate[j]==0){
					current_im[j]%=imp.get(j).getNSlices();
					double[] x = VectorConv.float2double((float[])(imp.get(j).getImageStack().getProcessor(current_im[j]+1).convertToFloat().getPixels()));
					update_input(update_modules[j], x);
					current_im[j]++;
				}
			}
			// iterating through the current setup
			iterate(duration);
			if (i%output==0){
				for (int j=0; j<neurons.size(); j++){
					out.get(j).add(getNeuronStates(j));
				}
			}
		}
		return out;
	}
	
	public int[][][] weightHistogram(int numbin){
		int[][][] histo = new int[neurons.size()][4][numbin];
		double[] temp;
		for (int i=0; i<neurons.size(); i++){
			for (int j=0; j<neurons.get(i).size(); j++){
				// excitatory post synaptic neuron
				if (neurons.get(i).get(j).getType()){
					// excitatory in synapses
					temp = VectorFun.mult(neurons.get(i).get(j).getWeights_exc(),1);
					for (int k=0; k<temp.length; k++){
						if (temp[k]>=1){
							histo[i][0][numbin-1]++;
						}
						else{
							histo[i][0][(int)(temp[k]*numbin)]++;
						}
					}
					// inhibitory in synapses
					temp = VectorFun.mult(neurons.get(i).get(j).getWeights_inh(),1);
					for (int k=0; k<temp.length; k++){
						if (temp[k]>=1){
							histo[i][1][numbin-1]++;
						}
						else{
							histo[i][1][(int)(temp[k]*numbin)]++;
						}
					}
				}
				else {
					// excitatory in synapses
					temp = VectorFun.mult(neurons.get(i).get(j).getWeights_exc(),1);
					for (int k=0; k<temp.length; k++){
						if (temp[k]>=1){
							histo[i][2][numbin-1]++;
						}
						else{
							histo[i][2][(int)(temp[k]*numbin)]++;
						}
					}
					// inhibitory in synapses
					temp = VectorFun.mult(neurons.get(i).get(j).getWeights_inh(),1);
					for (int k=0; k<temp.length; k++){
						if (temp[k]>=1){
							histo[i][3][numbin-1]++;
						}
						else{
							histo[i][3][(int)(temp[k]*numbin)]++;
						}
					}
				}
			}
		}
		return histo;
	}
	
	// getter functions
	
	public ArrayList<ArrayList<SingleThresholdUnit>> getNeurons(){
		return this.neurons;
	}
	
	public TreeSet<SynapticEvent> getEventList(){
		return this.eventlist;
	}
	
	public ImagePlus getNeuronStates(int module_number){
		int w = this.module_size[module_number][0];
		int h = this.module_size[module_number][1];
		int d = this.module_size[module_number][2]+this.module_size[module_number][3];
		double[] temp = new double[w*h*d];
		for (int i=0; i<temp.length; i++){
			temp[i] = 1/(neurons.get(module_number).get(i).getRate()+1e-6);
		}
		return StackOperations.convert2Stack(temp,w,h,d);
	}
	
	public ImagePlus getAmp_exc(int module_number){
		int w = this.module_size[module_number][0];
		int h = this.module_size[module_number][1];
		int d = this.module_size[module_number][2]+this.module_size[module_number][3];
		double[] temp = new double[d*w*h];
		for (int i=0; i<w*h*d; i++){
			temp[i] = this.neurons.get(module_number).get(i).getAmp_exc();
		}
		return StackOperations.convert2Stack(temp,w,h,d);
	}
	
	public ImagePlus getAmp_inh(int module_number){
		int w = this.module_size[module_number][0];
		int h = this.module_size[module_number][1];
		int d = this.module_size[module_number][2]+this.module_size[module_number][3];
		double[] temp = new double[d*w*h];
		for (int i=0; i<w*h*d; i++){
			temp[i] = this.neurons.get(module_number).get(i).getAmp_inh();
		}
		return StackOperations.convert2Stack(temp,w,h,d);
	}
	
	// setter functions
	// learning rates
	public void setAlpha_exc_exc(double alpha, int module_number){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setAlpha_exc(alpha);
		}
	}
	
	public void setAlpha_exc_exc(double alpha){
		for (int i=0; i<neurons.size(); i++){
			setAlpha_exc_exc(alpha,i);
		}
	}
	
	public void setAlpha_exc_inh(double alpha, int module_number){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setAlpha_exc(alpha);
		}
	}
	
	public void setAlpha_exc_inh(double alpha){
		for (int i=0; i<neurons.size(); i++){
			setAlpha_exc_inh(alpha,i);
		}
	}
	
	public void setAlpha_inh_exc(double alpha, int module_number){
		int a = module_number;
		for (int i=0; i<this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i++){
			neurons.get(module_number).get(i).setAlpha_inh(alpha);
		}
	}
	
	public void setAlpha_inh_exc(double alpha){
		for (int i=0; i<neurons.size(); i++){
			setAlpha_inh_exc(alpha,i);
		}
	}
	
	public void setAlpha_inh_inh(double alpha, int module_number){
		int a = module_number;
		for (int i=this.module_size[a][0]*this.module_size[a][1]*this.module_size[a][2]; i<this.module_size[a][0]*this.module_size[a][1]*(this.module_size[a][2]+this.module_size[a][3]); i++){
			neurons.get(module_number).get(i).setAlpha_inh(alpha);
		}
	}
	
	public void setAlpha_inh_inh(double alpha){
		for (int i=0; i<neurons.size(); i++){
			setAlpha_inh_inh(alpha,i);
		}
	}
	
	// shut off all learning
	public void setAlpha(){
		setAlpha_exc_exc(0);
		setAlpha_exc_inh(0);
		setAlpha_inh_exc(0);
		setAlpha_inh_inh(0);
	}
	
	public void setAmp_fact_exc(){
		for (int i=0; i<this.neurons.size(); i++){
			for (int j=0; j<this.neurons.get(i).size(); j++){
				neurons.get(i).get(j).setAmp_fact_exc(0);
			}
		}
	}
	
	public void setAmp_fact_inh(){
		for (int i=0; i<this.neurons.size(); i++){
			for (int j=0; j<this.neurons.get(i).size(); j++){
				neurons.get(i).get(j).setAmp_fact_inh(0);
			}
		}
	}
	
	public void setSpikeFrame(double spike_frame){
		for (int i=0; i<this.neurons.size(); i++){
			for (int j=0; j<this.neurons.get(i).size(); j++){
				neurons.get(i).get(j).setSpikeFrame(spike_frame);
			}
		}
	}
	
	public void emptyEventlist(){
		this.eventlist = new TreeSet<SynapticEvent>();
	}
}