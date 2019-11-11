package flib.neurons;

import java.util.ArrayList;
//import java.lang.FastMath;
import org.apache.commons.math3.util.FastMath;
import flib.math.VectorFun;
import flib.datastructures.TypeConversion;

public class SingleThresholdUnit implements 
java.io.Serializable {
	// synaptic weighting of incoming excitatory signals
	private double[][] weights_exc;
	// synaptic weighting of incoming inhibitory signals
	private double[][] weights_inh;
	// excitatory input,  past input, long term input average
	private double[][] input_exc, input_exc_p, input_exc_a;
	// inhibitory input,  past input, long term input average
	private double[][] input_inh, input_inh_p, input_inh_a;
	// excitatory firing average
	private double[] avg_exc;
	// inhibitory firing average
	private double[] avg_inh;
	// excitatory firing average
	private double[] var_exc;
	// inhibitory firing average
	private double[] var_inh;
	// exciatory input inverse
	private double[] inv_exc;
	// inhibitory  input inverse
	private double[] inv_inh;
	// learning rates
	private double[][] learning_param_exc, learning_param_inh;
	// amplification rate of the input signal
	private double[] amp_exc, amp_inh;
	// update rate. Related to conduction delay... maybe
	private double update;
	// neuron type. Still need to have a specific listing for all types
	private boolean type;
	// modules which are targeted by this neuron
	private int[] target_modules;
	// which synapse of the target corresponds
	private int[][][] syn_loc;
	// synapse origin
	private int[][][] syn_orig_exc, syn_orig_inh;
	// threshold value. Values below zero very likely will lead to disfunctional network behavior
	// zero is likely to be a good value
	private double threshold;
	// maximum firing rate
	private double max_rate;
	// memory length for neuron firing
	private double fact;
	// current firing rate
	private double rate, current_rate, past_rate;
	// temporal avering for the excitatory and inhibitory rate output
	private double tc;
	// number of compartments
	private int num_comp_exc, num_comp_inh;
	
	
	public SingleThresholdUnit(final double[][] weights_exc, final double[][] weights_inh, final double[] amp_exc, final double[]amp_inh, double[][] learning_param_exc, double[][] learning_param_inh, double update, boolean type, final int[] target_modules, final int[][][] syn_loc, final int[][][] syn_orig_exc, final int[][][] syn_orig_inh, double threshold, double max_rate, double fact, double tc){
		// copy all inputs with need to be stored as private variables
		this.weights_exc = (double[][])TypeConversion.copyMultiArrayObject(weights_exc);
		this.weights_inh = (double[][])TypeConversion.copyMultiArrayObject(weights_inh);
		this.input_exc = new double[weights_exc.length][];
		this.input_exc_p = new double[weights_exc.length][];
		this.input_exc_a = new double[weights_exc.length][];
		this.avg_exc = new double[weights_exc.length];
		this.var_exc = new double[weights_exc.length];
		this.inv_exc = new double[weights_exc.length];
		for (int i=0; i<weights_exc.length; i++){
			this.input_exc[i] = new double[weights_exc[i].length];
			this.input_exc_p[i] = new double[weights_exc[i].length];
			this.avg_exc[i] = learning_param_exc[i][5];
			//this.var_exc[i] = learning_param_exc[i][5]*learning_param_exc[i][5]+learning_param_exc[i][8]*learning_param_exc[i][8];
			this.inv_exc[i] = 1./(learning_param_exc[i][6]+1);
			this.input_exc_a[i] = VectorFun.add(new double[weights_exc[i].length],learning_param_exc[i][5]);
		}
		this.input_inh = new double[weights_inh.length][];
		this.input_inh_p = new double[weights_inh.length][];
		this.avg_inh = new double[weights_inh.length];
		this.var_inh = new double[weights_exc.length];
		this.inv_inh = new double[weights_exc.length];
		this.input_inh_a = new double[weights_inh.length][];
		for (int i=0; i<weights_inh.length; i++){
			this.input_inh[i] = new double[weights_inh[i].length];
			this.input_inh_p[i] = new double[weights_inh[i].length];
			this.avg_inh[i] = learning_param_inh[i][5];
			//this.var_inh[i] = learning_param_inh[i][5]*learning_param_inh[i][5]+learning_param_inh[i][8]*learning_param_inh[i][8];
			this.inv_inh[i] = 1./(learning_param_inh[i][6]+1);
			this.input_inh_a[i] = VectorFun.add(new double[weights_inh[i].length],learning_param_inh[i][5]);
		}
		this.amp_exc = amp_exc.clone();
		this.amp_inh = amp_inh.clone();
		this.num_comp_exc = weights_exc.length;
		this.num_comp_inh = weights_inh.length;
		this.learning_param_exc = new double[num_comp_exc][learning_param_exc[0].length];
		this.learning_param_inh = new double[num_comp_inh][learning_param_inh[0].length];
		for (int i=0; i<num_comp_exc; i++){
			this.learning_param_exc[i] = learning_param_exc[i].clone();
			if (this.learning_param_exc[i][3]>0){
				this.learning_param_exc[i][3] = 1./(this.learning_param_exc[i][3]*this.learning_param_exc[i][3]);
			}
			/*
			if (this.learning_param_exc[i][7]>0){
				this.learning_param_exc[i][7] = 1./(this.learning_param_exc[i][7]*this.learning_param_exc[i][7]);
			}
			*/
			if (this.learning_param_exc[i][8]>0){
				this.learning_param_exc[i][8] = 1./(this.learning_param_exc[i][8]*this.learning_param_exc[i][8]);
			}
			if (this.learning_param_exc[i][10]>0){
				this.learning_param_exc[i][10] = 1./(this.learning_param_exc[i][10]);
			}
		}
		for (int i=0; i<num_comp_inh; i++){
			this.learning_param_inh[i] = learning_param_inh[i].clone();
			if (this.learning_param_inh[i][3]>0){
				this.learning_param_inh[i][3] = 1./(this.learning_param_inh[i][3]*this.learning_param_inh[i][3]);
			}
			/*
			if (this.learning_param_inh[i][7]>0){
				this.learning_param_inh[i][7] = 1./(this.learning_param_inh[i][7]*this.learning_param_inh[i][7]);
			}
			*/
			if (this.learning_param_inh[i][8]>0){
				this.learning_param_inh[i][8] = 1./(this.learning_param_inh[i][8]*this.learning_param_inh[i][8]);
			}
			if (this.learning_param_inh[i][10]>0){
				this.learning_param_inh[i][10] = 1./(this.learning_param_inh[i][10]);
			}
		}
		this.update = update;
		this.type = type;
		this.target_modules = target_modules.clone();
		this.syn_loc = (int[][][])TypeConversion.copyMultiArrayObject(syn_loc);
		this.syn_orig_exc = (int[][][])TypeConversion.copyMultiArrayObject(syn_orig_exc);
		this.syn_orig_inh = (int[][][])TypeConversion.copyMultiArrayObject(syn_orig_inh);
		this.threshold = threshold;
		this.max_rate = max_rate;
		this.fact = fact;
		this.rate = 0;
		this.current_rate = 0;
		this.past_rate = 0;
		this.tc = tc;
	}
	
	
	// this function is called by the units that form synapses onto this unit once per firing update and per synapse
	public void updateInput(boolean type, double frequency, int compartment, int syn_num){
		if (type){
			this.input_exc_p[compartment][syn_num] = this.input_exc[compartment][syn_num];
			this.input_exc[compartment][syn_num] = frequency;
			this.input_exc_a[compartment][syn_num] = (this.learning_param_exc[compartment][6]*this.input_exc_a[compartment][syn_num]+frequency)*inv_exc[compartment];
		}
		else {
			this.input_inh_p[compartment][syn_num] = this.input_inh[compartment][syn_num];
			this.input_inh[compartment][syn_num] = frequency;
			this.input_inh_a[compartment][syn_num] = (this.learning_param_inh[compartment][6]*this.input_inh_a[compartment][syn_num]+frequency)*inv_inh[compartment];
		}
	}
	
	/* old version
	public void updateWeightsFireWire(){
		double e = this.rate;//*(this.rate-this.past_rate);
		double d = this.rate-this.past_rate;
		for (int i=0; i<num_comp_exc; i++){
			double wt = 0;
			double b = (this.learning_param_exc[i][5]-this.avg_exc[i]);
			//double d = (learning_param_exc[i][7]-this.rate);
			double n = this.weights_exc[i].length;
			for (int j=0; j<this.weights_exc[i].length; j++){
				//double a = d*e*(this.input_exc[i][j]-this.input_exc_p[i][j]);
				double g = FastMath.pow(this.weights_exc[i][j]*n+1,-this.learning_param_exc[i][9]);
				double f = this.input_exc[i][j]-this.input_exc_p[i][j];
				//double a = FastMath.signum(d*f)*e*(this.learning_param_exc[i][7]-this.rate)*this.input_exc[i][j];
				double a = e*this.input_exc[i][j];
				double c = (this.learning_param_exc[i][5]-this.input_exc_a[i][j]);
				//this.weights_exc[i][j]+=this.learning_param_exc[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_exc[i][3]+(d*d+f*f)*learning_param_exc[i][8])*0.5)*FastMath.pow(FastMath.abs(a),learning_param_exc[i][1])*FastMath.signum(a);
				this.weights_exc[i][j]+=g*(FastMath.signum(d*f*b)+this.learning_param_exc[i][7])*this.learning_param_exc[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_exc[i][3]+(d*d+f*f)*learning_param_exc[i][8])*0.5)*FastMath.pow(a,learning_param_exc[i][1]);
				if (this.weights_exc[i][j]<0){
					this.weights_exc[i][j] = 0;
				}
				wt+= this.weights_exc[i][j];
			}
			wt = 1/wt;
			for (int j=0; j<this.weights_exc[i].length; j++){
				this.weights_exc[i][j]*=wt;
			}
			this.amp_exc[i]+=this.learning_param_exc[i][4]*FastMath.pow(FastMath.abs(b),learning_param_exc[i][2])*FastMath.signum(b);
			if (this.amp_exc[i]<0){
				this.amp_exc[i] = 0;
			}
		}
		for (int i=0; i<num_comp_inh; i++){
			double wt = 0;
			double b = (this.learning_param_inh[i][5]-this.avg_inh[i]);
			//double d = (learning_param_inh[i][7]-this.rate);
			double n = this.weights_inh[i].length;
			for (int j=0; j<this.weights_inh[i].length; j++){
				//double a = -d*e*(this.input_inh[i][j]-this.input_inh_p[i][j]);
				//double a = e*FastMath.abs(this.learning_param_inh[i][7]-this.rate)*this.input_inh[i][j];//*FastMath.signum(this.input_inh_p[i][j]-this.input_inh[i][j]);
				double f = this.input_inh[i][j]-this.input_inh_p[i][j];
				//double a = FastMath.signum(d*f)*e*(this.learning_param_inh[i][7]-this.rate)*this.input_inh[i][j];
				double a = e*this.input_inh[i][j];
				double g = FastMath.pow(this.weights_inh[i][j]*n+1,-this.learning_param_inh[i][9]);
				double c = (this.learning_param_inh[i][5]-this.input_inh_a[i][j]);
				//this.weights_inh[i][j]+=this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5)*FastMath.pow(FastMath.abs(a),learning_param_inh[i][1])*FastMath.signum(a);
				this.weights_inh[i][j]+=g*(-FastMath.signum(d*f*b)+this.learning_param_inh[i][7])*this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5)*FastMath.pow(a,learning_param_inh[i][1]);
				if (this.weights_inh[i][j]<0){
					this.weights_inh[i][j] = 0;
				}
				wt+= this.weights_inh[i][j];
			}
			wt = 1/wt;
			for (int j=0; j<this.weights_inh[i].length; j++){
				this.weights_inh[i][j]*=wt;
			}
			this.amp_inh[i]+=-this.learning_param_inh[i][4]*FastMath.pow(FastMath.abs(b),learning_param_inh[i][2])*FastMath.signum(b);
			if (this.amp_inh[i]<0){
				this.amp_inh[i] = 0;
			}
		}
	}
	*/
	
	public void updateWeightsFireWire(){
		double e = this.rate;//*(this.rate-this.past_rate);
		double d = this.rate-this.past_rate;
		
		for (int i=0; i<num_comp_exc; i++){
			double wt = 0;
			double b = (this.learning_param_exc[i][5]-this.avg_exc[i]);
			double a = 0;
			double n = this.weights_exc[i].length;
			/*
			for (int j=0; j<this.weights_exc[i].length; j++){
				a+=this.input_exc[i][j];
			}
			// compartment in average
			//a/=this.weights_exc[i].length;
			//a*=this.learning_param_exc[i][1];
			*/
			// if this unit is above target average firing learning is done
			for (int j=0; j<this.weights_exc[i].length; j++){
				double f = this.input_exc[i][j]-this.input_exc_p[i][j];
				double g = FastMath.pow(this.weights_exc[i][j]*n+1,-this.learning_param_exc[i][9]);
				double c = (this.learning_param_exc[i][5]-this.input_exc_a[i][j]);
				//double h = e-this.input_exc[i][j];
				//this.weights_exc[i][j]+=this.learning_param_exc[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_exc[i][3]+(d*d+f*f)*learning_param_exc[i][8])*0.5)*FastMath.pow(FastMath.abs(a),learning_param_exc[i][1])*FastMath.signum(a);
				if (d*f>0){
					this.weights_exc[i][j]+=g*this.learning_param_exc[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_exc[i][3]+(d*d+f*f)*learning_param_exc[i][8])*0.5);
				}
				/*
				else if (d*f<0){
					this.weights_exc[i][j]-=this.learning_param_exc[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_exc[i][3]+(d*d+f*f)*learning_param_exc[i][8])*0.5);
				}
				*/
				if (this.weights_exc[i][j]<0){
					this.weights_exc[i][j] = 0;
				}
				wt+= this.weights_exc[i][j];
			}
			wt = 1/wt;
			for (int j=0; j<this.weights_exc[i].length; j++){
				this.weights_exc[i][j]*=wt;
			}
			this.amp_exc[i]+=this.learning_param_exc[i][4]*FastMath.pow(FastMath.abs(b),learning_param_exc[i][2])*FastMath.signum(b);
			if (this.amp_exc[i]<0){
				this.amp_exc[i] = 0;
			}
		}
		for (int i=0; i<num_comp_inh; i++){
			double wt = 0;
			double b = (this.learning_param_inh[i][5]-this.avg_inh[i]);
			double a = 0;
			double n = this.weights_inh[i].length;
			if (this.avg_inh[i]>this.learning_param_inh[i][7]*this.learning_param_inh[i][5]){
				for (int j=0; j<this.weights_inh[i].length; j++){
					double f = this.input_inh[i][j]-this.input_inh_p[i][j];
					if (d*f>0){
						double g = FastMath.pow(this.weights_inh[i][j]*n+1,-this.learning_param_inh[i][9]);
						double c = (this.learning_param_inh[i][5]-this.input_inh_a[i][j]);
						//double a2 = Math.abs(d*d+f*f-2*this.learning_param_inh[i][7]*this.learning_param_inh[i][7])*learning_param_inh[i][8];
						//this.weights_inh[i][j]+=this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5)*FastMath.pow(FastMath.abs(a),learning_param_inh[i][1])*FastMath.signum(a);
						this.weights_inh[i][j]+=g*this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5);
						//this.weights_inh[i][j]+=-Math.signum(d*f)*g*this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+a2)*0.5);
						if (this.weights_inh[i][j]<0){
							this.weights_inh[i][j] = 0;
						}
					}
					
					if (d*f<0){
						double g = FastMath.pow(this.weights_inh[i][j]*n+1,-this.learning_param_inh[i][9]);
						double c = (this.learning_param_inh[i][5]-this.input_inh_a[i][j]);
						//double a2 = Math.abs(d*d+f*f-2*this.learning_param_inh[i][7]*this.learning_param_inh[i][7])*learning_param_inh[i][8];
						//this.weights_inh[i][j]+=this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5)*FastMath.pow(FastMath.abs(a),learning_param_inh[i][1])*FastMath.signum(a);
						this.weights_inh[i][j]+=-this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5);
						//this.weights_inh[i][j]+=-Math.signum(d*f)*g*this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+a2)*0.5);
						if (this.weights_inh[i][j]<0){
							this.weights_inh[i][j] = 0;
						}
					}
					
					wt+= this.weights_inh[i][j];
				}
				wt = 1/wt;
				for (int j=0; j<this.weights_inh[i].length; j++){
					this.weights_inh[i][j]*=wt;
				}
			}
			if (this.avg_inh[i]<1./this.learning_param_inh[i][7]*this.learning_param_inh[i][5]){
				for (int j=0; j<this.weights_inh[i].length; j++){
					double f = this.input_inh[i][j]-this.input_inh_p[i][j];
					if (d*f<0){
						double g = FastMath.pow(this.weights_inh[i][j]*n+1,-this.learning_param_inh[i][9]);
						double c = (this.learning_param_inh[i][5]-this.input_inh_a[i][j]);
						//double a2 = Math.abs(d*d+f*f-2*this.learning_param_inh[i][7]*this.learning_param_inh[i][7])*learning_param_inh[i][8];
						//this.weights_inh[i][j]+=this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5)*FastMath.pow(FastMath.abs(a),learning_param_inh[i][1])*FastMath.signum(a);
						this.weights_inh[i][j]+=g*this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5);
						//this.weights_inh[i][j]+=-Math.signum(d*f)*g*this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+a2)*0.5);
						if (this.weights_inh[i][j]<0){
							this.weights_inh[i][j] = 0;
						}
					}
					
					if (d*f>0){
						double g = FastMath.pow(this.weights_inh[i][j]*n+1,-this.learning_param_inh[i][9]);
						double c = (this.learning_param_inh[i][5]-this.input_inh_a[i][j]);
						//double a2 = Math.abs(d*d+f*f-2*this.learning_param_inh[i][7]*this.learning_param_inh[i][7])*learning_param_inh[i][8];
						//this.weights_inh[i][j]+=this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5)*FastMath.pow(FastMath.abs(a),learning_param_inh[i][1])*FastMath.signum(a);
						this.weights_inh[i][j]+=-this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+(d*d+f*f)*learning_param_inh[i][8])*0.5);
						//this.weights_inh[i][j]+=-Math.signum(d*f)*g*this.learning_param_inh[i][0]*FastMath.exp(-((c*c+b*b)*learning_param_inh[i][3]+a2)*0.5);
						if (this.weights_inh[i][j]<0){
							this.weights_inh[i][j] = 0;
						}
					}
					
					wt+= this.weights_inh[i][j];
				}
				wt = 1/wt;
				for (int j=0; j<this.weights_inh[i].length; j++){
					this.weights_inh[i][j]*=wt;
				}
			}
			this.amp_inh[i]+=-this.learning_param_inh[i][4]*FastMath.pow(FastMath.abs(b),learning_param_inh[i][2])*FastMath.signum(b);
			if (this.amp_inh[i]<0){
				this.amp_inh[i] = 0;
			}
		}
	}
	
	// the projection function when a neuron fires
	// it updates the input state on its appropriate synapse in the target neuron
	public double project(final ArrayList<ArrayList<SingleThresholdUnit>> list, double freq){
		int a;
		double b;
		b = freq;
		// go through all modules which the current neuron targets
		for (int i=0; i<this.target_modules.length; i++){
			// store the target module number
			a = this.target_modules[i];
			// go through all synapses which this neuron forms on targets in the target module
			for (int j=0; j<this.syn_loc[i].length; j++){
				// check the target neuron's type to determine to decide the amplication factor for this
				// connection type
				list.get(a).get(this.syn_loc[i][j][0]).updateInput(this.type,b,this.syn_loc[i][j][1],this.syn_loc[i][j][2]);
			}
		}
		return freq;
	}
	
	// this is the more commonly used case, where the current frequency to project is computed first from the 
	// neuron's current input and weights
	public double project(final ArrayList<ArrayList<SingleThresholdUnit>> list){
		double freq = this.getRate();
		return project(list,freq);
	}
	
	// setter functions
	public void setAlpha_exc(double alpha_exc){
		for (int i=0; i<this.num_comp_exc; i++){
			this.learning_param_exc[i][0] = alpha_exc;
		}
	}
	
	public void setAlpha_inh(double alpha_inh){
		for (int i=0; i<this.num_comp_inh; i++){
			this.learning_param_inh[i][0] = alpha_inh;
		}
	}
	
	public void setAlpha_exc(double[] alpha_exc){
		for (int i=0; i<this.num_comp_exc; i++){
			this.learning_param_exc[i][0] = alpha_exc[i];
		}
	}
	
	public void setAlpha_inh(double[] alpha_inh){
		for (int i=0; i<this.num_comp_inh; i++){
			this.learning_param_inh[i][0] = alpha_inh[i];
		}
	}
	
	public void setAlpha_exc(double alpha_exc, int num){
		this.learning_param_exc[num][0] = alpha_exc;
	}
	
	public void setAlpha_inh(double alpha_inh, int num){
		this.learning_param_inh[num][0] = alpha_inh;
	}
	
	public void setDelta_exc(double delta_exc){
		for (int i=0; i<this.num_comp_exc; i++){
			this.learning_param_exc[i][4] = delta_exc;
		}
	}
	
	public void setDelta_inh(double delta_inh){
		for (int i=0; i<this.num_comp_inh; i++){
			this.learning_param_inh[i][4] = delta_inh;
		}
	}
	
	public void setDelta_exc(double[] delta_exc){
		for (int i=0; i<this.num_comp_exc; i++){
			this.learning_param_exc[i][4] = delta_exc[i];
		}
	}
	
	public void setDelta_inh(double[] delta_inh){
		for (int i=0; i<this.num_comp_inh; i++){
			this.learning_param_inh[i][4] = delta_inh[i];
		}
	}
	
	public void setDelta_exc(double delta_exc, int num){
		this.learning_param_exc[num][4] = delta_exc;
	}
	
	public void setDelta_inh(double delta_inh, int num){
		this.learning_param_inh[num][4] = delta_inh;
	}
	
	public void setLearning_exc(double val, int num, int type){
		this.learning_param_exc[num][type] = val;
	}
	
	public void setLearning_inh(double val, int num, int type){
		this.learning_param_inh[num][type] = val;
	}
	
	public void setRate(double rate){
		this.rate = rate;
	}
	
	public void setAmp_exc(double amp_exc){
		for (int i=0; i<num_comp_exc; i++){
			this.amp_exc[i] = amp_exc;
		}
	}
	
	public void setAmp_inh(double amp_inh){
		for (int i=0; i<num_comp_inh; i++){
			this.amp_inh[i] = amp_inh;
		}
	}
	
	public void setAmp_exc(double amp_exc, int num){
		this.amp_exc[num] = amp_exc;
	}
	
	public void setAmp_inh(double amp_inh, int num){
		this.amp_inh[num] = amp_inh;
	}
	
	// getter functions
	// getFrequency also sets a current rate
	public double getFrequency(){
		double f = 0;
		//double ar = 1;
		for (int i=0; i<this.weights_exc.length; i++){
			double a = 1./this.learning_param_exc[i][10];
			//double a = FastMath.pow(this.rate*this.learning_param_exc[i][10],this.learning_param_exc[i][11]);
			for (int j=0; j<this.weights_exc[i].length; j++){
				f+=this.weights_exc[i][j]*FastMath.pow(this.input_exc[i][j]*this.learning_param_exc[i][10],this.learning_param_exc[i][11])*a*amp_exc[i];
				//f+=this.weights_exc[i][j]*this.input_exc[i][j]*amp_exc[i];
			}
		}
		for (int i=0; i<this.weights_inh.length; i++){
			//double a = FastMath.pow(this.rate*this.learning_param_inh[i][10],this.learning_param_inh[i][11]);
			double a = 1./this.learning_param_inh[i][10];
			for (int j=0; j<this.weights_inh[i].length; j++){
				f-=this.weights_inh[i][j]*FastMath.pow(this.input_inh[i][j]*this.learning_param_inh[i][10],this.learning_param_inh[i][11])*a*amp_inh[i];
				//f-=this.weights_inh[i][j]*this.input_inh[i][j]*amp_inh[i];
			}
		}
		f-=threshold;
		this.current_rate = f;
		f = (this.tc*this.rate+f)/(this.tc+1);
		if (f<0){
			f = 0;
		}
		else if(f>max_rate){
			f = max_rate;
		}
		this.past_rate = this.rate;
		this.rate = f;
		for (int i=0; i<this.avg_exc.length; i++){
			avg_exc[i] = (avg_exc[i]*this.learning_param_exc[i][6]+f)/(this.learning_param_exc[i][6]+1);
			//var_exc[i] = (var_exc[i]*this.learning_param_exc[i][6]+f*f)/(this.learning_param_exc[i][6]+1);
		}
		for (int i=0; i<this.avg_inh.length; i++){
			avg_inh[i] = (avg_inh[i]*this.learning_param_inh[i][6]+f)/(this.learning_param_inh[i][6]+1);
			//var_inh[i] = (var_inh[i]*this.learning_param_inh[i][6]+f*f)/(this.learning_param_inh[i][6]+1);
		}
		return f;
	}
	
	public double getRate(){
		return rate;
	}
	
	public double[] getAvg_exc(){
		return this.avg_exc;
	}
	
	public double[] getAvg_inh(){
		return this.avg_inh;
	}
	
	public double[] getAmp_exc(){
		return this.amp_exc.clone();
	}
	
	public double[] getAmp_inh(){
		return this.amp_inh.clone();
	}
	
	public double[][] getWeights_exc(){
		return (double[][])TypeConversion.copyMultiArrayObject(weights_exc);
	}
	
	public double[][] getWeights_inh(){
		return (double[][])TypeConversion.copyMultiArrayObject(weights_inh);
	}
	
	public int[][][] getSynapseLocations(){
		return (int[][][])TypeConversion.copyMultiArrayObject(syn_loc);
	}
	
	public int[][][] getSynapseOrigins_exc(){
		return (int[][][])TypeConversion.copyMultiArrayObject(syn_orig_exc);
	}
	
	public int[][][] getSynapseOrigins_inh(){
		return (int[][][])TypeConversion.copyMultiArrayObject(syn_orig_inh);
	}
	
	public boolean getType(){
		return this.type;
	}
	
	public double getUpdate(){
		return this.update;
	}
	
	public double[][] getInput_exc(){
		return (double[][])TypeConversion.copyMultiArrayObject(input_exc);
	}
	
	public double[][] getInput_inh(){
		return (double[][])TypeConversion.copyMultiArrayObject(input_inh);
	}
	
	public int getCompartmentNumber_exc(){
		return this.num_comp_exc;
	}
	
	public int getCompartmentNumber_inh(){
		return this.num_comp_inh;
	}
}
	