package flib.neurons;

import java.util.ArrayList;
import java.util.Random;
import java.util.Arrays;
import java.lang.Math;
import ij.ImagePlus;
import flib.neurons.SingleThresholdUnit;
import flib.algorithms.sampling.NeighborhoodSample;
import flib.math.VectorFun;
import flib.math.VectorConv;
import flib.math.random.Sample;
import flib.ij.stack.StackOperations;
import flib.external.indexedtreeset.IndexedTreeSet;

public class PatchObserver implements
java.io.Serializable {
	// dimensions of the input patch
	private int w, h, current_im;
	// collection of all input units
	private ArrayList<SingleThresholdUnit> neurons;
	private double[] update_prob;
	private double[] avg_rates;
	private double[] avg_variance;
	private double avg_window;
	
	public PatchObserver(final ImagePlus imp, int num_exc, int num_inh, double cutoff_exc, double cutoff_inh, double frac_dim, double syn_neuron_exc, double syn_neuron_inh, int type, double alpha_exc_exc, double alpha_exc_inh, double alpha_inh_exc, double alpha_inh_inh, double amp_exc_exc, double amp_exc_inh, double amp_inh_exc, double amp_inh_inh, double update_exc, double update_inh, double adapt, double max_rate, double avg_window){
		// all input fields should have this size
		this.w = imp.getWidth();
		this.h = imp.getHeight();
		this.current_im = 0;
		this.avg_window = avg_window;
		int[][] coord_exc = NeighborhoodSample.spiralCoord(1, cutoff_exc, 1, 1, frac_dim, 4);
		int[][] coord_inh = NeighborhoodSample.spiralCoord(1, cutoff_inh, 1, 1, frac_dim, 4);
		double p_exc = syn_neuron_exc/(coord_exc.length*(num_exc+num_inh));
		double p_exc_exc = p_exc;
		double p_exc_inh = p_exc;
		double p_inh = syn_neuron_inh/(coord_inh.length*num_inh*(num_exc+num_inh));
		double p_inh_exc = p_inh;
		double p_inh_inh = p_inh;
		int num_in = w*h;
		// neuron list initialization
		neurons = new ArrayList<SingleThresholdUnit>();
		int tot = num_in*(2+num_exc+num_inh); 
		// neurons which the other neurons receive synapses from
		int[] rec_exc = new int[tot];
		int[] rec_inh = new int[tot];
		ArrayList<ArrayList<Integer>> target = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> syn_loc = new ArrayList<ArrayList<Integer>>();
		Random rng = new Random();
		for (int i=0; i<tot; i++){
			target.add(new ArrayList<Integer>());
			syn_loc.add(new ArrayList<Integer>());
		}
		// preparation of the input synapses. These don't target inhibitory synapses
		for (int i=0; i<num_in*2; i++){
			int t = i%num_in;
			int[] temp = NeighborhoodSample.shapeNeighbor2d(coord_exc,w,h,t%w,t/w,0);
			for (int j=0; j<temp.length; j++){
				if (temp[j]!=-1){
					for (int k=0; k<num_exc; k++){
						if (rng.nextDouble()<p_exc_exc){
							int loc = num_in*(2+k)+temp[j];
							if (loc!=i){
								target.get(i).add(loc);
								syn_loc.get(i).add(rec_exc[loc]);
								rec_exc[loc]++;
							}
						}
					}
					/*
					for (int k=0; k<num_inh; k++){
						if (rng.nextDouble()<p_exc_inh){
							int loc = num_in*(2+num_exc+k)+temp[j];
							target.get(i).add(loc);
							syn_loc.get(i).add(rec_exc[loc]);
							rec_exc[loc]++;
						}
					}
					*/
				}
			}
		}
		// preparation of all excitatory synapses
		for (int i=2*num_in; i<num_in*(2+num_exc); i++){
			int t = i%num_in;
			int[] temp = NeighborhoodSample.shapeNeighbor2d(coord_exc,w,h,t%w,t/w,0);
			for (int j=0; j<temp.length; j++){
				if (temp[j]!=-1){
					for (int k=0; k<num_exc; k++){
						if (rng.nextDouble()<p_exc_exc){
							int loc = num_in*(2+k)+temp[j];
							if (loc!=i){
								target.get(i).add(loc);
								syn_loc.get(i).add(rec_exc[loc]);
								rec_exc[loc]++;
							}
						}
					}
					for (int k=0; k<num_inh; k++){
						if (rng.nextDouble()<p_exc_inh){
							int loc = num_in*(2+num_exc+k)+temp[j];
							target.get(i).add(loc);
							syn_loc.get(i).add(rec_exc[loc]);
							rec_exc[loc]++;
						}
					}
				}
			}
		}
		// preparation of all inhibitory synapses
		for (int i=num_in*(2+num_exc); i<num_in*(2+num_exc+num_inh); i++){
			int t = i%num_in;
			int[] temp = NeighborhoodSample.shapeNeighbor2d(coord_inh,w,h,t%w,t/w,0);
			for (int j=0; j<temp.length; j++){
				if (temp[j]!=-1){
					for (int k=0; k<num_exc; k++){
						if (rng.nextDouble()<p_inh_exc){
							int loc = num_in*(2+k)+temp[j];
							target.get(i).add(loc);
							syn_loc.get(i).add(rec_inh[loc]);
							rec_inh[loc]++;
						}
					}
					for (int k=0; k<num_inh; k++){
						if (rng.nextDouble()<p_inh_inh){
							int loc = num_in*(2+num_exc+k)+temp[j];
							if (loc!=i){
								target.get(i).add(loc);
								syn_loc.get(i).add(rec_inh[loc]);
								rec_inh[loc]++;
							}
						}
					}
				}
			}
		}
		
		this.update_prob = new double[tot];
		// preparation of the single threshold unit neurons
		// input units
		for (int i=0; i<2*num_in; i++){
			// prepare a weight vector
			double[] weights_exc = new double[]{1};
			double[] weights_inh = new double[]{1};
			int[] t = new int[target.get(i).size()];
			int[] s = new int[target.get(i).size()];
			for (int j=0; j<t.length; j++){
				t[j] = target.get(i).get(j);
				s[j] = syn_loc.get(i).get(j);
			}
			neurons.add(new SingleThresholdUnit(weights_exc, weights_inh, 0, 0, amp_exc_exc, 0, 0,adapt,true,t,s,0,max_rate));
			update_prob[i] = 0;
		}
		// excitatory units
		for (int i=2*num_in; i<num_in*(2+num_exc); i++){
			double[] weights_exc, weights_inh;
			if (type==1){
				weights_exc = VectorFun.add(new double[rec_exc[i]],1./rec_exc[i]);
				weights_inh = VectorFun.add(new double[rec_inh[i]],1./rec_inh[i]);
			}
			else {
				//weights_exc = Sample.randomUnitVector(rec_exc[i],rng);
				//weights_inh = Sample.randomUnitVector(rec_inh[i],rng);
				weights_exc = Sample.randomUnitVectorL1(rec_exc[i],rng);
				weights_inh = Sample.randomUnitVectorL1(rec_inh[i],rng);
			}
			int[] t = new int[target.get(i).size()];
			int[] s = new int[target.get(i).size()];
			for (int j=0; j<t.length; j++){
				t[j] = target.get(i).get(j);
				s[j] = syn_loc.get(i).get(j);
			}
			neurons.add(new SingleThresholdUnit(weights_exc, weights_inh, alpha_exc_exc, alpha_exc_inh, amp_exc_exc, amp_exc_inh, update_exc, adapt, true,t,s,0,max_rate));
			update_prob[i] = update_exc;
		}
		// inhibitory units
		for (int i=num_in*(2+num_exc); i<num_in*(2+num_exc+num_inh); i++){
			double[] weights_exc, weights_inh;
			if (type==1){
				weights_exc = VectorFun.add(new double[rec_exc[i]],1./rec_exc[i]);
				weights_inh = VectorFun.add(new double[rec_inh[i]],1./rec_inh[i]);
			}
			else {
				//weights_exc = Sample.randomUnitVector(rec_exc[i],rng);
				//weights_inh = Sample.randomUnitVector(rec_inh[i],rng);
				weights_exc = Sample.randomUnitVectorL1(rec_exc[i],rng);
				weights_inh = Sample.randomUnitVectorL1(rec_inh[i],rng);
			}
			int[] t = new int[target.get(i).size()];
			int[] s = new int[target.get(i).size()];
			for (int j=0; j<t.length; j++){
				t[j] = target.get(i).get(j);
				s[j] = syn_loc.get(i).get(j);
			}
			neurons.add(new SingleThresholdUnit(weights_exc, weights_inh, alpha_inh_exc, alpha_inh_inh, amp_inh_exc, amp_inh_inh, update_inh, adapt, false,t,s,0,max_rate));
			update_prob[i] = update_inh;
		}
		update_prob = VectorFun.cumsum(update_prob);
		VectorFun.normi(update_prob);
		avg_rates = new double[tot];
		avg_variance = new double[tot];
	}
	
	// update the image which is being shown to the network
	public void updateInput(final ImagePlus imp){
		current_im%=imp.getNSlices();
		double[] x = VectorConv.float2double((float[])(imp.getImageStack().getProcessor(current_im+1).convertToFloat().getPixels()));
		for (int i=0; i<w*h; i++){
			neurons.get(i).updateInput(true,x[i],0);
			neurons.get(i).project(neurons);
			neurons.get(w*h+i).updateInput(true,-x[i],0);
			neurons.get(w*h+i).project(neurons);
		}
		current_im++;
	}
	
	// iterate through n randomly choosen neurons in the population and update their weights as well
	public void iterate(int n, final Random rng){
		for (int i=0; i<n; i++){
			double r = rng.nextDouble();
			int a = Arrays.binarySearch(update_prob,r);
			if (a<0){
				a = -(a+1);
			}
			double b = neurons.get(a).project(neurons);
			avg_rates[a] = (avg_rates[a]*avg_window+b)/(avg_window+1);
			avg_variance[a] = (avg_variance[a]*avg_window+(b-avg_rates[a])*(b-avg_rates[a]))/(avg_window+1);
			//neurons.get(a).update_weights();
			neurons.get(a).update_weightsL1();
		}
	}
	
	// produce Image Stacks of the current network activity
	public ImagePlus getNeuronStates(){
		int l = w*h;
		int n = neurons.size()/l-2;
		double[][] x = new double[n][l];
		for (int i=0; i<n; i++){
			for (int j=0; j<l; j++){
				x[i][j] = avg_rates[(2+i)*l+j];
			}
		}
		return StackOperations.convert2Stack(x,w,h);
	}
	
	// produce Image Stacks of the current network activity variance
	public ImagePlus getNeuronStd(){
		int l = w*h;
		int n = neurons.size()/l-2;
		double[][] x = new double[n][l];
		for (int i=0; i<n; i++){
			for (int j=0; j<l; j++){
				x[i][j] = Math.sqrt(avg_variance[(2+i)*l+j]);
			}
		}
		return StackOperations.convert2Stack(x,w,h);
	}
	
	public ArrayList<ImagePlus> run(final ImagePlus imp, int numit, int update, int epoch, int output, int current_im){
		ArrayList<ImagePlus> implist = new ArrayList<ImagePlus>();
		Random rng = new Random();
		this.current_im = current_im;
		for (int i=0; i<numit; i++){
			if(i%update==0){
				updateInput(imp);
			}
			iterate(epoch,rng);
			if (i%output==0){
				implist.add(getNeuronStates());
			}
		}
		return implist;
	}
	
	public void setAlpha_exc(double alpha_exc){
		for (int i=w*h*2; i<neurons.size(); i++){
			neurons.get(i).setAlpha_exc(alpha_exc);
		}
	}
	
	public void setAlpha_inh(double alpha_inh){
		for (int i=w*h*2; i<neurons.size(); i++){
			neurons.get(i).setAlpha_inh(alpha_inh);
		}
	}
	
	public void setAverage_window(int window){
		this.avg_window = window;
	}
	
	public int[][] weightHistogram(int numbin){
		int[][] histo = new int[4][numbin];
		double[] temp;
		for (int i=w*h*2; i<neurons.size(); i++){
			// excitatory post synaptic neuron
			if (neurons.get(i).getType()){
				// excitatory in synapses
				temp = neurons.get(i).getWeights_exc();
				for (int j=0; j<temp.length; j++){
					if (temp[j]>=1){
						histo[0][numbin-1]++;
					}
					else{
						histo[0][(int)(temp[j]*numbin)]++;
					}
				}
				// inhibitory in synapses
				temp = neurons.get(i).getWeights_inh();
				for (int j=0; j<temp.length; j++){
					if (temp[j]>=1){
						histo[1][numbin-1]++;
					}
					else{
						histo[1][(int)(temp[j]*numbin)]++;
					}
				}
			}
			else {
				// excitatory in synapses
				temp = neurons.get(i).getWeights_exc();
				for (int j=0; j<temp.length; j++){
					if (temp[j]>=1){
						histo[2][numbin-1]++;
					}
					else{
						histo[2][(int)(temp[j]*numbin)]++;
					}
				}
				// inhibitory in synapses
				temp = neurons.get(i).getWeights_inh();
				for (int j=0; j<temp.length; j++){
					if (temp[j]>=1){
						histo[3][numbin-1]++;
					}
					else{
						histo[3][(int)(temp[j]*numbin)]++;
					}
				}
			}
		}
		return histo;
	}
	
	public ArrayList<SingleThresholdUnit> getNeurons(){
		return this.neurons;
	}
	
	public void reset(){
		for (int i=0; i<neurons.size(); i++){
			for (int j=0; j<neurons.get(i).getWeights_exc().length; j++){
				neurons.get(i).updateInput(true,0,j);
			}
			for (int j=0; j<neurons.get(i).getWeights_inh().length; j++){
				neurons.get(i).updateInput(false,0,j);
			}
		}
		avg_rates = new double[avg_rates.length];
		avg_variance = new double[avg_variance.length];
	}
	
	public double[] receptiveField(int numflip, int stabilizationit, int[] locations, final double[] im, double fact){
		double[] field = im.clone();
		double f,f2;
		IndexedTreeSet<Integer> on = new IndexedTreeSet<Integer>();
		IndexedTreeSet<Integer> off = new IndexedTreeSet<Integer>();
		int loc, loc2,a,b;
		Random rng = new Random();
		for (int i=0; i<im.length; i++){
			if (im[i]>0){
				on.add(i);
			}
			else{
				off.add(i);
			}
		}
		double factor = neurons.get(0).getAmp_exc()*fact;
		for (int i=0; i<2*w*h; i++){
			neurons.get(i).updateInput(true,field[i]*factor,0);
			neurons.get(i).project(neurons);
		}
		for (int i=0; i<numflip; i++){
			loc = rng.nextInt(on.size());
			loc2 = rng.nextInt(off.size());
			f = 0;
			for (int j=0; j<locations.length; j++){
				f+=avg_rates[2*w*h+locations[j]];
			}
			// flipping
			a = on.exact(loc);
			on.remove(a);
			b = off.exact(loc2);
			off.remove(b);
			on.add(b);
			off.add(a);
			field[a] = 0;
			field[b] = 1;
			// updating
			neurons.get(a).updateInput(true,field[a]*factor,0);
			neurons.get(a).project(neurons);
			neurons.get(b).updateInput(true,field[b]*factor,0);
			neurons.get(b).project(neurons);
			// use the updated input
			iterate(stabilizationit,rng);
			f2= 0;
			for (int j=0; j<locations.length; j++){
				f2+=avg_rates[2*w*h+locations[j]];
			}
			if (f2<f){
				// flipping
				on.remove(b);
				off.remove(a);
				on.add(a);
				off.add(b);
				field[a] = 1;
				field[b] = 0;
				// updating
				neurons.get(a).updateInput(true,field[a]*factor,0);
				neurons.get(a).project(neurons);
				neurons.get(b).updateInput(true,field[b]*factor,0);
				neurons.get(b).project(neurons);
				// reverse iterate
				iterate(stabilizationit,rng);
			}
		}
		return field;
	}

	public double[] receptiveField(int numflip, int stabilization, int[] loc, double ratio, double fact){
		double im[] = new double[w*h*2];
		Random rng = new Random();
		for (int i=0; i<im.length; i++){
			if (rng.nextDouble()<ratio){
				im[i] = 1;
			}
		}
		return receptiveField(numflip,stabilization,loc,im,fact);
	}
}