package flib.neurons.spiking;

public class SynapticEvent implements
java.io.Serializable, Comparable<SynapticEvent> {
	public double time;
	public double identity;
	public int type;
	public int target_module;
	public int target_neuron;
	public double target_synapse;
	
	public SynapticEvent(double time, double identity, int type, int target_module, int target_neuron, double target_synapse){
		this.time = time;
		this.identity = identity;
		this.type = type;
		this.target_module = target_module;
		this.target_neuron = target_neuron;
		this.target_synapse = target_synapse;
	}
	
	public SynapticEvent(final SynapticEvent o){
		this.time = o.time;
		this.identity = o.identity;
		this.type = o.type;
		this.target_module = o.target_module;
		this.target_neuron = o.target_neuron;
		this.target_synapse = o.target_synapse;
	}
	
	@Override public int compareTo(SynapticEvent o){
		int a = Double.compare(time,o.time);
		if (a==0){
			a = Double.compare(identity, o.identity);
		}
		return a;
	}
}
	