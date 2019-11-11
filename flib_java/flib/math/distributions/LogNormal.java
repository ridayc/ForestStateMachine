package flib.math.distributions;

import java.util.Random;
import java.lang.Math;

public class LogNormal {
	private Random rng;
	private double mu;
	private double sigma;
	
	public LogNormal(double m, double std){
		this.rng = new Random();
		this.mu = Math.log(m*m/Math.sqrt(std*std+m*m));
		this.sigma = Math.sqrt(Math.log(1+std*std/(m*m)));
	}
	
	public double generate(){
		return Math.exp(mu+sigma*rng.nextGaussian());
	}
	
	public double generate(double m){
		double a = Math.log(m)-0.5*sigma*sigma;
		return Math.exp(a+sigma*rng.nextGaussian());
	}
	
	public double generate(double m, double std){
		double a = Math.log(m*m/Math.sqrt(std*std+m*m));
		double b = Math.sqrt(Math.log(1+std*std/(m*m)));
		return Math.exp(a+b*rng.nextGaussian());
	}
	
	public void set(double m, double std){
		this.mu = Math.log(m*m/Math.sqrt(std*std+m*m));
		this.sigma = Math.sqrt(Math.log(1+std*std/(m*m)));
	}
}