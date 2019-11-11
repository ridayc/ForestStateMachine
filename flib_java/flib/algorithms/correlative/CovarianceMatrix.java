package flib.algorithms.correlative;

public class CovarianceMatrix {
	public static double[][] covariance(final double[][] x){
		int dim = x.length;
		int n = x[0].length;
		double[][] cov = new double[dim][dim];
		double[] m = mean(x);
		for (int i=0; i<dim; i++){
			for (int j=0; j<dim; j++){
				for (int k=0; k<n; k++){
					cov[i][j]+=x[i][k]*x[j][k];
				}
				cov[i][j]/=n;
				cov[i][j]-=m[i]*m[j];
			}
		}
		return cov;	
	}
	
	public static double[] mean(final double[][] x){
		int dim = x.length;
		int n = x[0].length;
		double[] m = new double[dim];
		for (int i=0; i<dim; i++){
			for (int j=0; j<n; j++){
				m[i]+=x[i][j];
			}
			m[i]/=n;
		}
		return m;
	}
	
	public static double[] std(final double[][] x){
		int dim = x.length;
		int n = x[0].length;
		double[] m = mean(x);
		double[] m2 = new double[dim];
		for (int i=0; i<dim; i++){
			for (int j=0; j<n; j++){
				m2[i]+=x[i][j]*x[i][j];
			}
			m2[i]/=n;
			m2[i]-=m[i]*m[i];
		}
		return m2;
	}
	
	public static void sum(double[] m, double[] x){
		for (int i=0; i<m.length; i++){
			m[i]+=x[i];
		}
	}
	
	public static void sum2(double[][] cov, double[] m, double[] x){
		// we assume that cov is symmetric
		for (int i=0; i<cov.length; i++){
			for (int j=0; j<cov.length; j++){
				cov[i][j]+=x[i]*x[j];
			}
			m[i]+=x[i];
		}
	}
	
	public static void mean(double[] m, int n){
		for (int i=0; i<m.length; i++){
			m[i]/=n;
		}
	}
	
	public static void covariance(double[][] cov, double[] m, int n){
		// we assume that cov is symmetric
		for (int i=0; i<cov.length; i++){
			for (int j=0; j<cov.length; j++){
				cov[i][j] = cov[i][j]/n-m[i]*m[j];
			}
		}
	}
	
	public static double[][] outerProduct(final double[] x, final double[] y){
		int dim = x.length;
		double[][] z = new double[dim][dim];
		for (int i=0; i<dim; i++){
			for (int j=0; j<dim; j++){
				z[i][j] = x[i]*y[j];
			}
		}
		return z;
	}
}
	