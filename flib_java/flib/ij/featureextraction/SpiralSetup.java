package flib.ij.featureextraction;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import flib.io.ReadWrite;
import flib.ij.featureextraction.FolderNames;
import flib.algorithms.sampling.NeighborhoodSample;

public class SpiralSetup {
	public static void run(String orig, String targetdir, final double[] parameters){
		// make sure the storage folder exists
		new File(targetdir).mkdirs();
		// store the name of the origin directory
		ReadWrite.writeObject(targetdir+File.separator+FolderNames.SPIRALORIG,orig);
		// store the parameters for the spiral
		ReadWrite.writeObject(targetdir+File.separator+FolderNames.SPIRALPARAM,parameters);
		String name = targetdir+File.separator+"config.txt";
		// write a config file
		writeConfigTXT(name,orig,parameters);
	}

	private static void writeConfigTXT(final String name, final String orig, final double[] parameters){
		try {
			File infotxt = new File(name);
			BufferedWriter writer = new BufferedWriter(new FileWriter(infotxt));
			String newline = System.getProperty("line.separator");
			writer.write("Spiral parameters"+newline+
				"Intial cutoff radius: "+String.format("%.2f",parameters[0])+newline+
				"Radial increment multiplier: "+String.format("%.2f",parameters[1])+newline+
				"Angular increment: "+String.format("%.2f",parameters[2])+newline+
				"Fill parameter: "+String.format("%d",(int)parameters[3])+newline+
				"Spatial Coverage exponent: "+String.format("%.2f",parameters[4])+newline+
				"Angular exponent: "+String.format("%.2f",parameters[5])+newline+
				"Number of arms: "+String.format("%d",(int)parameters[6])+newline+
				"Sample origin directory: "+orig);
			writer.close();
		}
		catch (Exception e){}
	}
}